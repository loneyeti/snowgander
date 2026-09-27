import Anthropic from "@anthropic-ai/sdk";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  ModelConfig,
  Chat,
  ChatResponse,
  ContentBlock,
  ToolUseBlock,
  Message,
  // MCPTool, // MCPTool type might not be needed if sendMCPChat is removed/integrated
  UsageResponse,
  ImageBlock,
  ImageDataBlock,
  NotImplementedError, // Import error type
  ImageGenerationResponse, // Import response type
  Citation,
  WebSearchOptions,
} from "../types";
import { computeResponseCost } from "../utils";
// Import necessary types from Anthropic SDK
import {
  Tool as AnthropicTool,
  ToolUseBlockParam,
  ServerToolUseBlockParam,
  WebSearchToolResultBlockParam,
  WebFetchToolResultBlockParam,
} from "@anthropic-ai/sdk/resources/messages";
// Removed Prisma Model import
// Removed application-specific imports (updateUserUsage, getCurrentAPIUser)

// --- Model capability tiers ---
//
// Anthropic's request shape for thinking/effort/sampling parameters differs across
// model generations. These two independent classifications drive that behavior:
//
// - Thinking tier: which thinking API a model speaks (legacy budget_tokens vs
//   adaptive), and which effort levels are meaningful.
// - Sampling policy: how many of temperature/top_p a model accepts. This is NOT
//   1:1 with the thinking tier — e.g. Sonnet 4.5 and Haiku 4.5 use legacy
//   budget_tokens thinking, but like every Claude 4+ model they reject sending
//   both temperature and top_p together.
type ThinkingTier = "legacyBudget" | "adaptiveTransitional" | "adaptiveOnly";
type SamplingPolicy = "both" | "single" | "none";

// Anthropic content-block param shapes used when formatting our messages for the
// request. Kept local to this file since these are request-only shapes.
type FormattedAnthropicContentBlock =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "thinking";
      thinking: string;
      signature: string;
    }
  | {
      type: "redacted_thinking";
      data: string;
    }
  | {
      type: "tool_result";
      tool_use_id: string;
      content: string | Array<{ type: "text"; text: string }>;
    }
  | ToolUseBlockParam // Add Anthropic's ToolUseBlockParam type
  | ServerToolUseBlockParam
  // Echoed verbatim from a previous response (encrypted_content etc.)
  | WebSearchToolResultBlockParam
  | WebFetchToolResultBlockParam;

// Server tools stop with pause_turn after about 10 iterations. The adapter
// sends the paused turn back to continue, up to this many times per request.
const MAX_PAUSE_CONTINUATIONS = 3;

type FormattedAnthropicMessage = {
  role: "user" | "assistant";
  content: string | FormattedAnthropicContentBlock[];
};

export class AnthropicAdapter implements AIVendorAdapter {
  private client: Anthropic;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;
  private webSearchCost?: number;

  // Constructor now accepts ModelConfig instead of Prisma Model
  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.client = new Anthropic({
      apiKey: config.apiKey,
      baseURL: config.baseURL, // Allow overriding base URL
    });

    // Use fields from modelConfig
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;
    this.webSearchCost = modelConfig.webSearchCost;

    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  /**
   * Classify a model into a thinking-capability tier. Checks are ordered
   * most-specific-first so newer model families aren't accidentally caught by a
   * broader/older pattern (e.g. "claude-opus-4-8" must be checked before a bare
   * "claude-opus-4" fallback would ever be considered).
   *
   * Unknown/future model strings fall back to "legacyBudget" — the behavior every
   * Claude model has supported since thinking was introduced, and the safest
   * default until this table is updated for a new release.
   *
   * @param modelName - The model identifier (e.g., "claude-opus-4-8")
   */
  private getThinkingTier(modelName: string): ThinkingTier {
    // Adaptive-only: thinking is always adaptive, budget_tokens is rejected (400),
    // and effort supports the full low/medium/high/xhigh/max range.
    if (
      modelName.includes("claude-opus-4-7") ||
      modelName.includes("claude-opus-4-8") ||
      modelName.includes("claude-opus-5") || // Opus 5 and Opus 5.5
      modelName.includes("claude-sonnet-5") ||
      modelName.includes("claude-fable-5") ||
      modelName.includes("claude-mythos-5")
    ) {
      return "adaptiveOnly";
    }

    // Adaptive-transitional: adaptive thinking is supported and recommended;
    // effort supports low/medium/high/max (no xhigh).
    if (
      modelName.includes("claude-opus-4-6") ||
      modelName.includes("claude-sonnet-4-6")
    ) {
      return "adaptiveTransitional";
    }

    // Legacy: Claude 3.x, Opus 4.0/4.1/4.5, Sonnet 4.0/4.5, Haiku 3.x/4.5, and any
    // unrecognized/future model string. Thinking (if supported) uses
    // {type: "enabled", budget_tokens}; no output_config/effort support.
    return "legacyBudget";
  }

  /**
   * Classify a model's sampling-parameter (temperature/top_p) constraints.
   * Independent from the thinking tier — see the type comment above.
   *
   * @param modelName - The model identifier (e.g., "claude-opus-4-8")
   */
  private getSamplingPolicy(modelName: string): SamplingPolicy {
    // No Claude 4+ model accepts sampling parameters once it's adaptive-only.
    if (this.getThinkingTier(modelName) === "adaptiveOnly") {
      return "none";
    }

    // Claude 2.x / 3.x models accept temperature and top_p together.
    if (modelName.includes("claude-3") || modelName.includes("claude-2")) {
      return "both";
    }

    // Every other Claude 4.x-family model (4.0/4.1/4.5, the 4.6 family, Sonnet
    // 4.5, Haiku 4.5, etc.) accepts at most one sampling parameter.
    return "single";
  }

  /**
   * Opus 4.0 / 4.1 get special handling: a lower output ceiling, and the SDK
   * refuses non-streaming requests above 8192 max_tokens for them.
   */
  private isOpus40or41(modelName: string): boolean {
    return (
      /claude-opus-4-(0|1|20\d{6})/.test(modelName) ||
      modelName.includes("claude-4-opus")
    );
  }

  /**
   * Maximum output tokens (thinking + text) the model accepts for max_tokens.
   * Unknown/future models get 64000, which every current model supports.
   */
  private getMaxOutputTokens(modelName: string): number {
    if (
      modelName.includes("claude-3-opus") ||
      modelName.includes("claude-3-haiku") ||
      modelName.includes("claude-3-sonnet")
    ) {
      return 4096;
    }
    if (modelName.includes("claude-3-5")) return 8192;
    if (this.isOpus40or41(modelName)) return 32000;
    return 64000;
  }

  /**
   * Default max_tokens used when the caller doesn't set one. max_tokens is a
   * hard cap on thinking + text combined, so a small default silently
   * truncates long answers. It's only a ceiling (billing is per token
   * actually produced), so we default high:
   * - streaming: 64000 (Anthropic's recommended starting point for xhigh/max effort)
   * - non-streaming: 16000, because the SDK throws for non-streaming requests
   *   whose max_tokens implies >10 minutes (~21.3k tokens; 8192 for Opus 4.0/4.1)
   * Both are clamped to the model's max output.
   */
  private getDefaultMaxTokens(modelName: string, streaming: boolean): number {
    let limit = Math.min(
      streaming ? 64000 : 16000,
      this.getMaxOutputTokens(modelName)
    );
    if (!streaming && this.isOpus40or41(modelName)) {
      limit = Math.min(limit, 8192);
    }
    return limit;
  }

  /**
   * Map budgetTokens to an effort level for adaptive-thinking tiers, used only
   * when the caller hasn't provided an explicit `effort`. This heuristic only
   * reaches low/medium/high — "xhigh" and "max" are only reachable via an
   * explicit `effort` value, since there's no principled budget-token threshold
   * to derive them from.
   * @param budgetTokens - Token budget for thinking mode
   * @returns Effort level (low/medium/high)
   */
  private mapBudgetToEffort(budgetTokens?: number): "low" | "medium" | "high" {
    if (!budgetTokens || budgetTokens === 0) return "low";
    if (budgetTokens < 8192) return "medium";
    return "high";
  }

  /**
   * Validates temperature/top_p against the model's sampling policy and returns
   * the fragment to merge into the request params. Throws a descriptive error
   * when the combination is invalid for this model, rather than letting an
   * incompatible request reach the API as an opaque 400.
   */
  private buildSamplingParams(
    model: string,
    temperature: number | undefined,
    topP: number | undefined
  ): { temperature?: number; top_p?: number } {
    const policy = this.getSamplingPolicy(model);

    if (policy === "none") {
      if (temperature !== undefined || topP !== undefined) {
        throw new Error(
          `Model '${model}' does not support sampling parameters (temperature, top_p). ` +
            `Remove these parameters when using this model.`
        );
      }
      return {};
    }

    if (policy === "single" && temperature !== undefined && topP !== undefined) {
      throw new Error(
        `Models like '${model}' do not support using both temperature and top_p. ` +
          `Please provide only one sampling parameter.`
      );
    }

    const params: { temperature?: number; top_p?: number } = {};
    if (temperature !== undefined) {
      params.temperature = temperature;
    }
    if (topP !== undefined) {
      params.top_p = topP;
    }
    return params;
  }

  /**
   * Whether the model runs adaptive thinking even when the request has no
   * `thinking` field. Opus 4.7/4.8 are adaptive-only but think only when asked;
   * the Claude 5 family thinks by default (Opus 5.5 can't disable it at all).
   */
  private thinksByDefault(modelName: string): boolean {
    return (
      this.getThinkingTier(modelName) === "adaptiveOnly" &&
      !modelName.includes("claude-opus-4-7") &&
      !modelName.includes("claude-opus-4-8")
    );
  }

  /**
   * Normalize the shared `effort` option (which also carries OpenAI-only
   * values) to a level Anthropic accepts for this tier. Returns undefined when
   * no effort should be sent. "none" means "don't think": on the 4.6 tier that's
   * expressed by omitting thinking, but Claude 5-family models always think, so
   * the closest valid request is "low".
   */
  private normalizeEffort(
    tier: ThinkingTier,
    effort: AIRequestOptions["effort"]
  ): "low" | "medium" | "high" | "xhigh" | "max" | undefined {
    if (!effort) return undefined;
    if (effort === "minimal") return "low";
    if (effort === "none") return tier === "adaptiveOnly" ? "low" : undefined;
    // The 4.6 family has no xhigh level.
    if (effort === "xhigh" && tier === "adaptiveTransitional") return "high";
    return effort;
  }

  /**
   * Builds the `thinking` / `output_config` request fragment for a model.
   * Shared between generateResponse and streamResponse.
   */
  private buildThinkingParams(
    model: string,
    thinkingMode: boolean | undefined,
    budgetTokens: number | undefined,
    maxTokens: number,
    effort: AIRequestOptions["effort"],
    outputFormat: any,
    thinkingDisplay: AIRequestOptions["thinkingDisplay"]
  ): { thinking?: any; output_config?: any } {
    const tier = this.getThinkingTier(model);
    const result: { thinking?: any; output_config?: any } = {};

    if (tier === "legacyBudget") {
      if (thinkingMode && this.isThinkingCapable) {
        // Use legacy budget_tokens for models without adaptive thinking support.
        // budget_tokens must be < max_tokens; reserve at least 1024 tokens for
        // the answer so thinking can't consume the whole output (1024 is also
        // the API's minimum budget).
        const maxBudget = Math.max(1024, maxTokens - 1024);
        result.thinking = {
          type: "enabled",
          budget_tokens: Math.min(
            budgetTokens || Math.floor(maxTokens / 2),
            maxBudget
          ),
        };
      }
    } else {
      const wantsThinking =
        !!thinkingMode && this.isThinkingCapable && effort !== "none";

      // adaptiveOnly models default thinking.display to "omitted" (empty
      // thinking text); default to "summarized" so reasoning and the progress
      // notes written between tool calls stay visible. The 4.6 family already
      // defaults to summarized, so only send display there when asked.
      const display =
        thinkingDisplay ?? (tier === "adaptiveOnly" ? "summarized" : undefined);

      // Send thinking when requested, or when the model thinks regardless and
      // we need to set its display.
      if (wantsThinking || (this.thinksByDefault(model) && display)) {
        result.thinking = {
          type: "adaptive",
          ...(display && { display }),
        };
      }

      // Effort applies whether or not thinking was requested (on the Claude 5
      // family it's the only thinking control). Without an explicit effort,
      // derive one from budgetTokens when thinking was requested.
      const resolvedEffort =
        this.normalizeEffort(tier, effort) ??
        (wantsThinking && budgetTokens !== undefined
          ? this.mapBudgetToEffort(budgetTokens)
          : undefined);
      if (resolvedEffort) {
        result.output_config = { effort: resolvedEffort };
      }
    }

    // Structured output support is available on adaptive-capable tiers.
    if (outputFormat && tier !== "legacyBudget") {
      result.output_config = {
        ...result.output_config,
        format: outputFormat,
      };
    }

    return result;
  }

  /**
   * ErrorBlock for stop reasons the caller must not miss (refusal, max_tokens,
   * model_context_window_exceeded), or undefined for normal stops. Shared
   * between generateResponse and streamResponse.
   */
  private buildStopReasonError(
    stopReason: string | null | undefined,
    stopDetails: { category?: string | null } | null | undefined,
    maxTokens: number
  ): ContentBlock | undefined {
    if (stopReason === "refusal") {
      const category = stopDetails?.category;
      console.warn(
        `Model refused to respond to this request${
          category ? ` (category: ${category})` : ""
        }`
      );
      return {
        type: "error",
        publicMessage: "The model declined to respond to this request.",
        privateMessage: `Stop reason: ${stopReason}${
          category ? `, category: ${category}` : ""
        }`,
      };
    }
    if (stopReason === "max_tokens") {
      console.warn(`Response truncated: hit max_tokens limit (${maxTokens})`);
      return {
        type: "error",
        publicMessage:
          "Response was truncated because it hit the max token limit.",
        privateMessage: `Stop reason: max_tokens (max_tokens: ${maxTokens})`,
      };
    }
    if (stopReason === "pause_turn") {
      // Only reached once MAX_PAUSE_CONTINUATIONS is used up
      console.warn("Response still paused after the continuation limit");
      return {
        type: "error",
        publicMessage:
          "The research step limit was reached before the answer finished.",
        privateMessage: `Stop reason: pause_turn after ${MAX_PAUSE_CONTINUATIONS} continuations`,
      };
    }
    if (stopReason === "model_context_window_exceeded") {
      console.warn("Response stopped due to context window limit");
      return {
        type: "error",
        publicMessage: "Response stopped due to context limit.",
        privateMessage: `Stop reason: ${stopReason}`,
      };
    }
    return undefined;
  }

  /**
   * Builds the server-side web_search (and optionally web_fetch) tools. Models
   * on an adaptive thinking tier (4.6+, Claude 5 family) get the _20260209
   * variants with dynamic filtering; older models get the basic variants.
   */
  private buildWebTools(model: string, opts: WebSearchOptions): any[] {
    const current = this.getThinkingTier(model) !== "legacyBudget";
    // The API accepts allowed_domains or blocked_domains, never both
    let domains: Record<string, string[]> = {};
    if (opts.allowedDomains?.length) {
      domains = { allowed_domains: opts.allowedDomains };
      if (opts.blockedDomains?.length) {
        console.warn(
          "webSearch.allowedDomains and blockedDomains are exclusive; ignoring blockedDomains."
        );
      }
    } else if (opts.blockedDomains?.length) {
      domains = { blocked_domains: opts.blockedDomains };
    }
    const maxUses = opts.maxUses !== undefined ? { max_uses: opts.maxUses } : {};

    const loc = opts.userLocation;
    const search: any = {
      type: current ? "web_search_20260209" : "web_search_20250305",
      name: "web_search",
      ...maxUses,
      ...domains,
    };
    if (loc && (loc.city || loc.region || loc.country || loc.timezone)) {
      search.user_location = {
        type: "approximate",
        ...(loc.city && { city: loc.city }),
        ...(loc.region && { region: loc.region }),
        ...(loc.country && { country: loc.country }),
        ...(loc.timezone && { timezone: loc.timezone }),
      };
    }
    const tools = [search];
    if (opts.fetch) {
      tools.push({
        type: current ? "web_fetch_20260209" : "web_fetch_20250910",
        name: "web_fetch",
        ...maxUses,
        ...domains,
        citations: { enabled: true },
      });
    }
    return tools;
  }

  /**
   * Caller tools plus the web tools requested via `webSearch`. A caller tool
   * with the same name wins.
   */
  private buildRequestTools(
    model: string,
    tools: any[] | undefined,
    webSearch: WebSearchOptions | undefined
  ): any[] | undefined {
    if (!webSearch) return tools;
    const existing = new Set((tools ?? []).map((tool) => tool?.name));
    const webTools = this.buildWebTools(model, webSearch).filter(
      (tool) => !existing.has(tool.name)
    );
    return [...(tools ?? []), ...webTools];
  }

  /**
   * Normalizes an Anthropic citation. Web search citations carry a URL.
   * Citations into fetched documents carry only a document title, so they are
   * resolved against the documents fetched earlier in the same response.
   */
  private mapCitation(
    citation: any,
    fetchedDocs: Map<string, string>
  ): Citation | null {
    let url: string | undefined = citation?.url;
    const title: string | undefined =
      citation?.title ?? citation?.document_title ?? undefined;
    if (!url && title) {
      url = fetchedDocs.get(title);
    }
    if (!url) return null;
    return {
      type: "url_citation",
      url,
      ...(title && { title }),
      ...(citation.cited_text && { citedText: citation.cited_text }),
      vendor: "anthropic",
      raw: citation,
    };
  }

  /**
   * Maps a server tool result block from the API to our normalized block.
   * Records fetched document titles in `fetchedDocs` for citation lookup.
   */
  private mapServerToolResult(
    block: any,
    fetchedDocs: Map<string, string>
  ): ContentBlock | null {
    if (block.type === "web_search_tool_result") {
      const content = block.content;
      // Errors return HTTP 200 with a single error object instead of a list
      if (!Array.isArray(content)) {
        return {
          type: "web_search_tool_result",
          toolUseId: block.tool_use_id,
          results: [],
          errorCode: content?.error_code ?? "unknown",
          vendor: "anthropic",
          raw: block,
        };
      }
      return {
        type: "web_search_tool_result",
        toolUseId: block.tool_use_id,
        results: content
          .filter((result: any) => result?.type === "web_search_result")
          .map((result: any) => ({
            url: result.url,
            ...(result.title && { title: result.title }),
            ...(result.page_age !== undefined && { pageAge: result.page_age }),
          })),
        vendor: "anthropic",
        raw: block,
      };
    }
    if (block.type === "web_fetch_tool_result") {
      const content = block.content;
      if (content?.type !== "web_fetch_result") {
        return {
          type: "web_fetch_tool_result",
          toolUseId: block.tool_use_id,
          errorCode: content?.error_code ?? "unknown",
          vendor: "anthropic",
          raw: block,
        };
      }
      const title: string | undefined = content.content?.title ?? undefined;
      if (title && content.url) {
        fetchedDocs.set(title, content.url);
      }
      return {
        type: "web_fetch_tool_result",
        toolUseId: block.tool_use_id,
        url: content.url,
        ...(title && { title }),
        retrievedAt: content.retrieved_at ?? null,
        vendor: "anthropic",
        raw: block,
      };
    }
    return null;
  }

  /** Web search fee for a request: the model's per-search cost x searches. */
  private computeWebSearchFee(searches: number): number {
    return (this.webSearchCost ?? 0) * searches;
  }

  /**
   * Claude Opus 4.6+ and the Claude 5 family reject a request whose last
   * message is an assistant turn (prefill). Throw a descriptive error up front
   * rather than letting it reach the API as an opaque 400.
   */
  private assertNoPrefill(model: string, messages: Message[]): void {
    const last = messages[messages.length - 1];
    if (
      last?.role === "assistant" &&
      this.getThinkingTier(model) !== "legacyBudget"
    ) {
      throw new Error(
        `Model '${model}' does not support assistant message prefill. ` +
          `End the conversation with a user message; use outputFormat or the system prompt to shape the response.`
      );
    }
  }

  /**
   * Converts our Message[]/ContentBlock[] format into Anthropic's request
   * message shape. Shared between generateResponse and streamResponse so the
   * two methods can't drift out of sync with each other.
   */
  private formatMessages(
    messages: Message[],
    model: string
  ): FormattedAnthropicMessage[] {
    return messages.map((msg) => {
      const role =
        msg.role === "assistant" ? ("assistant" as const) : ("user" as const);

      if (typeof msg.content === "string") {
        return { role, content: msg.content };
      }

      // A server_tool_use block must be followed by its result, so only echo
      // Claude-origin pairs where both halves are present.
      const serverToolUseIds = new Set<string>();
      const serverToolResultIds = new Set<string>();
      if (msg.role === "assistant") {
        for (const block of msg.content) {
          if (block.type === "server_tool_use" && block.vendor === "anthropic") {
            serverToolUseIds.add(block.id);
          } else if (
            (block.type === "web_search_tool_result" ||
              block.type === "web_fetch_tool_result") &&
            block.vendor === "anthropic" &&
            block.raw
          ) {
            serverToolResultIds.add(block.toolUseId);
          }
        }
      }

      // Map our content blocks to Anthropic's ContentBlockParam format
      const mappedContent = msg.content.reduce<FormattedAnthropicContentBlock[]>(
        (acc, block) => {
          if (
            block.type === "server_tool_use" ||
            block.type === "web_search_tool_result" ||
            block.type === "web_fetch_tool_result"
          ) {
            const id =
              block.type === "server_tool_use" ? block.id : block.toolUseId;
            if (!serverToolUseIds.has(id) || !serverToolResultIds.has(id)) {
              return acc; // OpenAI-origin, or an unpaired half
            }
            if (block.type === "server_tool_use") {
              let input: unknown = {};
              try {
                input = JSON.parse(block.input);
              } catch {
                console.warn(
                  `Server tool input was not valid JSON: ${block.input}`
                );
              }
              acc.push({
                type: "server_tool_use",
                id: block.id,
                name: block.name as ServerToolUseBlockParam["name"],
                input,
              });
            } else {
              acc.push(block.raw as FormattedAnthropicContentBlock);
            }
            return acc;
          }
          if (block.type === "text") {
            acc.push({
              type: "text",
              text: block.text,
            });
          } else if (block.type === "thinking") {
            // Map thinking block - Although Anthropic doesn't accept these in requests,
            // the test expects the formatting logic to handle them.
            acc.push({
              type: "thinking",
              thinking: block.thinking,
              signature: block.signature, // Assuming signature maps directly, adjust if needed
            });
          } else if (block.type === "redacted_thinking") {
            // Must be echoed back unmodified alongside thinking blocks in tool loops.
            acc.push({ type: "redacted_thinking", data: block.data });
          } else if (block.type === "tool_result" && msg.role === "user") {
            // Handle tool_result specifically for user messages
            // Anthropic expects tool_result content to be a string or an array of text blocks.
            // We'll convert our ContentBlock[] to a simple string for now.
            // TODO: Improve this to handle potential multiple text blocks in tool_result.content if needed.
            const toolResultContent = block.content
              .map((c) => (c.type === "text" ? c.text : JSON.stringify(c)))
              .join("\n");

            acc.push({
              type: "tool_result",
              tool_use_id: block.toolUseId, // Use the string ID directly
              content: toolResultContent, // Send as string for simplicity
              // Alternatively, format as [{ type: 'text', text: toolResultContent }] if preferred
            });
          } else if (block.type === "tool_use" && msg.role === "assistant") {
            // Map tool_use blocks from assistant history correctly for the API request
            // Ensure the ID exists and input is valid JSON
            if (block.id && typeof block.input === "string") {
              try {
                const parsedInput = JSON.parse(block.input);
                acc.push({
                  type: "tool_use",
                  id: block.id,
                  name: block.name,
                  input: parsedInput, // Anthropic expects the parsed object here
                });
              } catch (e) {
                console.error(
                  `Skipping tool_use block due to invalid JSON input: ${block.input}`,
                  e
                );
              }
            } else {
              console.warn(
                "Skipping tool_use block due to missing ID or invalid input type."
              );
            }
          } else if (
            block.type === "image" &&
            this.isVisionCapable &&
            msg.role === "user"
          ) {
            // Handle ImageBlock (URL) for vision-capable models in user messages
            // Map to Anthropic's expected format based on their documentation example
            acc.push({
              type: "image",
              source: {
                type: "url",
                // Ensure 'url' property exists on our ImageBlock type
                url: block.url,
              },
            } as any); // Use 'as any' for now if the SDK type doesn't perfectly align or needs broader compatibility
          } else if (
            block.type === "image_data" &&
            this.isVisionCapable &&
            msg.role === "user"
          ) {
            // Handle ImageDataBlock (base64) - Anthropic example only shows URL source.
            // Warn the developer, as this might not be directly supported or requires different formatting.
            console.warn(
              "Anthropic adapter received ImageDataBlock (base64). Anthropic API example uses URL source. Skipping image. Check Anthropic documentation for base64 support."
            );
            // If Anthropic *did* support base64 via a specific format (e.g., type: 'base64'), the mapping would go here.
            // Example (hypothetical, verify with docs):
            // acc.push({
            //   type: "image",
            //   source: {
            //     type: "base64",
            //     media_type: block.mimeType,
            //     data: block.base64Data,
            //   }
            // } as any);
          } else if (
            (block.type === "image" || block.type === "image_data") &&
            (!this.isVisionCapable || msg.role !== "user")
          ) {
            // Skip images if model isn't vision capable or if not in a user message
            console.warn(
              `Skipping image block (type: ${block.type}) in role '${msg.role}' for model '${model}'. Vision capable: ${this.isVisionCapable}`
            );
          }
          // Note: Image blocks are now handled above.
          return acc;
        },
        []
      );

      // If we have no valid content blocks, convert to a text block with the stringified content
      return {
        role,
        content:
          mappedContent.length > 0
            ? mappedContent
            : JSON.stringify(msg.content),
      };
    });
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    // Removed user fetching logic
    const {
      model,
      messages,
      maxTokens,
      budgetTokens,
      systemPrompt,
      thinkingMode,
      tools, // Destructure tools from options
      effort,
      outputFormat,
      temperature,
      topP,
      thinkingDisplay,
    } = options;

    // Validate sampling parameters (throws a descriptive error for
    // incompatible model/parameter combinations rather than an opaque 400)
    const samplingParams = this.buildSamplingParams(model, temperature, topP);

    this.assertNoPrefill(model, messages);

    // Convert messages to Anthropic format
    const formattedMessages = this.formatMessages(messages, model);

    const resolvedMaxTokens =
      maxTokens || this.getDefaultMaxTokens(model, false);

    // Build thinking / output_config request fragment
    const thinkingParams = this.buildThinkingParams(
      model,
      thinkingMode,
      budgetTokens,
      resolvedMaxTokens,
      effort,
      outputFormat,
      thinkingDisplay
    );

    // Build request parameters based on model tier
    const requestParams: any = {
      model,
      messages: formattedMessages,
      system: systemPrompt,
      max_tokens: resolvedMaxTokens,
      ...samplingParams,
      ...thinkingParams,
    };

    // Add tools if provided
    const requestTools = this.buildRequestTools(
      model,
      tools,
      options.webSearch
    );
    if (requestTools) {
      requestParams.tools = requestTools;
    }

    let response = await this.client.messages.create(requestParams);
    // Server tools can pause a long turn. Send the paused content back to
    // continue, and collect every round's content and usage.
    const responseContent: any[] = [...response.content];
    let inputTokens = response.usage.input_tokens;
    let outputTokens = response.usage.output_tokens;
    let webSearches = response.usage.server_tool_use?.web_search_requests ?? 0;
    let webFetches = response.usage.server_tool_use?.web_fetch_requests ?? 0;
    for (
      let round = 0;
      response.stop_reason === "pause_turn" && round < MAX_PAUSE_CONTINUATIONS;
      round++
    ) {
      requestParams.messages = [
        ...requestParams.messages,
        { role: "assistant", content: response.content },
      ];
      response = await this.client.messages.create(requestParams);
      responseContent.push(...response.content);
      inputTokens += response.usage.input_tokens;
      outputTokens += response.usage.output_tokens;
      webSearches += response.usage.server_tool_use?.web_search_requests ?? 0;
      webFetches += response.usage.server_tool_use?.web_fetch_requests ?? 0;
    }

    let usage: UsageResponse | undefined = undefined;

    if (
      inputTokens &&
      outputTokens &&
      this.inputTokenCost &&
      this.outputTokenCost
    ) {
      const inputCost = computeResponseCost(inputTokens, this.inputTokenCost);
      const outputCost = computeResponseCost(
        outputTokens,
        this.outputTokenCost
      );
      const webSearchCost = this.computeWebSearchFee(webSearches);
      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        ...(webSearchCost > 0 && { webSearchCost }),
        ...(webSearches > 0 && {
          didWebSearch: true,
          webSearchCount: webSearches,
        }),
        ...(webFetches > 0 && { webFetchCount: webFetches }),
        totalCost: inputCost + outputCost + webSearchCost,
      };
    }

    // Convert Anthropic response blocks to our ContentBlock format
    const contentBlocks: ContentBlock[] = [];
    const fetchedDocs = new Map<string, string>();

    for (const block of responseContent) {
      if (block.type === "thinking") {
        contentBlocks.push({
          type: "thinking",
          thinking: block.thinking,
          signature: block.signature,
        });
      } else if (block.type === "redacted_thinking") {
        contentBlocks.push({ type: "redacted_thinking", data: block.data });
      } else if (block.type === "text") {
        const citations = (block.citations ?? [])
          .map((citation: any) => this.mapCitation(citation, fetchedDocs))
          .filter((citation: Citation | null): citation is Citation => !!citation);
        contentBlocks.push({
          type: "text",
          text: block.text,
          ...(citations.length > 0 && { citations }),
        });
      } else if (block.type === "server_tool_use") {
        contentBlocks.push({
          type: "server_tool_use",
          id: block.id,
          name: block.name,
          input: JSON.stringify(block.input ?? {}),
          status: "completed",
          vendor: "anthropic",
        });
      } else if (
        block.type === "web_search_tool_result" ||
        block.type === "web_fetch_tool_result"
      ) {
        const mapped = this.mapServerToolResult(block, fetchedDocs);
        if (mapped) contentBlocks.push(mapped);
      } else if (block.type === "tool_use") {
        // Map Anthropic tool_use to our ToolUseBlock, including the ID
        contentBlocks.push({
          type: "tool_use",
          id: block.id, // Capture the tool use ID from Anthropic
          name: block.name,
          input: JSON.stringify(block.input), // Stringify the input object
        });
      }
      // Skip any other unknown block types
    }

    // Surface refusals, truncation, and context-window stops
    const stopError = this.buildStopReasonError(
      response.stop_reason,
      response.stop_details,
      resolvedMaxTokens
    );
    if (stopError) {
      contentBlocks.push(stopError);
    }

    // Removed usage calculation and user update logic
    // The calling application should handle cost calculation.
    // const usage = response.usage; // Could potentially return usage if needed

    return {
      role: "assistant",
      content: contentBlocks,
      usage: usage,
    };
  }

  // Updated signature to match AIVendorAdapter interface
  async generateImage(options: AIRequestOptions): Promise<AIResponse> {
    // Throw NotImplementedError as Anthropic doesn't support image generation
    throw new NotImplementedError(
      "Image generation not supported by Anthropic"
    );
  }

  // snowgander/src/vendors/anthropic.ts

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const {
      model,
      messages,
      maxTokens,
      budgetTokens,
      systemPrompt,
      thinkingMode,
      tools,
      effort,
      outputFormat,
      temperature,
      topP,
      thinkingDisplay,
    } = options;

    // Validate sampling parameters (throws a descriptive error for
    // incompatible model/parameter combinations rather than an opaque 400)
    const samplingParams = this.buildSamplingParams(model, temperature, topP);

    this.assertNoPrefill(model, messages);

    // This message formatting logic is shared with generateResponse.
    const formattedMessages = this.formatMessages(messages, model);

    try {
      const resolvedMaxTokens =
        maxTokens || this.getDefaultMaxTokens(model, true);

      const thinkingParams = this.buildThinkingParams(
        model,
        thinkingMode,
        budgetTokens,
        resolvedMaxTokens,
        effort,
        outputFormat,
        thinkingDisplay
      );
      const requestTools = this.buildRequestTools(
        model,
        tools,
        options.webSearch
      );

      // State for the final meta block, summed across pause_turn rounds
      let finalResponseId: string | undefined = undefined;
      let inputTokens = 0;
      let outputTokens = 0;
      let webSearches = 0;
      let webFetches = 0;
      let stopReason: string | null | undefined = undefined;
      let stopDetails: { category?: string | null } | null | undefined =
        undefined;
      // Fetched document title -> URL, for resolving document citations
      const fetchedDocs = new Map<string, string>();
      let requestMessages: any[] = formattedMessages;

      for (let round = 0; ; round++) {
        const stream = await this.client.messages.create({
          model,
          messages: requestMessages,
          system: systemPrompt,
          max_tokens: resolvedMaxTokens,
          stream: true as const,
          ...thinkingParams,
          ...samplingParams,
          ...(requestTools && { tools: requestTools }),
        });

        const partialBlocks = new Map<number, ContentBlock>();
        // Verbatim API blocks for this round, sent back on pause_turn
        const rawBlocks = new Map<number, any>();
        let roundOutputTokens = 0;
        let roundSearches = 0;
        let roundFetches = 0;
        stopReason = undefined;
        stopDetails = undefined;

        for await (const event of stream) {
          switch (event.type) {
            case "message_start":
              // Capture the response ID and input tokens, but do not yield yet.
              finalResponseId = event.message.id;
              inputTokens += event.message.usage.input_tokens;
              break;

            case "content_block_start": {
              const { index, content_block } = event;
              const raw: any = { ...content_block };
              if (raw.type === "text") {
                raw.text = "";
                if (!raw.citations?.length) delete raw.citations;
              }
              if (raw.type === "thinking") {
                raw.thinking = "";
                raw.signature = "";
              }
              if (raw.type === "tool_use" || raw.type === "server_tool_use") {
                raw.input = "";
              }
              rawBlocks.set(index, raw);

              switch (content_block.type) {
                case "text":
                  partialBlocks.set(index, { type: "text", text: "" });
                  break;
                case "thinking":
                  partialBlocks.set(index, {
                    type: "thinking",
                    thinking: "",
                    signature: "anthropic",
                  });
                  break;
                case "redacted_thinking":
                  // Arrives complete in content_block_start; no deltas follow.
                  yield { type: "redacted_thinking", data: content_block.data };
                  break;
                case "tool_use":
                  partialBlocks.set(index, {
                    type: "tool_use",
                    id: content_block.id,
                    name: content_block.name,
                    input: "",
                  });
                  break;
                case "server_tool_use":
                  partialBlocks.set(index, {
                    type: "server_tool_use",
                    id: content_block.id,
                    name: content_block.name,
                    input: "",
                    status: "completed",
                    vendor: "anthropic",
                  });
                  break;
                case "web_search_tool_result":
                case "web_fetch_tool_result": {
                  // Results arrive complete in content_block_start.
                  const mapped = this.mapServerToolResult(
                    content_block,
                    fetchedDocs
                  );
                  if (mapped) yield mapped;
                  break;
                }
              }
              break;
            }

            case "content_block_delta": {
              const { index, delta } = event;
              const block = partialBlocks.get(index);
              const raw = rawBlocks.get(index);
              if (!block) break;

              switch (delta.type) {
                case "text_delta":
                  if (block.type === "text") {
                    if (raw) raw.text += delta.text;
                    yield { type: "text", text: delta.text };
                  }
                  break;
                case "citations_delta":
                  if (block.type === "text") {
                    if (raw) (raw.citations ??= []).push(delta.citation);
                    const citation = this.mapCitation(
                      delta.citation,
                      fetchedDocs
                    );
                    // An empty text chunk that only carries the citation
                    if (citation) {
                      yield { type: "text", text: "", citations: [citation] };
                    }
                  }
                  break;
                case "thinking_delta":
                  if (block.type === "thinking") {
                    if (raw) raw.thinking += delta.thinking;
                    yield {
                      type: "thinking",
                      thinking: delta.thinking,
                      signature: "",
                    };
                  }
                  break;
                case "signature_delta":
                  if (block.type === "thinking") {
                    if (raw) raw.signature += delta.signature;
                    yield {
                      type: "thinking",
                      thinking: "",
                      signature: delta.signature,
                    };
                  }
                  break;
                case "input_json_delta":
                  if (
                    block.type === "tool_use" ||
                    block.type === "server_tool_use"
                  ) {
                    block.input += delta.partial_json;
                    if (raw) raw.input += delta.partial_json;
                  }
                  break;
              }
              break;
            }

            case "content_block_stop": {
              const { index } = event;
              const block = partialBlocks.get(index);
              const raw = rawBlocks.get(index);
              if (
                raw &&
                (raw.type === "tool_use" || raw.type === "server_tool_use")
              ) {
                try {
                  raw.input = raw.input ? JSON.parse(raw.input) : {};
                } catch {
                  raw.input = {};
                }
              }
              if (block && block.type === "tool_use") {
                yield block;
              }
              if (block && block.type === "server_tool_use") {
                yield { ...block, input: block.input || "{}" };
              }
              break;
            }

            case "message_delta":
              // The usage in message_delta is cumulative for this round.
              // We capture the latest value on each delta event.
              if (event.usage) {
                roundOutputTokens = event.usage.output_tokens;
                if (event.usage.server_tool_use) {
                  roundSearches =
                    event.usage.server_tool_use.web_search_requests ?? 0;
                  roundFetches =
                    event.usage.server_tool_use.web_fetch_requests ?? 0;
                }
              }
              if (event.delta?.stop_reason) {
                stopReason = event.delta.stop_reason;
              }
              if (event.delta?.stop_details) {
                stopDetails = event.delta.stop_details;
              }
              break;

            case "message_stop":
              // This event signals the end of the stream. We will yield our meta block after the loop.
              break;
          }
        }

        outputTokens += roundOutputTokens;
        webSearches += roundSearches;
        webFetches += roundFetches;

        if (stopReason !== "pause_turn" || round >= MAX_PAUSE_CONTINUATIONS) {
          break;
        }
        // Continue the paused turn: send its content back as-is.
        const pausedContent = Array.from(rawBlocks.entries())
          .sort(([a], [b]) => a - b)
          .map(([, raw]) => raw);
        requestMessages = [
          ...requestMessages,
          { role: "assistant", content: pausedContent },
        ];
      }

      // Surface refusals, truncation, and context-window stops so callers
      // don't silently get a partial or empty answer.
      const stopError = this.buildStopReasonError(
        stopReason,
        stopDetails,
        resolvedMaxTokens
      );
      if (stopError) {
        yield stopError;
      }

      // After the loop, calculate the final usage and yield the meta block.
      if (finalResponseId) {
        let finalUsage: UsageResponse | undefined = undefined;
        if (this.inputTokenCost && this.outputTokenCost) {
          const inputCost = computeResponseCost(
            inputTokens,
            this.inputTokenCost
          );
          const outputCost = computeResponseCost(
            outputTokens,
            this.outputTokenCost
          );
          const webSearchCost = this.computeWebSearchFee(webSearches);
          finalUsage = {
            inputCost,
            outputCost,
            ...(webSearchCost > 0 && { webSearchCost }),
            ...(webSearches > 0 && {
              didWebSearch: true,
              webSearchCount: webSearches,
            }),
            ...(webFetches > 0 && { webFetchCount: webFetches }),
            totalCost: inputCost + outputCost + webSearchCost,
          };
        }
        yield {
          type: "meta",
          responseId: finalResponseId,
          usage: finalUsage,
        };
      }
    } catch (error) {
      console.error("Error during Anthropic stream:", error);
      // Yield a structured error that the frontend can safely render.
      const errorMessage =
        error instanceof Error
          ? error.message
          : "An unknown stream error occurred";
      yield {
        type: "error",
        publicMessage: "An error occurred while streaming from the provider.",
        privateMessage: errorMessage,
      };
      throw error; // Also re-throw to ensure the server logs it.
    }
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    // Combine history with the current prompt
    const messagesToSend = [...chat.responseHistory]; // Copy history

    // Prepare content for the current user message, potentially including an image
    let currentUserContent: ContentBlock[] = [];

    // Add image first if provided and model supports vision (Anthropic expects image before text)
    if (chat.visionUrl && this.isVisionCapable) {
      const imageBlock: ImageBlock = {
        type: "image", // Use 'image' type for URL-based images
        url: chat.visionUrl,
      };
      currentUserContent.push(imageBlock);
    } else if (chat.visionUrl && !this.isVisionCapable) {
      console.warn(
        `Vision URL provided for non-vision capable Anthropic model '${chat.model}'. Ignoring image.`
      );
    }

    // Add text prompt after image (if any)
    if (chat.prompt) {
      currentUserContent.push({ type: "text", text: chat.prompt });
    }

    // Add the combined user message content to the history if it's not empty
    if (currentUserContent.length > 0) {
      // Ensure the content array isn't empty before pushing
      messagesToSend.push({
        role: "user",
        content: currentUserContent, // Pass the array potentially containing image and text
      });
    }

    // Format MCP tools for Anthropic API if available
    let formattedTools: AnthropicTool[] | undefined = undefined;
    if (chat.mcpAvailableTools && chat.mcpAvailableTools.length > 0) {
      formattedTools = chat.mcpAvailableTools.map((tool): AnthropicTool => {
        // Ensure the map callback returns AnthropicTool
        // Define a fallback schema (JSON Schema object)
        const fallbackSchema: Record<string, any> = {
          // Use Record<string, any> for generic JSON schema
          type: "object",
          properties: {},
        };
        let schemaObject: Record<string, any>; // Use Record<string, any>

        if (typeof tool.input_schema === "string") {
          try {
            const parsedSchema = JSON.parse(tool.input_schema);
            // Basic validation to ensure it looks like an InputSchema
            if (
              typeof parsedSchema === "object" &&
              parsedSchema !== null &&
              "type" in parsedSchema
            ) {
              schemaObject = parsedSchema; // Assign directly
            } else {
              console.error(
                `Parsed input_schema string for tool ${tool.name} is not a valid schema object.`,
                parsedSchema
              );
              schemaObject = fallbackSchema;
            }
          } catch (error) {
            console.error(
              `Error parsing input_schema string for tool ${tool.name}:`,
              error
            );
            schemaObject = fallbackSchema; // Use fallback on parsing error
          }
        } else if (
          typeof tool.input_schema === "object" &&
          tool.input_schema !== null
        ) {
          // Basic validation for existing object
          if ("type" in tool.input_schema) {
            schemaObject = tool.input_schema; // Assign directly
          } else {
            console.error(
              `Provided input_schema object for tool ${tool.name} is missing 'type' property.`,
              tool.input_schema
            );
            schemaObject = fallbackSchema;
          }
        } else {
          console.error(
            `Invalid input_schema type for tool ${
              tool.name
            }: expected string or object, got ${typeof tool.input_schema}`
          );
          schemaObject = fallbackSchema; // Use fallback for invalid types
        }

        // Always return a valid AnthropicTool structure
        return {
          name: tool.name,
          description: tool.description,
          // Cast schemaObject to 'any' to satisfy the strict InputSchema type expected by AnthropicTool,
          // relying on our runtime checks to ensure it has the necessary 'type' property.
          input_schema: schemaObject as any,
        };
      });
    }

    const response = await this.generateResponse({
      model: chat.model,
      messages: messagesToSend, // Pass the combined messages
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined,
      systemPrompt: chat.systemPrompt || undefined,
      // Thinking is enabled if a budget was given OR an explicit effort was
      // requested (adaptive-tier callers may want thinking on without setting
      // a budgetTokens value at all).
      thinkingMode: (chat.budgetTokens ?? 0) > 0 || !!chat.effort,
      tools: formattedTools, // Pass formatted tools
      effort: chat.effort,
      temperature: chat.temperature,
      topP: chat.topP,
      outputFormat: chat.outputFormat,
      thinkingDisplay: chat.thinkingDisplay,
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  // sendMCPChat method is removed as tool handling is now integrated into sendChat/generateResponse
}
