import OpenAI from "openai";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  Message,
  ModelConfig,
  Chat,
  ChatResponse,
  ContentBlock,
  MCPTool,
  TextBlock,
  UsageResponse,
  ImageDataBlock, // Correctly import ImageDataBlock
  ImageBlock, // Correctly import ImageBlock
  ImageGenerationCallBlock, // Import the new type
  OpenAIImageGenerationOptions,
  Citation,
  ServerToolUseBlock,
  WebSearchOptions,
  WebSearchToolResultBlock,
  // Import other ContentBlock types if needed for construction
} from "../types";
// Removed duplicate } from "../types";
import { computeResponseCost } from "../utils";

// --- Re-introduce Custom Types for OpenAI /v1/responses Input Structure ---

// Represents the content parts we construct for the API call's message content array
type OpenAIMessageContentPart =
  | { type: "input_text"; text: string } // For user text input
  | { type: "output_text"; text: string } // For assistant text output (in history)
  | {
      type: "input_image";
      image_url: string;
    }; // For user image input

// Represents a single message structure we construct for the API call's 'input' array
type OpenAIMessageInput = {
  role: "user" | "assistant" | "developer"; // Roles for input messages
  content: OpenAIMessageContentPart[]; // Content is always an array of parts for the API
};

// START of new code to add
// Define the structure of the reasoning text item within the summary array
interface OpenAIReasoningSummaryTextItem {
  type: "summary_text";
  text: string;
}

// Define the structure of the top-level reasoning item in the output array for non-streaming
interface OpenAIReasoningOutput {
  type: "reasoning";
  summary: OpenAIReasoningSummaryTextItem[]; // Changed from 'content' to 'summary'
}

// Type guard to check if an output item matches our OpenAIReasoningOutput interface
function isReasoningOutput(item: any): item is OpenAIReasoningOutput {
  return item && item.type === "reasoning" && Array.isArray(item.summary); // Changed to 'summary'
}
// END of new code to add

export class OpenAIAdapter implements AIVendorAdapter {
  private client: OpenAI;
  private modelConfig: ModelConfig;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.modelConfig = modelConfig;
    this.client = new OpenAI({
      apiKey: config.apiKey,
      organization: config.organizationId,
      baseURL: config.baseURL,
    });
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;

    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  // --- Model tier classification ---
  //
  // OpenAI's reasoning support and sampling-parameter constraints don't move in lockstep
  // across model families, so (mirroring the Anthropic adapter) we classify each request's
  // model into two *independent* axes rather than a single generation string.
  //
  // getReasoningTier(model) -> controls which `reasoning.effort` values are valid, whether
  // `reasoning.mode: "pro"` and `text.verbosity` are available.
  // getSamplingPolicy(model) -> controls whether `temperature`/`top_p` may be sent at all.
  //
  // | Reasoning tier    | Models                              | Effort levels                          | verbosity | pro mode |
  // |-------------------|--------------------------------------|-----------------------------------------|-----------|----------|
  // | gpt5_6            | gpt-5.6, gpt-5.6-sol/terra/luna       | none/minimal/low/medium/high/xhigh/max  | yes       | yes      |
  // | gpt5              | gpt-5, gpt-5.4, gpt-5.5 (incl. mini/nano) | none/minimal/low/medium/high        | yes       | no       |
  // | reasoningLegacy   | o1, o3, o4 (incl. -mini variants)     | low/medium/high                        | no        | no       |
  // | nonReasoning (default/fallback) | gpt-4o, gpt-4.1, gpt-4-turbo, gpt-4, gpt-3.5, any unrecognized/future model | n/a (no reasoning param) | no | no |
  //
  // | Sampling policy | Reasoning tier(s)            | Behavior                                   |
  // |------------------|------------------------------|---------------------------------------------|
  // | both             | nonReasoning                  | temperature and top_p may both be sent      |
  // | none             | gpt5_6 / gpt5 / reasoningLegacy | temperature/top_p rejected by the API; we omit them and warn |
  private getReasoningTier(
    modelName: string
  ): "gpt5_6" | "gpt5" | "reasoningLegacy" | "nonReasoning" {
    if (modelName.includes("gpt-5.6")) {
      return "gpt5_6";
    }

    if (modelName.includes("gpt-5")) {
      return "gpt5";
    }

    if (/(^|[^a-zA-Z0-9])o[134](-mini)?([^a-zA-Z0-9]|$)/.test(modelName)) {
      return "reasoningLegacy";
    }

    // gpt-4o, gpt-4.1, gpt-4-turbo, gpt-4, gpt-3.5, and any unrecognized/future model.
    return "nonReasoning";
  }

  private getSamplingPolicy(modelName: string): "both" | "none" {
    return this.getReasoningTier(modelName) === "nonReasoning"
      ? "both"
      : "none";
  }

  // Backwards-compat bridge: derive an effort level from the legacy `budgetTokens` field
  // when no explicit `effort` is supplied. Only reaches low/medium/high — xhigh/max/none/
  // minimal require an explicit `effort` value.
  private mapBudgetToEffort(
    budgetTokens?: number | null
  ): "low" | "medium" | "high" {
    if (!budgetTokens || budgetTokens <= 0) return "low";
    if (budgetTokens < 8192) return "medium";
    return "high";
  }

  private buildReasoningParam(
    options: AIRequestOptions,
    modelName: string
  ): { reasoning: Record<string, any> } | undefined {
    const { effort, budgetTokens } = options;

    const reasoningRequested =
      effort !== undefined ||
      (typeof budgetTokens === "number" && budgetTokens >= 0);

    if (!reasoningRequested) {
      return undefined;
    }

    const tier = this.getReasoningTier(modelName);

    if (tier === "nonReasoning") {
      console.warn(
        `Reasoning effort/budgetTokens requested for non-reasoning model '${modelName}'. Ignoring.`
      );
      return undefined;
    }

    const reasoning: Record<string, any> = {
      effort: effort || this.mapBudgetToEffort(budgetTokens),
      summary: "auto",
    };

    if (options.reasoningMode === "pro") {
      if (tier === "gpt5_6") {
        reasoning.mode = "pro";
      } else {
        console.warn(
          `reasoningMode "pro" is only supported on GPT-5.6 models; ignoring for '${modelName}'.`
        );
      }
    }

    return { reasoning };
  }

  private buildSamplingParams(
    options: AIRequestOptions,
    modelName: string
  ): { temperature?: number; top_p?: number } {
    const { temperature, topP } = options;

    if (temperature === undefined && topP === undefined) {
      return {};
    }

    const policy = this.getSamplingPolicy(modelName);

    if (policy === "none") {
      console.warn(
        `Model '${modelName}' does not support sampling parameters (temperature/top_p) while reasoning is active. Ignoring.`
      );
      return {};
    }

    const params: { temperature?: number; top_p?: number } = {};
    if (temperature !== undefined) params.temperature = temperature;
    if (topP !== undefined) params.top_p = topP;
    return params;
  }

  private buildTextParam(
    options: AIRequestOptions,
    modelName: string
  ): { text: Record<string, any> } | undefined {
    const { outputFormat, verbosity } = options;
    const text: Record<string, any> = {};

    if (outputFormat) {
      text.format = outputFormat;
    }

    if (verbosity) {
      const tier = this.getReasoningTier(modelName);
      if (tier === "gpt5_6" || tier === "gpt5") {
        text.verbosity = verbosity;
      } else {
        console.warn(
          `verbosity is only supported on GPT-5+ models; ignoring for '${modelName}'.`
        );
      }
    }

    return Object.keys(text).length > 0 ? { text } : undefined;
  }

  /**
   * Maps internal messages to the Responses API `input` array. If any message
   * carries an ImageGenerationCallBlock, it is returned separately so the caller
   * can append the `{type: "image_generation_call", id}` reference for multi-turn
   * image editing (other image data is skipped in that case).
   */
  private mapMessagesToApiInput(
    messages: Message[],
    model: string
  ): {
    apiInput: OpenAIMessageInput[];
    imageGenerationCallReference: { type: "image_generation_call"; id: string } | null;
  } {
    // <<< CHANGE START: Find imageGenerationCallReference first
    let imageGenerationCallReference: {
      type: "image_generation_call";
      id: string;
    } | null = null;
    for (const msg of messages) {
      if (Array.isArray(msg.content)) {
        const refBlock = msg.content.find(
          (block): block is ImageGenerationCallBlock =>
            block.type === "image_generation_call"
        );
        if (refBlock) {
          imageGenerationCallReference = {
            type: "image_generation_call",
            id: refBlock.id,
          };
          break; // Found it, no need to look further
        }
      }
    }
    // <<< CHANGE END


    // Map internal Message format to OpenAI's responses.create input format
    // Use our custom types that reflect the expected API structure.
    const apiInput: OpenAIMessageInput[] = messages
      .map((msg): OpenAIMessageInput | null => {
        // Map internal roles to OpenAI's expected input roles
        let role: "user" | "assistant" | "developer"; // Roles for input messages
        switch (msg.role) {
          case "user":
          case "assistant":
            role = msg.role;
            break;
          case "system": // System prompts are handled by 'instructions', skip here
            console.warn(
              "System role message found in 'messages' array; use 'systemPrompt'/'instructions' instead. Skipping message."
            );
            return null;
          default:
            console.warn(`Unsupported role '${msg.role}' mapped to 'user'.`);
            role = "user"; // Defaulting unrecognized roles
        }
        // Removed the extra closing brace here

        // Map internal ContentBlock array to OpenAI's expected input content part array
        // Ensure msg.content is treated as an array
        if (!Array.isArray(msg.content)) {
          console.warn(
            `Message content is not an array for role '${role}'. Skipping message.`
          );
          return null;
        }

        const apiContentParts = msg.content
          .map((block): OpenAIMessageInput["content"][number] | null => {
            if (block.type === "text") {
              // Map TextBlock based on role
              if (role === "assistant") {
                // Assistant messages in history should use output_text
                return { type: "output_text", text: block.text };
              } else {
                // User messages (and fallbacks) use input_text
                return { type: "input_text", text: block.text };
              }
            } else if (
              // <<< CHANGE START: Add conditions to prevent sending conflicting images during an edit
              (block.type === "image" || block.type === "image_data") &&
              imageGenerationCallReference
              // <<< CHANGE END
            ) {
              // If we are performing an edit (imageGenerationCallReference is present),
              // we must not send any other image data.
              console.warn(
                `Skipping '${block.type}' block because an 'image_generation_call' is present for an edit operation.`
              );
              return null;
            } else if (
              (block.type === "image" || block.type === "image_data") &&
              role !== "user"
            ) {
              // Assistant turns can't carry input_image parts (e.g. a previously
              // generated image persisted in history); the API would reject them.
              return null;
            } else if (block.type === "image" && this.isVisionCapable) {
              // Map ImageBlock (URL) to input_image part, only for user role
              return {
                type: "input_image",
                image_url: block.url,
              };
            } else if (block.type === "image_data" && this.isVisionCapable) {
              // Map ImageDataBlock (base64) to data URL for input_image part, only for user role
              return {
                type: "input_image",
                image_url: `data:${block.mimeType};base64,${block.base64Data}`,
              };
            } else if (
              (block.type === "image" || block.type === "image_data") &&
              !this.isVisionCapable
            ) {
              // This condition might be redundant now due to the role check above, but kept for clarity
              console.warn(
                `Image block provided for non-vision model '${model}'. Skipping image.`
              );
              return null;
            } else if (
              block.type === "thinking" ||
              block.type === "redacted_thinking" ||
              block.type === "image_generation_call" ||
              block.type === "server_tool_use" ||
              block.type === "web_search_tool_result" ||
              block.type === "web_fetch_tool_result"
            ) {
              // Research steps are display-only here: previous_response_id
              // carries OpenAI's own search state between turns.
              return null;
            }
            console.warn(
              `Unsupported content block type '${block.type}' for OpenAI input mapping.`
            );
            return null;
            // Use our custom type for the content parts
          })
          .filter((part): part is OpenAIMessageContentPart => part !== null); // Filter out nulls

        // If no valid content parts were generated for this message, skip the message
        if (apiContentParts.length === 0) {
          console.warn(
            `Skipping message with role '${role}' due to no mappable content.`
          );
          return null;
        }

        // Construct the final message object using our custom type
        return {
          role: role,
          content: apiContentParts,
        };
      })
      .filter((msg): msg is OpenAIMessageInput => msg !== null); // Filter out any skipped messages

    // Check if apiInput is empty after filtering
    if (apiInput.length === 0) {
      throw new Error(
        "No valid messages could be mapped for the OpenAI API request."
      );
    }

    return { apiInput, imageGenerationCallReference };
  }

  /**
   * Builds the Responses API `image_generation` tool from our options.
   * "auto" and undefined values are omitted so the API defaults apply.
   */
  private buildImageGenerationTool(opts: OpenAIImageGenerationOptions): any {
    const tool: any = {
      type: "image_generation",
      partial_images: opts.partialImages ?? 2,
    };
    const set = (key: string, value: unknown) => {
      if (value !== undefined && value !== null && value !== "auto") {
        tool[key] = value;
      }
    };
    set("model", opts.model);
    set("quality", opts.quality);
    set("size", opts.size);
    set("background", opts.background);
    set("output_format", opts.outputFormat);
    set("output_compression", opts.outputCompression);
    set("moderation", opts.moderation);
    set("action", opts.action);
    set("input_fidelity", opts.inputFidelity);
    if (opts.inputImageMask?.fileId || opts.inputImageMask?.imageUrl) {
      tool.input_image_mask = {
        ...(opts.inputImageMask.fileId && { file_id: opts.inputImageMask.fileId }),
        ...(opts.inputImageMask.imageUrl && {
          image_url: opts.inputImageMask.imageUrl,
        }),
      };
    }
    return tool;
  }

  /**
   * Builds the Responses API `web_search` tool. Options OpenAI doesn't
   * support (fetch, maxUses, blockedDomains) are ignored.
   */
  private buildWebSearchTool(opts: WebSearchOptions): any {
    const tool: any = { type: "web_search" };
    if (opts.allowedDomains?.length) {
      tool.filters = { allowed_domains: opts.allowedDomains };
    }
    const loc = opts.userLocation;
    if (loc && (loc.city || loc.region || loc.country || loc.timezone)) {
      tool.user_location = {
        type: "approximate",
        ...(loc.city && { city: loc.city }),
        ...(loc.region && { region: loc.region }),
        ...(loc.country && { country: loc.country }),
        ...(loc.timezone && { timezone: loc.timezone }),
      };
    }
    return tool;
  }

  /**
   * Assembles the request's tools: caller tools (with the legacy
   * `web_search_preview` upgraded to `web_search`), the web search tool from
   * `webSearch`, and the image generation tool. Also returns the `include`
   * list, which asks for search sources when web search is on.
   */
  private buildTools(options: AIRequestOptions): {
    tools: any[];
    include: string[] | undefined;
  } {
    const tools = (options.tools ?? []).map((tool) =>
      typeof tool?.type === "string" &&
      tool.type.startsWith("web_search_preview")
        ? { ...tool, type: "web_search" }
        : tool
    );
    if (
      options.webSearch &&
      !tools.some((tool) => tool?.type === "web_search")
    ) {
      tools.push(this.buildWebSearchTool(options.webSearch));
    }
    if (options.openaiImageGenerationOptions) {
      tools.push(
        this.buildImageGenerationTool(options.openaiImageGenerationOptions)
      );
    }
    const hasWebSearch = tools.some(
      (tool) =>
        typeof tool?.type === "string" && tool.type.startsWith("web_search")
    );
    return {
      tools,
      include: hasWebSearch ? ["web_search_call.action.sources"] : undefined,
    };
  }

  /**
   * Maps a `web_search_call` output item to a research step, plus a result
   * block when the API returned sources for a search.
   */
  private mapWebSearchCall(
    item: any,
    status: ServerToolUseBlock["status"]
  ): ContentBlock[] {
    const action = item?.action;
    let input: Record<string, unknown> = {};
    if (action?.type === "search") {
      input = action.queries?.length
        ? { query: action.query ?? action.queries[0], queries: action.queries }
        : { query: action.query };
    } else if (action?.type === "open_page") {
      input = { url: action.url };
    } else if (action?.type === "find") {
      input = { url: action.url, pattern: action.pattern };
    }
    const blocks: ContentBlock[] = [
      {
        type: "server_tool_use",
        id: item.id,
        name: "web_search",
        input: JSON.stringify(input),
        status: item.status === "failed" ? "failed" : status,
        vendor: "openai",
      },
    ];
    if (Array.isArray(action?.sources) && action.sources.length > 0) {
      const result: WebSearchToolResultBlock = {
        type: "web_search_tool_result",
        toolUseId: item.id,
        results: action.sources
          .filter((source: any) => typeof source?.url === "string")
          .map((source: any) => ({ url: source.url })),
        vendor: "openai",
      };
      blocks.push(result);
    }
    return blocks;
  }

  /** Maps a `url_citation` annotation, or returns null for other kinds. */
  private mapUrlCitation(annotation: any, offset = 0): Citation | null {
    if (annotation?.type !== "url_citation" || !annotation.url) return null;
    return {
      type: "url_citation",
      url: annotation.url,
      ...(annotation.title && { title: annotation.title }),
      ...(typeof annotation.start_index === "number" && {
        startIndex: annotation.start_index + offset,
      }),
      ...(typeof annotation.end_index === "number" && {
        endIndex: annotation.end_index + offset,
      }),
      vendor: "openai",
    };
  }

  /**
   * Number of billable web searches: every web_search_call except page opens
   * and in-page finds, which are follow-up actions.
   */
  private countWebSearches(output: any[] | undefined): number {
    return (output ?? []).filter(
      (item) =>
        item?.type === "web_search_call" &&
        item.action?.type !== "open_page" &&
        item.action?.type !== "find"
    ).length;
  }

  private imageMimeType(opts?: OpenAIImageGenerationOptions): string {
    switch (opts?.outputFormat) {
      case "jpeg":
        return "image/jpeg";
      case "webp":
        return "image/webp";
      default:
        return "image/png";
    }
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model, messages, systemPrompt, tools, store, previousResponseId } =
      options;
    // Flags to track tool usage from the response
    let didGenerateImage = false;

    const { apiInput, imageGenerationCallReference } =
      this.mapMessagesToApiInput(messages, model);

    // Get the reasoning/sampling/text parameters via our tier-aware helper methods
    const reasoningParam = this.buildReasoningParam(options, model);
    const samplingParams = this.buildSamplingParams(options, model);
    const textParam = this.buildTextParam(options, model);

    const { tools: finalTools, include } = this.buildTools(options);

    // Build the final payload, including the image_generation_call reference if it exists
    const finalApiPayload = [...apiInput];
    if (imageGenerationCallReference) {
      finalApiPayload.push(imageGenerationCallReference as any); // Add the reference to the payload
    }

    const response = await this.client.responses.create({
      model: model,
      instructions: systemPrompt,
      previous_response_id: previousResponseId, // <-- ADDED
      // Use type assertion 'as any' to bypass strict SDK checks for the input array structure
      input: finalApiPayload as any,
      tools: finalTools,
      ...(include && { include: include as any }),
      store: store,
      // Spread the reasoning/sampling/text fragments if they exist.
      ...(reasoningParam as any),
      ...(samplingParams as any),
      ...(textParam as any),
    });

    // Check for tool usage in the response
    if (response.output && response.output.length > 0) {
      for (const outputItem of response.output) {
        if (outputItem.type === "image_generation_call") {
          didGenerateImage = true;
        }
      }
    }
    const webSearchCount = this.countWebSearches(response.output as any[]);
    const didUseWebSearch = webSearchCount > 0;

    let usage: UsageResponse | undefined = undefined; // Initialize usage

    if (response.usage && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(
        response.usage.input_tokens,
        this.inputTokenCost
      );

      // Determine which output cost to use
      const outputCostPerToken =
        didGenerateImage && this.modelConfig.imageOutputTokenCost
          ? this.modelConfig.imageOutputTokenCost
          : this.outputTokenCost;

      let outputCost = computeResponseCost(
        response.usage.output_tokens,
        outputCostPerToken
      );

      // Until OpenAI can provide the actual cost of the transaction
      // we must hard code in the most expensive image generation
      // outputCost = didGenerateImage ? outputCost + 0.25 : outputCost;

      // webSearchCost is charged per search call
      const webSearchCost = (this.modelConfig.webSearchCost ?? 0) * webSearchCount;

      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        webSearchCost: webSearchCost > 0 ? webSearchCost : undefined,
        didGenerateImage: didGenerateImage,
        didWebSearch: didUseWebSearch,
        ...(webSearchCount > 0 && { webSearchCount }),
        totalCost: inputCost + outputCost + webSearchCost,
      };
    }

    // --- Refined Content Extraction ---
    // Text parts are concatenated into one block. Citation offsets are shifted
    // so they index into the concatenated text.
    let messageText = "";
    const citations: Citation[] = [];
    for (const outputItem of (response.output ?? []) as any[]) {
      if (outputItem.type === "message" && outputItem.content) {
        for (const contentItem of outputItem.content) {
          if (contentItem.type === "output_text" && contentItem.text) {
            for (const annotation of contentItem.annotations ?? []) {
              const citation = this.mapUrlCitation(
                annotation,
                messageText.length
              );
              if (citation) citations.push(citation);
            }
            messageText += contentItem.text;
          }
        }
      }
    }
    // Prefer the convenience property if available
    const extractedText = response.output_text || messageText;

    // Simplified check: if we didn't extract any text, log an error.
    // A more robust check might inspect for specific non-text outputs like tool calls if needed.
    if (!extractedText) {
      // Check if there's *any* output item, even if not text, to avoid erroring on valid non-text responses
      const hasAnyOutput = response.output && response.output.length > 0;
      if (!hasAnyOutput) {
        console.error(
          "OpenAI Response (No Text/Output):",
          JSON.stringify(response, null, 2)
        );
        throw new Error("No content or output received from OpenAI");
      } else {
        // It has output, just not text we could extract easily (e.g., maybe tool calls)
        console.warn(
          "OpenAI Response contained non-text output:",
          JSON.stringify(response.output, null, 2)
        );
      }
    }

    // --- Map response back to ContentBlock array ---
    const responseBlock: ContentBlock[] = [];

    if (response.output && Array.isArray(response.output)) {
      const reasoningOutputs = response.output.filter(
        isReasoningOutput
      ) as OpenAIReasoningOutput[]; // Uses our type guard

      if (reasoningOutputs.length > 0) {
        // Create a temporary array to hold all summary items from all reasoning blocks.
        const allSummaryItems: OpenAIReasoningSummaryTextItem[] = [];

        // Use a for...of loop. Inside this loop, TypeScript *guarantees*
        // that 'output' is of type OpenAIReasoningOutput.
        for (const output of reasoningOutputs) {
          allSummaryItems.push(...output.summary);
        }

        // Now, map over the flattened and correctly typed array.
        const reasoningText = allSummaryItems
          .map((item) => (item.type === "summary_text" ? item.text : ""))
          .join("\n\n");

        if (reasoningText) {
          responseBlock.push({
            type: "thinking",
            thinking: reasoningText,
            signature: "openai",
          });
        }
      }
    }

    // Research steps, in the order the model ran them
    for (const outputItem of (response.output ?? []) as any[]) {
      if (outputItem.type === "web_search_call") {
        responseBlock.push(...this.mapWebSearchCall(outputItem, "completed"));
      }
    }

    if (extractedText) {
      responseBlock.push({
        type: "text",
        text: extractedText,
        ...(citations.length > 0 && { citations }),
      });
    }

    // Handle image generation output if tools were used
    if (finalTools?.some((tool) => tool.type === "image_generation")) {
      const imageGenerationOutput = response.output
        .filter((output) => (output as any).type === "image_generation_call")
        .filter((output) => typeof (output as any).result === "string");

      for (const imageCall of imageGenerationOutput) {
        responseBlock.push({
          type: "image_data",
          id: (imageCall as any).id, // <-- CAPTURE THE ID HERE
          mimeType: this.imageMimeType(options.openaiImageGenerationOptions),
          base64Data: (imageCall as any).result,
        });
      }
    }

    return {
      role: "assistant",
      content: responseBlock,
      usage: usage,
    };
  }

  // Removed generateImage method. Image generation should be handled by OpenAIImageAdapter.
  // async generateImage(chat: Chat): Promise<string> { ... }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    // Ensure history message content is always ContentBlock[]
    const historyMessages: Message[] = chat.responseHistory.map((res) => {
      let contentBlocks: ContentBlock[];
      if (typeof res.content === "string") {
        // If content is a simple string, wrap it in a TextBlock array
        contentBlocks = [{ type: "text", text: res.content }];
      } else if (Array.isArray(res.content)) {
        // Assume it's already ContentBlock[] if it's an array
        // TODO: Add validation here if stricter type checking is needed
        contentBlocks = res.content;
      } else {
        // Handle unexpected content types (e.g., log a warning, skip message)
        console.warn(
          `Unexpected content type in response history for role '${res.role}'. Skipping content.`
        );
        contentBlocks = []; // Or handle as appropriate
      }
      return {
        role: res.role,
        content: contentBlocks,
      };
    });

    let currentMessageContentBlocks: ContentBlock[] = [];

    if (chat.prompt) {
      currentMessageContentBlocks.push({ type: "text", text: chat.prompt });
    }

    if (chat.visionUrl && this.isVisionCapable) {
      // Use the correct internal type 'ImageBlock' for URL-based images
      const imageBlock: ImageBlock = {
        type: "image", // Use 'image' type for URL
        url: chat.visionUrl,
      };
      currentMessageContentBlocks.push(imageBlock);
    } else if (chat.visionUrl && !this.isVisionCapable) {
      console.warn(
        "Image data provided to a non-vision capable model. Ignoring image."
      );
    }

    if (currentMessageContentBlocks.length > 0) {
      historyMessages.push({
        role: "user",
        content: currentMessageContentBlocks, // Pass the array of blocks
      });
    }

    const response = await this.generateResponse({
      model: chat.model,
      messages: historyMessages,
      maxTokens: chat.maxTokens || undefined,
      systemPrompt: chat.systemPrompt,
      previousResponseId: chat.previousResponseId,
      budgetTokens: chat.budgetTokens ?? undefined,
      effort: chat.effort,
      temperature: chat.temperature,
      topP: chat.topP,
      outputFormat: chat.outputFormat,
      verbosity: chat.verbosity,
      reasoningMode: chat.reasoningMode,
      openaiImageGenerationOptions: chat.openaiImageGenerationOptions,
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
      responseId: response.responseId, // Pass the new ID back
    };
  }

  // Removed sendMCPChat method as it's optional in the interface and not implemented
  // for this specific API endpoint (client.responses.create).
  // Tool handling would need to be integrated into generateResponse/sendChat
  // if using the chat completions endpoint in the future.

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const { model, messages, systemPrompt, tools, store, previousResponseId } =
      options;
    let finalResponseId: string | undefined = undefined;
    let finalUsage: UsageResponse | undefined = undefined;
    const imageMimeType = this.imageMimeType(options.openaiImageGenerationOptions);
    // Ids of image_generation_call items whose final image has been yielded
    const yieldedFinalImageIds = new Set<string>();

    const { apiInput, imageGenerationCallReference } =
      this.mapMessagesToApiInput(messages, model);
    const finalApiPayload: any[] = [...apiInput];
    if (imageGenerationCallReference) {
      finalApiPayload.push(imageGenerationCallReference);
    }

    const { tools: finalTools, include } = this.buildTools(options);

    const reasoningParam = this.buildReasoningParam(options, model);
    const samplingParams = this.buildSamplingParams(options, model);
    const textParam = this.buildTextParam(options, model);

    const requestBody = {
      model: model,
      instructions: systemPrompt,
      previous_response_id: previousResponseId, // Pass state ID
      input: finalApiPayload,
      tools: finalTools,
      ...(include && { include }),
      store: store,
      stream: true,
      ...(reasoningParam as any),
      ...(samplingParams as any),
      ...(textParam as any),
    };

    // Omit `input` from the log: it can contain large base64 images
    console.log(
      `SNOWGANDER: sending this request: ${JSON.stringify({
        ...requestBody,
        input: `[${finalApiPayload.length} items]`,
      })}`
    );

    try {
      // First cast to unknown, then to AsyncIterable to satisfy TypeScript's type safety
      const stream = (await this.client.responses.create(
        requestBody as any
      )) as unknown as AsyncIterable<any>;

      for await (const event of stream) {
        // Log the type only: image events carry megabytes of base64
        console.log(`SNOWGANDER: Received stream event: ${event.type}`);
        switch (event.type) {
          case "response.output_text.delta":
            if (event.delta) {
              yield {
                type: "text",
                text: event.delta,
              };
            }
            break;

          case "response.image_generation_call.partial_image":
            if (event.partial_image_b64) {
              yield {
                type: "image_data",
                id: event.item_id ?? null,
                mimeType: imageMimeType,
                base64Data: event.partial_image_b64,
                isPartial: true,
                partialImageIndex: event.partial_image_index,
              };
            }
            break;

          case "response.output_item.added":
            if (event.item?.type === "web_search_call") {
              yield* this.mapWebSearchCall(event.item, "in_progress");
            }
            break;

          case "response.output_text.annotation.added": {
            // Citations stream separately from text. Yield them as an empty
            // text chunk so consumers can attach them to the current text block.
            const citation = this.mapUrlCitation(event.annotation);
            if (citation) {
              yield { type: "text", text: "", citations: [citation] };
            }
            break;
          }

          case "response.output_item.done":
            if (event.item?.type === "web_search_call") {
              yield* this.mapWebSearchCall(event.item, "completed");
              break;
            }
            // The finished image arrives on the completed output item, not in a
            // partial_image event.
            if (
              event.item?.type === "image_generation_call" &&
              typeof event.item.result === "string" &&
              event.item.result
            ) {
              yieldedFinalImageIds.add(event.item.id);
              yield {
                type: "image_data",
                id: event.item.id ?? null,
                mimeType: imageMimeType,
                base64Data: event.item.result,
                isPartial: false,
              };
            }
            break;

          case "response.reasoning_summary_text.delta":
            if (event.delta) {
              yield {
                type: "thinking",
                thinking: event.delta,
                signature: "openai",
              };
            }
            break;

          case "response.completed": // Capture final data
            finalResponseId = event.response.id;

            // Fallback: yield any final image not already delivered via output_item.done
            for (const item of event.response.output ?? []) {
              if (
                item?.type === "image_generation_call" &&
                typeof item.result === "string" &&
                item.result &&
                !yieldedFinalImageIds.has(item.id)
              ) {
                yieldedFinalImageIds.add(item.id);
                yield {
                  type: "image_data",
                  id: item.id ?? null,
                  mimeType: imageMimeType,
                  base64Data: item.result,
                  isPartial: false,
                };
              }
            }

            if (
              event.response.usage &&
              this.inputTokenCost &&
              this.outputTokenCost
            ) {
              const didGenerateImage = event.response.output?.some(
                (o: any) => o.type === "image_generation_call"
              );
              const webSearchCount = this.countWebSearches(
                event.response.output
              );
              const didWebSearch = webSearchCount > 0;
              const inputCost = computeResponseCost(
                event.response.usage.input_tokens,
                this.inputTokenCost
              );
              const outputCostPerToken =
                didGenerateImage && this.modelConfig.imageOutputTokenCost
                  ? this.modelConfig.imageOutputTokenCost
                  : this.outputTokenCost;
              let outputCost = computeResponseCost(
                event.response.usage.output_tokens,
                outputCostPerToken
              );
              // Until OpenAI can provide the actual cost of the transaction
              // we must hard code in the most expensive image generation
              // outputCost = didGenerateImage ? outputCost + 0.25 : outputCost;
              // webSearchCost is charged per search call
              const webSearchCost =
                (this.modelConfig.webSearchCost ?? 0) * webSearchCount;
              finalUsage = {
                inputCost,
                outputCost,
                webSearchCost: webSearchCost > 0 ? webSearchCost : undefined,
                didGenerateImage,
                didWebSearch,
                ...(webSearchCount > 0 && { webSearchCount }),
                totalCost: inputCost + outputCost + webSearchCost,
              };
            }

            break;

          case "response.incomplete": {
            const reason =
              event.response?.incomplete_details?.reason || "unknown";
            yield {
              type: "error",
              code: reason,
              publicMessage: "The response was cut off before it finished.",
              privateMessage: `OpenAI response incomplete: ${reason}`,
            };
            break;
          }

          case "response.failed": {
            const error = event.response?.error;
            const errorMessage = error?.message || "Response failed in stream.";
            console.error(`OpenAI stream failed: ${errorMessage}`);
            yield {
              type: "error",
              code: error?.code ?? undefined,
              publicMessage:
                error?.code === "moderation_blocked"
                  ? "The request was blocked by content moderation."
                  : "The request failed during streaming.",
              privateMessage: error?.moderation_details
                ? `${errorMessage} (moderation_details: ${JSON.stringify(
                    error.moderation_details
                  )})`
                : errorMessage,
            };
            return; // Terminate the generator on a failure event
          }

          case "error": {
            // Top-level stream error event (e.g. moderation_blocked)
            const errorMessage = event.message || "Stream error.";
            console.error(`OpenAI stream error: ${errorMessage}`);
            yield {
              type: "error",
              code: event.code ?? undefined,
              publicMessage:
                event.code === "moderation_blocked"
                  ? "The request was blocked by content moderation."
                  : "The request failed during streaming.",
              privateMessage: event.moderation_details
                ? `${errorMessage} (moderation_details: ${JSON.stringify(
                    event.moderation_details
                  )})`
                : errorMessage,
            };
            return;
          }
        }
      }
      if (finalResponseId) {
        yield { type: "meta", responseId: finalResponseId, usage: finalUsage };
      }
    } catch (error: any) {
      console.error("Error during OpenAI stream:", error);
      yield {
        type: "error",
        code: error?.code ?? undefined,
        publicMessage:
          error?.code === "moderation_blocked"
            ? "The request was blocked by content moderation."
            : "An error occurred while streaming the response.",
        privateMessage: error.message || String(error),
      };
    }
  }
}
