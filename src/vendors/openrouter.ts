import OpenAI from "openai";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  Message,
  ModelConfig, // Import ModelConfig
  Chat, // Import from ../types now
  ChatResponse, // Import from ../types now
  ContentBlock, // Import from ../types now
  MCPTool,
  UsageResponse, // Import from ../types now
} from "../types";
import { computeResponseCost } from "../utils";
// Removed Prisma Model import
// Removed incorrect Chat, ChatResponse, ContentBlock import path
// Removed application-specific imports (getCurrentAPIUser, updateUserUsage)

export class OpenRouterAdapter implements AIVendorAdapter {
  private client: OpenAI;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  // Constructor now accepts ModelConfig instead of Prisma Model
  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    // Removed direct process.env access - Headers should be configured by the caller if needed
    // The VendorConfig could be extended to include custom headers if necessary.
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://openrouter.ai/api/v1", // Use provided baseURL or default
      // defaultHeaders: { // Headers should be set by the calling application if required
      //   "HTTP-Referer": `${siteURL}`,
      //   "X-Title": "SnowGoose AI Assistant",
      // },
    });

    // Use fields from modelConfig
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking; // OpenRouter routes to models, capability depends on the routed model
    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    // Removed user fetching logic
    const {
      model,
      messages,
      maxTokens,
      temperature = 1,
      systemPrompt,
    } = options;

    // Prepare messages for OpenAI Chat Completions format
    const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }
    messages.forEach((msg) => {
      // Map roles and content, handling potential ContentBlock arrays
      if (typeof msg.content === "string") {
        apiMessages.push({
          role: msg.role as "user" | "assistant",
          content: msg.content,
        });
      } else {
        // Handle complex content blocks - requires mapping to OpenAI's expected format
        // This might involve converting our ContentBlock[] to OpenAI's array format
        // For simplicity, we might just stringify or extract text if complex mapping isn't done
        console.warn(
          "OpenRouter adapter received complex content block, attempting simple text extraction."
        );
        const textContent = msg.content
          .map((b) => (b.type === "text" ? b.text : ""))
          .join("\n");
        apiMessages.push({
          role: msg.role as "user" | "assistant",
          content: textContent,
        });
        // TODO: Implement proper mapping for vision/other block types if needed for OpenRouter models
      }
    });

    const response = await this.client.chat.completions.create({
      model, // Keep only one model property
      messages: apiMessages,
      max_tokens: maxTokens,
      temperature,
    });

    let usage: UsageResponse | undefined = undefined; // Initialize usage

    if (response.usage && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(response.usage.prompt_tokens, this.inputTokenCost);
      const outputCost = computeResponseCost(response.usage.completion_tokens, this.outputTokenCost);
      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        totalCost: inputCost + outputCost,
      }
    }

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error("No content received from OpenRouter");
    }

    // Removed usage calculation and user update logic
    // const usage = response.usage; // Could potentially return usage if needed

    const responseBlock: ContentBlock[] = [
      {
        type: "text",
        text: content,
      },
    ];

    return {
      role: response.choices[0]?.message?.role || "assistant", // Default to assistant if role missing
      content: responseBlock,
      usage: usage
    };
  }

  async generateImage(chat: Chat): Promise<string> {
    // OpenRouter primarily routes to LLMs, image generation might depend on the specific routed model.
    // Check OpenRouter docs if they offer a dedicated image endpoint or specific model routing.
    // For now, assume not directly supported via this adapter structure.
    throw new Error(
      "Image generation not directly supported by OpenRouter adapter. Check specific model capabilities."
    );
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    // Map Chat object to AIRequestOptions
    const options: AIRequestOptions = {
      model: chat.model, // This is the OpenRouter model identifier (e.g., "openai/gpt-4-turbo")
      messages: chat.responseHistory,
      maxTokens: chat.maxTokens || undefined,
      systemPrompt: chat.personaPrompt,
      imageData: chat.imageData || undefined, // Pass base64 image data if available and model supports vision
    };

    // Add the current prompt as the last user message if it exists
    if (chat.prompt) {
      const promptContent: string | ContentBlock[] = chat.prompt;
      options.messages = [
        ...options.messages,
        { role: "user", content: promptContent },
      ];
    }

    const response = await this.generateResponse(options);

    return {
      role: response.role,
      content: response.content,
      usage: response.usage
    };
  }

  async sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse> {
    // Tool/Function calling support depends on the underlying model OpenRouter routes to.
    // The OpenAI SDK might handle function calling if the routed model supports it.
    // TODO: Investigate if OpenRouter passes through function/tool calling parameters.
    throw new Error(
      "MCP tool support via OpenRouter depends on the routed model and is not explicitly implemented here."
    );
  }
}
