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
  ImageBlock, // Add ImageBlock for URL-based images
  ImageDataBlock, // Add ImageDataBlock for base64 images
  TextBlock, // Explicitly import TextBlock if needed for type guards
  NotImplementedError, // Import error type
  ImageGenerationResponse, // Import response type
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
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://openrouter.ai/api/v1", // Use provided baseURL or default
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
    const {
      model,
      messages,
      maxTokens,
      temperature = 1,
      systemPrompt,
      budgetTokens, // Destructure budgetTokens
    } = options;

    // Prepare messages for OpenAI Chat Completions format
    const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }

    for (const msg of messages) {
      // Skip any unexpected system messages in the array
      if (msg.role === "system") {
        console.warn(
          "System message found in messages array, skipping. Use systemPrompt parameter instead."
        );
        continue;
      }

      const role = msg.role as "user" | "assistant"; // Assert role after check

      // Map content: Handle string or ContentBlock[]
      if (typeof msg.content === "string") {
        // Simple string content - push directly
        apiMessages.push({ role, content: msg.content });
      } else if (Array.isArray(msg.content)) {
        // Array of ContentBlocks - map to OpenAI multimodal format

        // Optimization: If it's a user message with only one text block, send as simple string
        if (
          role === "user" &&
          msg.content.length === 1 &&
          msg.content[0].type === "text"
        ) {
          apiMessages.push({ role: "user", content: msg.content[0].text });
          continue; // Skip further processing for this message
        }

        // Otherwise, process as potentially multimodal
        const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [];

        for (const block of msg.content) {
          if (block.type === "text") {
            contentParts.push({ type: "text", text: block.text });
          } else if (
            block.type === "image" &&
            this.isVisionCapable &&
            role === "user" // Only add images for user role
          ) {
            contentParts.push({
              type: "image_url",
              image_url: { url: block.url },
            });
          } else if (
            block.type === "image_data" &&
            this.isVisionCapable &&
            role === "user" // Only add images for user role
          ) {
            contentParts.push({
              type: "image_url",
              image_url: {
                url: `data:${block.mimeType};base64,${block.base64Data}`,
              },
            });
          } else if (
            (block.type === "image" || block.type === "image_data") &&
            role === "assistant" // Explicitly handle image blocks in assistant messages
          ) {
            // Assistant messages cannot contain images. Skip them.
            console.warn(
              `Skipping image block (type: ${block.type}) found in assistant message.`
            );
          } else if (
            (block.type === "image" || block.type === "image_data") &&
            !this.isVisionCapable &&
            role === "user" // Check role here too
          ) {
            // Skip images if model isn't vision capable (relevant for user messages)
            console.warn(
              `Skipping image block (type: ${block.type}) in user message for non-vision model '${model}'.`
            );
          } else if (
            block.type === "thinking" ||
            block.type === "redacted_thinking" ||
            block.type === "tool_use" ||
            block.type === "tool_result"
          ) {
            // Skip non-text/image blocks for the API call
            console.warn(
              `Skipping unsupported content block type '${block.type}' for OpenRouter API call.`
            );
          }
        } // End loop through blocks

        // Only add message if it has content parts after processing blocks
        if (contentParts.length > 0) {
          if (role === "user") {
            // User messages can contain text and image_url parts
            apiMessages.push({ role: "user", content: contentParts });
          } else {
            // Role must be "assistant" here.
            // Filter contentParts to *only* include text parts.
            const assistantTextParts = contentParts.filter(
              (part): part is OpenAI.Chat.ChatCompletionContentPartText =>
                part.type === "text"
            );

            if (assistantTextParts.length > 0) {
              // Join multiple text parts into a single string for assistant message content.
              const joinedText = assistantTextParts
                .map((p) => p.text)
                .join("\n");
              apiMessages.push({ role: "assistant", content: joinedText });
            } else {
              // This happens if the original assistant message only contained non-text blocks
              console.warn(
                `Skipping assistant message as it contained no text parts after filtering.`
              );
            }
          }
        } else {
          console.warn(
            `Skipping message with role '${role}' due to no mappable content.`
          );
        }
      } else {
        console.warn(
          `Skipping message with role '${role}' due to unexpected content type: ${typeof msg.content}`
        );
      }
    } // End loop through messages

    // Ensure there are messages to send (excluding system prompt)
    const nonSystemMessages = apiMessages.filter((m) => m.role !== "system");
    if (nonSystemMessages.length === 0) {
      throw new Error(
        "No valid user or assistant messages to send to OpenRouter API."
      );
    }

    // Prepare reasoning options if applicable
    const reasoningParams =
      this.isThinkingCapable && budgetTokens && budgetTokens > 0
        ? { reasoning: { max_tokens: budgetTokens } }
        : {};

    const response = await this.client.chat.completions.create({
      model,
      messages: apiMessages,
      max_tokens: maxTokens,
      temperature,
      ...reasoningParams, // Spread reasoning parameters if they exist
    });

    let usage: UsageResponse | undefined = undefined;

    if (response.usage && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(
        response.usage.prompt_tokens,
        this.inputTokenCost
      );
      const outputCost = computeResponseCost(
        response.usage.completion_tokens,
        this.outputTokenCost
      );
      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        totalCost: inputCost + outputCost,
      };
    }

    const content = response.choices[0]?.message?.content;
    if (content === null || content === undefined) {
      // Check for null or undefined
      // Revert to throwing an error as expected by the test
      // Consider if other finish reasons should also throw or return empty
      throw new Error(
        `No content received from OpenRouter. Finish reason: ${response.choices[0]?.finish_reason}`
      );
    }

    const responseBlock: ContentBlock[] = [];

    // Check for reasoning content (assuming OpenRouter adds it to the message object)
    // Note: The exact structure might differ; adjust based on actual API response.
    // The OpenRouter docs example uses response.json()['choices'][0]['message']['reasoning']
    // The OpenAI SDK type might not have this field directly, so we access it dynamically.
    const reasoningContent = (response.choices[0]?.message as any)?.reasoning;
    if (reasoningContent && typeof reasoningContent === "string") {
      responseBlock.push({
        type: "thinking", // Use ThinkingBlock as requested
        thinking: reasoningContent,
        signature: "openrouter", // Add a signature
      });
    }

    // Add the main text content
    responseBlock.push({
      type: "text",
      text: content,
    });

    return {
      role: response.choices[0]?.message?.role || "assistant",
      content: responseBlock,
      usage: usage,
    };
  }

  // Updated signature to match AIVendorAdapter interface
  async generateImage(
    options: AIRequestOptions
  ): Promise<ImageGenerationResponse> {
    // Throw NotImplementedError as OpenRouter routes requests and doesn't generate images directly
    throw new NotImplementedError(
      "Image generation not directly supported by OpenRouter adapter. Use a specific image generation model/vendor."
    );
  }

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const {
      model,
      messages,
      maxTokens,
      temperature = 1,
      systemPrompt,
      budgetTokens,
    } = options;

    // --- This message preparation logic is copied from generateResponse for consistency ---
    const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }

    for (const msg of messages) {
      if (msg.role === "system") {
        console.warn(
          "System message found in messages array, skipping. Use systemPrompt parameter instead."
        );
        continue;
      }

      const role = msg.role as "user" | "assistant";
      if (typeof msg.content === "string") {
        apiMessages.push({ role, content: msg.content });
      } else if (Array.isArray(msg.content)) {
        if (
          role === "user" &&
          msg.content.length === 1 &&
          msg.content[0].type === "text"
        ) {
          apiMessages.push({ role: "user", content: msg.content[0].text });
          continue;
        }

        const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [];
        for (const block of msg.content) {
          if (block.type === "text") {
            contentParts.push({ type: "text", text: block.text });
          } else if (
            block.type === "image" &&
            this.isVisionCapable &&
            role === "user"
          ) {
            contentParts.push({
              type: "image_url",
              image_url: { url: block.url },
            });
          } else if (
            block.type === "image_data" &&
            this.isVisionCapable &&
            role === "user"
          ) {
            contentParts.push({
              type: "image_url",
              image_url: {
                url: `data:${block.mimeType};base64,${block.base64Data}`,
              },
            });
          } else {
            console.warn(
              `Skipping unsupported/irrelevant content block type '${block.type}' for streaming.`
            );
          }
        }

        if (contentParts.length > 0) {
          if (role === "user") {
            apiMessages.push({ role: "user", content: contentParts });
          } else {
            const assistantTextParts = contentParts.filter(
              (part): part is OpenAI.Chat.ChatCompletionContentPartText =>
                part.type === "text"
            );
            if (assistantTextParts.length > 0) {
              const joinedText = assistantTextParts
                .map((p) => p.text)
                .join("\n");
              apiMessages.push({ role: "assistant", content: joinedText });
            }
          }
        }
      }
    }

    const nonSystemMessages = apiMessages.filter((m) => m.role !== "system");
    if (nonSystemMessages.length === 0) {
      throw new Error(
        "No valid user or assistant messages to send to OpenRouter API for streaming."
      );
    }
    // --- End of copied logic ---

    const reasoningParams =
      this.isThinkingCapable && budgetTokens && budgetTokens > 0
        ? { reasoning: { max_tokens: budgetTokens } }
        : {};

    // State variables to hold data accumulated during the stream
    let responseId: string | undefined = undefined;
    let promptTokens: number | undefined = undefined;
    let completionTokens: number | undefined = undefined;

    try {
      const stream = await this.client.chat.completions.create({
        model,
        messages: apiMessages,
        max_tokens: maxTokens,
        temperature,
        stream: true, // This is the crucial parameter for enabling streaming
        ...reasoningParams,
      });

      for await (const chunk of stream) {
        // Capture the response ID from the first chunk
        if (chunk.id && !responseId) {
          responseId = chunk.id;
        }

        // Handle content deltas
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          yield {
            type: "text",
            text: content,
          };
        }

        // Capture usage data from the final chunk
        if (chunk.usage) {
          promptTokens = chunk.usage.prompt_tokens;
          completionTokens = chunk.usage.completion_tokens;
        }
      }

      // After the stream is finished, yield the final meta block
      if (
        responseId &&
        promptTokens !== undefined &&
        completionTokens !== undefined
      ) {
        let finalUsage: UsageResponse | undefined = undefined;
        if (this.inputTokenCost && this.outputTokenCost) {
          const inputCost = computeResponseCost(
            promptTokens,
            this.inputTokenCost
          );
          const outputCost = computeResponseCost(
            completionTokens,
            this.outputTokenCost
          );
          finalUsage = {
            inputCost,
            outputCost,
            totalCost: inputCost + outputCost,
          };
        }

        yield {
          type: "meta",
          responseId: responseId,
          usage: finalUsage,
        };
      }
    } catch (error: any) {
      console.error("Error during OpenRouter stream:", error);
      // In case of an error, yield an error block to the consumer.
      yield {
        type: "error",
        publicMessage:
          "An error occurred while streaming the response from OpenRouter.",
        privateMessage: error.message || String(error),
      };
    }
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    // Map Chat object to AIRequestOptions
    const options: AIRequestOptions = {
      model: chat.model,
      messages: [...chat.responseHistory], // Start with history
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined, // Pass budgetTokens
      systemPrompt: chat.systemPrompt,
      // visionUrl is removed, handled within messages now
    };

    // Construct the current user message content (potentially multimodal)
    const currentUserContent: ContentBlock[] = [];
    if (chat.visionUrl && this.isVisionCapable) {
      // Add image first (common pattern)
      currentUserContent.push({ type: "image", url: chat.visionUrl });
    } else if (chat.visionUrl && !this.isVisionCapable) {
      console.warn(
        `Vision URL provided for non-vision capable OpenRouter model '${chat.model}'. Ignoring image.`
      );
    }
    /*
    if (chat.prompt) {
      currentUserContent.push({ type: "text", text: chat.prompt });
    }
      */

    // Add the current user message to the options if it has content
    if (currentUserContent.length > 0) {
      options.messages.push({ role: "user", content: currentUserContent });
    }

    const response = await this.generateResponse(options);

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  // Removed sendMCPChat method as it's optional in the interface and not implemented.
  // Tool/Function calling support depends on the underlying model OpenRouter routes to.
}
