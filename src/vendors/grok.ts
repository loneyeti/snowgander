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
  UsageResponse,
  ImageBlock,
  ImageDataBlock,
} from "../types";
import { computeResponseCost } from "../utils";

// Interface for xAI specific delta which includes reasoning_content
interface GrokChatCompletionChunkDelta {
  content?: string | null;
  role?: "system" | "user" | "assistant" | "tool";
  reasoning_content?: string | null;
}

export class GrokAdapter implements AIVendorAdapter {
  private client: OpenAI;
  private modelConfig: ModelConfig;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number;
  public outputTokenCost?: number;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.modelConfig = modelConfig;
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://api.x.ai/v1",
      dangerouslyAllowBrowser: true,
    });
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;

    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model } = options;

    // Route to Image Generation if:
    // 1. Explicit options are provided
    // 2. The flag useImageGeneration is true
    // 3. The model name implies it is an image generation model (e.g., "grok-2-image-latest")
    const isImageModel = model.toLowerCase().includes("-image");

    if (
      (options.openaiImageGenerationOptions ||
        options.useImageGeneration ||
        isImageModel) &&
      this.isImageGenerationCapable
    ) {
      return this.generateImage(options);
    }

    // Otherwise, handle as Standard Chat Completion
    const { messages, systemPrompt } = options;
    const apiMessages = this.mapMessages(messages, systemPrompt);

    const response = await this.client.chat.completions.create({
      model: model,
      messages: apiMessages,
      stream: false,
    });

    const choice = response.choices[0];
    const contentBlocks: ContentBlock[] = [];

    // Extract Reasoning/Thinking
    const messageRaw = choice.message as any;
    if (messageRaw.reasoning_content) {
      contentBlocks.push({
        type: "thinking",
        thinking: messageRaw.reasoning_content,
        signature: "grok",
      });
    }

    // Extract Text
    if (choice.message.content) {
      contentBlocks.push({
        type: "text",
        text: choice.message.content,
      });
    }

    // Calculate Usage
    let usage: UsageResponse | undefined;
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
        inputCost,
        outputCost,
        totalCost: inputCost + outputCost,
      };
    }

    return {
      role: "assistant",
      content: contentBlocks,
      usage,
    };
  }

  private async generateImage(options: AIRequestOptions): Promise<AIResponse> {
    const { openaiImageGenerationOptions, prompt, messages, model } = options;

    // Determine the prompt: use explicit prompt or last user message
    let imagePrompt = prompt;
    if (!imagePrompt && messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      if (Array.isArray(lastMsg.content)) {
        const textBlock = lastMsg.content.find((c) => c.type === "text");
        if (textBlock && textBlock.type === "text")
          imagePrompt = textBlock.text;
      }
    }

    if (!imagePrompt) {
      throw new Error("No prompt provided for image generation");
    }

    // Use the requested model, or fallback to a safe default if somehow empty
    const imageModel = model || "grok-2-image-latest";

    const response = await this.client.images.generate({
      model: imageModel,
      prompt: imagePrompt,
      n: openaiImageGenerationOptions?.n || 1,
      response_format: "b64_json",
      size:
        openaiImageGenerationOptions?.size === "auto"
          ? undefined
          : openaiImageGenerationOptions?.size,
    });

    const contentBlocks: ContentBlock[] = [];

    if (response.data) {
      response.data.forEach((img) => {
        if (img.b64_json) {
          contentBlocks.push({
            type: "image_data",
            mimeType: "image/jpeg",
            base64Data: img.b64_json,
          });
        }
      });
    }

    return {
      role: "assistant",
      content: contentBlocks,
    };
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const response = await this.generateResponse({
      model: chat.model,
      messages: chat.responseHistory.map((r) => ({
        role: r.role,
        content: r.content,
      })),
      systemPrompt: chat.systemPrompt,
      openaiImageGenerationOptions: chat.openaiImageGenerationOptions,
      prompt: chat.prompt,
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const { model, messages, systemPrompt } = options;

    // Check if this is actually an image request coming through the stream path
    const isImageModel = model.toLowerCase().includes("-image");
    if (
      (options.openaiImageGenerationOptions ||
        options.useImageGeneration ||
        isImageModel) &&
      this.isImageGenerationCapable
    ) {
      // Image generation cannot be streamed in the traditional sense,
      // so we await the generation and yield the result.
      const response = await this.generateImage(options);
      for (const block of response.content) {
        yield block;
      }
      return;
    }

    const apiMessages = this.mapMessages(messages, systemPrompt);

    const stream = await this.client.chat.completions.create({
      model: model,
      messages: apiMessages,
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta as GrokChatCompletionChunkDelta;

      if (!delta) continue;

      // Handle Reasoning (Thinking)
      if (delta.reasoning_content) {
        yield {
          type: "thinking",
          thinking: delta.reasoning_content,
          signature: "grok",
        };
      }

      // Handle Content
      if (delta.content) {
        yield {
          type: "text",
          text: delta.content,
        };
      }
    }
  }

  private mapMessages(messages: Message[], systemPrompt?: string): any[] {
    const apiMessages: any[] = [];

    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }

    for (const msg of messages) {
      if (msg.role === "system") continue;

      const contentParts: any[] = [];

      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === "text") {
            contentParts.push({ type: "text", text: block.text });
          } else if (
            (block.type === "image" || block.type === "image_data") &&
            this.isVisionCapable
          ) {
            const imageUrl =
              block.type === "image"
                ? block.url
                : `data:${block.mimeType};base64,${block.base64Data}`;
            contentParts.push({
              type: "image_url",
              image_url: { url: imageUrl },
            });
          }
        }
      }

      if (contentParts.length > 0) {
        apiMessages.push({
          role: msg.role,
          content: contentParts,
        });
      }
    }
    return apiMessages;
  }
}
