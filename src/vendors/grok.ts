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
  reasoning_content?: string | null; // xAI specific field
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
      baseURL: config.baseURL || "https://api.x.ai/v1", // Default to xAI API
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
    // 1. Handle explicit Image Generation Request if options are present
    if (options.openaiImageGenerationOptions && this.isImageGenerationCapable) {
      return this.generateImage(options);
    }

    // 2. Handle standard Chat Completion
    const { model, messages, systemPrompt } = options;

    const apiMessages = this.mapMessages(messages, systemPrompt);

    const response = await this.client.chat.completions.create({
      model: model,
      messages: apiMessages,
      stream: false,
    });

    const choice = response.choices[0];
    const contentBlocks: ContentBlock[] = [];

    // Extract Reasoning/Thinking (xAI uses reasoning_content)
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
    const { openaiImageGenerationOptions, prompt, messages } = options;

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

    const response = await this.client.images.generate({
      model: "grok-2-image",
      prompt: imagePrompt,
      n: openaiImageGenerationOptions?.n || 1,
      response_format: "b64_json",
      size:
        openaiImageGenerationOptions?.size === "auto"
          ? undefined
          : openaiImageGenerationOptions?.size,
    });

    const contentBlocks: ContentBlock[] = [];

    // Fix: Check if response.data exists before iterating
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
      if (msg.role === "system") continue; // Handled via systemPrompt arg usually

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
