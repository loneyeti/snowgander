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
  TextBlock, // Explicitly import TextBlock if needed for construction
  // Import other ContentBlock types if needed for construction
} from "../types";

// Re-introduce the type alias for the input expected by client.responses.create
// Based on documentation examples for message arrays.
type OpenAIInputItem = {
  role: string; // e.g., "user", "developer", "assistant"
  content: string | ContentBlock[]; // Content can be a string or array of blocks
};

export class OpenAIAdapter implements AIVendorAdapter {
  private client: OpenAI;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
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

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model, messages, systemPrompt } = options;

    // Map generic Message format to OpenAI's expected input format array
    // Use the OpenAIInputItem type alias
    const apiInput: OpenAIInputItem[] = messages.map((msg) => ({
      role: msg.role,
      content: msg.content, // Pass content directly. Assumes structure is compatible.
    }));

    // Remove the incorrect type assertion. Pass the structured array directly.
    // const typedApiInput = apiInput as OpenAI.Responses.ResponseInput; // Incorrect type

    const response = await this.client.responses.create({
      model: model,
      instructions: systemPrompt,
      // Cast input to 'any' to bypass strict SDK type checking issues for array input
      input: apiInput as any,
      // max_tokens and temperature are not direct params for this specific API endpoint
    });

    // --- Refined Content Extraction ---
    let extractedText = "";
    if (response.output_text) {
      // Use the convenience property if available
      extractedText = response.output_text;
    } else if (response.output && response.output.length > 0) {
      // Manually search for text output if output_text is not present
      for (const outputItem of response.output) {
        if (outputItem.type === "message" && outputItem.content) {
          for (const contentItem of outputItem.content) {
            if (contentItem.type === "output_text" && contentItem.text) {
              extractedText += contentItem.text; // Concatenate if multiple text parts exist
            }
          }
        }
      }
    }

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
    if (extractedText) {
      responseBlock.push({ type: "text", text: extractedText });
    }
    // TODO: Add mapping for other output types (e.g., tool calls) if necessary

    return {
      role: "assistant",
      content: responseBlock,
    };
  }

  async generateImage(chat: Chat): Promise<string> {
    if (!this.isImageGenerationCapable) {
      throw new Error("This model is not capable of image generation.");
    }
    const response = await this.client.images.generate({
      model: "dall-e-3",
      prompt: chat.prompt,
      n: 1,
      size: "1024x1024",
      quality: "standard",
    });

    if (!response.data?.[0]?.url) {
      throw new Error("No image URL received from OpenAI");
    }
    return response.data[0].url;
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const historyMessages: Message[] = chat.responseHistory.map((res) => ({
      role: res.role,
      content: res.content,
    }));

    let currentMessageContentBlocks: ContentBlock[] = [];

    if (chat.prompt) {
      currentMessageContentBlocks.push({ type: "text", text: chat.prompt });
    }

    if (chat.imageData && this.isVisionCapable) {
      const imageUrlContent: ContentBlock = {
        type: "image_url",
        image_url: {
          url: chat.imageData,
          detail: "low",
        },
      } as any;
      currentMessageContentBlocks.push(imageUrlContent);
    } else if (chat.imageData && !this.isVisionCapable) {
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
      temperature: undefined,
      systemPrompt: chat.personaPrompt,
    });

    return {
      role: response.role,
      content: response.content,
    };
  }

  async sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse> {
    throw new Error(
      "MCP tools require specific formatting not yet implemented for OpenAI adapter using client.responses.create."
    );
  }
}
