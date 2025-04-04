import {
  Content,
  GenerateContentRequest,
  GenerationConfig,
  GoogleGenerativeAI,
  Part, // Import Part type
} from "@google/generative-ai";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  ModelConfig,
  Chat,
  ChatResponse,
  ContentBlock, // Union type including ImageDataBlock
  MCPTool,
  ImageDataBlock, // Specific type for checking/casting
  Message,
  UsageResponse, // Ensure Message is imported
} from "../types";
import { computeResponseCost } from "../utils";
// Application-specific imports removed

export class GoogleAIAdapter implements AIVendorAdapter {
  private client: GoogleGenerativeAI;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.client = new GoogleGenerativeAI(config.apiKey);
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;
    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  // Helper to convert our Message format to Google's Content format
  // Correctly reference the imported Message type here
  private mapMessagesToGoogleContent(messages: Message[]): Content[] {
    return messages
      .filter((msg) => msg.role !== "system") // Filter out system messages for history
      .map((msg) => {
        const role = msg.role === "assistant" ? "model" : "user"; // Map roles
        let parts: Part[] = [];

        if (typeof msg.content === "string") {
          parts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) {
          // Handle ContentBlock array
          msg.content.forEach((block: ContentBlock) => {
            if (block.type === "text") {
              parts.push({ text: block.text });
              // Remove check for 'image_url' as it's not in ContentBlock union and needs pre-processing
              // } else if (block.type === "image_url" && this.isVisionCapable) {
              //   console.warn("Google adapter received image_url, needs pre-processing to base64 inlineData.");
            } else if (block.type === "image_data" && this.isVisionCapable) {
              // Handle pre-processed base64 data
              parts.push({
                inlineData: {
                  mimeType: block.mimeType,
                  data: block.base64Data,
                },
              });
            } else if (block.type === "thinking") {
              parts.push({ text: `<thinking>${block.thinking}</thinking>` });
            } else if (block.type === "redacted_thinking") {
              parts.push({
                text: `<redacted_thinking>${block.data}</redacted_thinking>`,
              });
            }
            // Handle 'image' type (URL-based) - needs pre-processing
            else if (block.type === "image" && this.isVisionCapable) {
              console.warn(
                "Google adapter received 'image' block (URL), needs pre-processing to base64 inlineData."
              );
            }
          });
        }
        // Ensure parts is never empty for a message
        if (parts.length === 0) {
          parts.push({ text: "" }); // Add empty text part if no other content
        }

        return { role, parts };
      });
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model, messages, maxTokens, systemPrompt, visionUrl } = options;

    const generationConfig: GenerationConfig = {};
    if (maxTokens) {
      generationConfig.maxOutputTokens = maxTokens;
    }

    const genAI = this.client.getGenerativeModel({
      model,
      generationConfig: generationConfig,
    });

    let formattedMessages = this.mapMessagesToGoogleContent(messages);

    if (systemPrompt) {
      formattedMessages = [
        { role: "user", parts: [{ text: systemPrompt }] },
        ...formattedMessages,
      ];
    }

    if (visionUrl && this.isVisionCapable) {
      const lastMessage = formattedMessages[formattedMessages.length - 1];
      if (lastMessage && lastMessage.role === "user") {
        const mimeType = "image/png"; // Example
        lastMessage.parts.push({
          inlineData: { mimeType: mimeType, data: visionUrl },
        });
      } else {
        formattedMessages.push({
          role: "user",
          parts: [{ inlineData: { mimeType: "image/png", data: visionUrl } }],
        });
        console.warn(
          "Image data provided but no suitable user message found to append to. Created new message."
        );
      }
    }

    const contentsRequest: GenerateContentRequest = {
      contents: formattedMessages,
    };

    const result = await genAI.generateContent(contentsRequest);
    const response = result.response;

    let usage: UsageResponse | undefined = undefined;

    if (response.usageMetadata && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(
        response.usageMetadata.promptTokenCount,
        this.inputTokenCost
      );
      const outputCost = computeResponseCost(
        response.usageMetadata.candidatesTokenCount,
        this.outputTokenCost
      );

      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        totalCost: inputCost + outputCost,
      };
    }

    if (!response?.candidates?.[0]?.content?.parts) {
      console.error(
        "Invalid response structure from Gemini API:",
        JSON.stringify(response)
      );
      throw new Error("Invalid response structure from Gemini API");
    }

    const contentBlocks: ContentBlock[] = [];
    for (const part of response.candidates[0].content.parts) {
      if (part.text) {
        contentBlocks.push({ type: "text", text: part.text });
      } else if (part.inlineData) {
        contentBlocks.push({
          type: "image_data",
          mimeType: part.inlineData.mimeType,
          base64Data: part.inlineData.data,
        } as ImageDataBlock);
      }
    }

    if (contentBlocks.length === 0) {
      try {
        const fallbackText = response.text();
        if (fallbackText) {
          contentBlocks.push({ type: "text", text: fallbackText });
        } else {
          throw new Error("Response has no processable parts or text.");
        }
      } catch (e) {
        console.error("Error getting fallback text:", e);
        throw new Error("No processable content found in Gemini response.");
      }
    }

    return {
      role: "assistant",
      content: contentBlocks,
      usage: usage,
    };
  }

  async generateImage(chat: Chat): Promise<string> {
    console.error("Google image generation is inline/multimodal");
    throw new Error("Error generating image.");
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const options: AIRequestOptions = {
      model: chat.model,
      messages: chat.responseHistory,
      maxTokens: chat.maxTokens || undefined,
      systemPrompt: chat.systemPrompt,
      visionUrl: chat.visionUrl || undefined,
    };

    if (chat.prompt) {
      // Ensure content is treated as an array if needed, or simple string
      const promptContent: ContentBlock[] = [
        {
          type: "text",
          text: chat.prompt,
        },
      ];
      options.messages = [
        ...options.messages,
        { role: "user", content: promptContent },
      ];
    }

    const response = await this.generateResponse(options);

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  async sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse> {
    console.warn(
      "GoogleAIAdapter.sendMCPChat called, but function calling logic is not implemented."
    );
    throw new Error(
      "MCP tool support (via function calling) not yet implemented for GoogleAIAdapter."
    );
  }
}
