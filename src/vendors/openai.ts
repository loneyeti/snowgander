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
              block.type === "image" &&
              this.isVisionCapable &&
              role === "user"
            ) {
              // Map ImageBlock (URL) to input_image part, only for user role
              return {
                type: "input_image",
                image_url: block.url,
              };
            } else if (
              block.type === "image_data" &&
              this.isVisionCapable &&
              role === "user"
            ) {
              // Map ImageDataBlock (base64) to data URL for input_image part, only for user role
              return {
                type: "input_image",
                image_url: `data:${block.mimeType};base64,${block.base64Data}`,
              };
            } else if (
              (block.type === "image" || block.type === "image_data") &&
              role !== "user"
            ) {
              // Images are only expected in user messages for this API structure
              console.warn(
                `Image block found in non-user message (role: ${role}). Skipping image.`
              );
              return null;
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
              block.type === "redacted_thinking"
            ) {
              return null; // Skip thinking blocks for OpenAI input
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

    const response = await this.client.responses.create({
      model: model,
      instructions: systemPrompt,
      // Use type assertion 'as any' to bypass strict SDK checks for the input array structure
      input: apiInput as any,
      // max_tokens and temperature are not direct params for this specific API endpoint
    });
    let usage: UsageResponse | undefined = undefined; // Initialize usage

    if (response.usage && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(
        response.usage.input_tokens,
        this.inputTokenCost
      );
      const outputCost = computeResponseCost(
        response.usage.output_tokens,
        this.outputTokenCost
      );
      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        totalCost: inputCost + outputCost,
      };
    }

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
      usage: usage,
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
      temperature: undefined,
      systemPrompt: chat.systemPrompt,
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  async sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse> {
    throw new Error(
      "MCP tools require specific formatting not yet implemented for OpenAI adapter using client.responses.create."
    );
  }
}
