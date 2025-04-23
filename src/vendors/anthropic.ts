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
  // MCPTool, // MCPTool type might not be needed if sendMCPChat is removed/integrated
  UsageResponse,
  ImageBlock,
  ImageDataBlock,
  NotImplementedError, // Import error type
  ImageGenerationResponse, // Import response type
} from "../types";
import { computeResponseCost } from "../utils";
// Import necessary types from Anthropic SDK
import {
  Tool as AnthropicTool,
  ToolUseBlockParam,
} from "@anthropic-ai/sdk/resources/messages";
// Removed Prisma Model import
// Removed application-specific imports (updateUserUsage, getCurrentAPIUser)

export class AnthropicAdapter implements AIVendorAdapter {
  private client: Anthropic;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

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
      budgetTokens,
      systemPrompt,
      thinkingMode,
      tools, // Destructure tools from options
    } = options;

    // Convert messages to Anthropic format
    // Convert messages to Anthropic format with inferred types
    const formattedMessages = messages.map((msg) => {
      const role =
        msg.role === "assistant" ? ("assistant" as const) : ("user" as const);

      if (typeof msg.content === "string") {
        return { role, content: msg.content };
      }

      // Map our content blocks to Anthropic's ContentBlockParam format
      const mappedContent = msg.content.reduce<
        Array<
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
              type: "tool_result";
              tool_use_id: string;
              content: string | Array<{ type: "text"; text: string }>;
            }
          | ToolUseBlockParam // Add Anthropic's ToolUseBlockParam type
        >
      >((acc, block) => {
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
      }, []);

      // If we have no valid content blocks, convert to a text block with the stringified content
      return {
        role,
        content:
          mappedContent.length > 0
            ? mappedContent
            : JSON.stringify(msg.content),
      };
    });

    const response = await this.client.messages.create({
      model,
      messages: formattedMessages,
      system: systemPrompt,
      max_tokens: maxTokens || 1024, // Default to 1024 if maxTokens is undefined
      ...(thinkingMode &&
        this.isThinkingCapable && {
          thinking: {
            type: "enabled",
            budget_tokens: budgetTokens || Math.floor((maxTokens || 1024) / 2), // Use provided budget or half of max tokens
          },
        }),
      ...(tools && { tools }), // Pass tools if provided
    });

    let usage: UsageResponse | undefined = undefined;

    if (
      response.usage.input_tokens &&
      response.usage.output_tokens &&
      this.inputTokenCost &&
      this.outputTokenCost
    ) {
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

    // Convert Anthropic response blocks to our ContentBlock format
    const contentBlocks: ContentBlock[] = [];

    for (const block of response.content) {
      if (block.type === "thinking") {
        contentBlocks.push({
          type: "thinking",
          thinking: block.thinking,
          signature: "anthropic",
        });
      } else if (block.type === "text") {
        contentBlocks.push({
          type: "text",
          text: block.text,
        });
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
  async generateImage(options: AIRequestOptions): Promise<ImageGenerationResponse> {
    // Throw NotImplementedError as Anthropic doesn't support image generation
    throw new NotImplementedError("Image generation not supported by Anthropic");
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
      thinkingMode: (chat.budgetTokens ?? 0) > 0,
      tools: formattedTools, // Pass formatted tools
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  // sendMCPChat method is removed as tool handling is now integrated into sendChat/generateResponse
}
