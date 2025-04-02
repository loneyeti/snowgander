import Anthropic from "@anthropic-ai/sdk";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  ModelConfig, // Import ModelConfig
  Chat, // Import from ../types now
  ChatResponse, // Import from ../types now
  ContentBlock, // Import from ../types now
  MCPTool, // Import from ../types now
} from "../types";
// Removed Prisma Model import
// Removed incorrect Chat, ChatResponse, ContentBlock import path
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
        >
      >((acc, block) => {
        if (block.type === "text") {
          acc.push({
            type: "text",
            text: block.text,
          });
        } else if (block.type === "thinking") {
          acc.push({
            type: "thinking",
            thinking: block.thinking,
            signature: block.signature,
          });
        }
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
    });

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
      }
      // Skip tool_use and any unknown block types for now
      // A more complete implementation might map tool_use blocks if needed by the caller
    }

    // Removed usage calculation and user update logic
    // The calling application should handle cost calculation.
    // const usage = response.usage; // Could potentially return usage if needed

    return {
      role: "assistant",
      content: contentBlocks,
    };
  }

  async generateImage(chat: Chat): Promise<string> {
    throw new Error("Image generation not supported by Anthropic");
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const response = await this.generateResponse({
      model: chat.model,
      messages: chat.responseHistory,
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined,
      systemPrompt: chat.personaPrompt || undefined,
      thinkingMode: (chat.budgetTokens ?? 0) > 0,
    });

    return {
      role: response.role,
      content: response.content,
    };
  }

  // Removed application-specific MCP logic. Tool handling should be done by the caller.
  async sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse> {
    // This adapter focuses on API interaction.
    // The calling application (Snowgoose) should:
    // 1. Prepare tool definitions based on mcpToolData.
    // 2. Call generateResponse (or a dedicated method if added) with tool definitions.
    // 3. Receive the API response, potentially containing a 'tool_use' block.
    // 4. If 'tool_use' is present, execute the tool using its own MCP logic (mcpManager).
    // 5. Send the tool execution result back to the API via another generateResponse call.
    console.warn(
      "AnthropicAdapter.sendMCPChat called, but MCP logic belongs in the calling application. Returning error."
    );
    throw new Error(
      "Direct MCP tool execution is not handled within the AIVendorAdapter. The calling application must manage tool definition, execution, and result submission."
    );

    // Example of how the *caller* might structure the flow:
    //
    // // --- In Snowgoose Server Action ---
    // const adapter = await AIVendorFactory.getAdapter(...)
    // const tools = await mcpManager.getAvailableTools(mcpToolData);
    // const formattedTools = tools.map(...) // Format for Anthropic
    //
    // let response = await adapter.generateResponse({ ..., tools: formattedTools });
    //
    // while (response.content.some(block => block.type === 'tool_use')) { // Pseudo-code check
    //    const toolUseBlock = response.content.find(block => block.type === 'tool_use');
    //    // Extract tool name, input from toolUseBlock
    //    const toolResult = await mcpManager.callTool(...);
    //    // Format toolResult for Anthropic API
    //    // Add tool result message to history
    //    response = await adapter.generateResponse({ ..., messages: updatedHistory });
    // }
    // return response;
    // // --- End Example ---
  }
}
