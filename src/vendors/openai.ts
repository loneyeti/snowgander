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

  private getReasoningParam(budgetTokens: number | null | undefined):
    | {
        reasoning: {
          effort: "low" | "medium" | "high";
          summary: "auto" | "concise" | "detailed";
        };
      }
    | undefined {
    // If budgetTokens is not provided, do not add the reasoning parameter.
    if (typeof budgetTokens !== "number" || budgetTokens < 0) {
      return undefined;
    }

    let effort: "low" | "medium" | "high";

    if (budgetTokens === 0) {
      effort = "low";
    } else if (budgetTokens > 0 && budgetTokens < 8192) {
      effort = "medium";
    } else {
      // budgetTokens >= 8192
      effort = "high";
    }

    return {
      reasoning: { effort: effort, summary: "detailed" },
    };
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model, messages, systemPrompt, tools, store, previousResponseId } =
      options;
    // Flags to track tool usage from the response
    let didGenerateImage = false;
    let didUseWebSearch = false;

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

    const filteredMessages = messages;

    // Map internal Message format to OpenAI's responses.create input format
    // Use our custom types that reflect the expected API structure.
    const apiInput: OpenAIMessageInput[] = filteredMessages
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
              block.type === "image_generation_call"
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

    // Get the reasoning parameter using our new helper method
    const reasoningParam = this.getReasoningParam(options.budgetTokens);

    // New code to be inserted starts here
    let finalTools = tools ? [...tools] : [];
    if (options.openaiImageGenerationOptions) {
      const imageGenOptions = options.openaiImageGenerationOptions;

      const imageGenerationTool: any = {
        type: "image_generation",
        partial_images: 1,
      };

      if (imageGenOptions.quality && imageGenOptions.quality !== "auto") {
        imageGenerationTool.quality = imageGenOptions.quality;
      }
      if (imageGenOptions.size && imageGenOptions.size !== "auto") {
        imageGenerationTool.size = imageGenOptions.size;
      }
      if (imageGenOptions.background && imageGenOptions.background !== "auto") {
        imageGenerationTool.background = imageGenOptions.background;
      }

      // Add the fully constructed tool to our tools array.
      finalTools.push(imageGenerationTool);
      //console.log(
      //  "[SNOWGANDER DIAGNOSTIC] - generateResponse - Sending tools to OpenAI:",
      //  JSON.stringify(finalTools, null, 2)
      //);
    }
    // New code to be inserted ends here

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
      tools: finalTools, // <-- This line is changed
      store: store,
      // Use spread syntax to add the reasoning parameter if it exists.
      ...(reasoningParam as any),
    });

    // Check for tool usage in the response
    if (response.output && response.output.length > 0) {
      for (const outputItem of response.output) {
        if (outputItem.type === "image_generation_call") {
          didGenerateImage = true;
        }
        if (outputItem.type === "web_search_call") {
          didUseWebSearch = true;
        }
      }
    }

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

      // Check for web search flat fee
      const webSearchCost =
        didUseWebSearch && this.modelConfig.webSearchCost
          ? this.modelConfig.webSearchCost
          : 0;

      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        webSearchCost: webSearchCost > 0 ? webSearchCost : undefined,
        didGenerateImage: didGenerateImage,
        didWebSearch: didUseWebSearch,
        totalCost: inputCost + outputCost + webSearchCost,
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

    if (extractedText) {
      responseBlock.push({ type: "text", text: extractedText });
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
          mimeType: "image/png", // Assuming PNG
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
      temperature: undefined,
      systemPrompt: chat.systemPrompt,
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
    let currentImageGenerationId: string | null = null;
    let finalResponseId: string | undefined = undefined;
    let finalUsage: UsageResponse | undefined = undefined;
    const apiInput: OpenAIMessageInput[] = messages
      .map((msg): OpenAIMessageInput | null => {
        let role: "user" | "assistant" | "developer";
        switch (msg.role) {
          case "user":
          case "assistant":
            role = msg.role;
            break;
          case "system":
            return null;
          default:
            role = "user";
        }
        if (!Array.isArray(msg.content)) return null;

        const apiContentParts = msg.content
          .map((block): OpenAIMessageInput["content"][number] | null => {
            if (block.type === "text") {
              return {
                type: role === "assistant" ? "output_text" : "input_text",
                text: block.text,
              };
            }
            if (
              block.type === "image" &&
              this.isVisionCapable &&
              role === "user"
            ) {
              return { type: "input_image", image_url: block.url };
            }
            if (
              block.type === "image_data" &&
              this.isVisionCapable &&
              role === "user"
            ) {
              return {
                type: "input_image",
                image_url: `data:${block.mimeType};base64,${block.base64Data}`,
              };
            }
            return null;
          })
          .filter((part): part is OpenAIMessageContentPart => part !== null);

        if (apiContentParts.length === 0) return null;
        return { role, content: apiContentParts };
      })
      .filter((msg): msg is OpenAIMessageInput => msg !== null);

    if (apiInput.length === 0) {
      throw new Error(
        "No valid messages could be mapped for the OpenAI API request."
      );
    }

    let finalTools = tools ? [...tools] : [];
    if (options.openaiImageGenerationOptions) {
      const imageGenOptions = options.openaiImageGenerationOptions;
      const imageGenerationTool: any = {
        type: "image_generation",
        partial_images: 1,
      };
      if (imageGenOptions.quality && imageGenOptions.quality !== "auto") {
        imageGenerationTool.quality = imageGenOptions.quality;
      }
      if (imageGenOptions.size && imageGenOptions.size !== "auto") {
        imageGenerationTool.size = imageGenOptions.size;
      }
      if (imageGenOptions.background && imageGenOptions.background !== "auto") {
        imageGenerationTool.background = imageGenOptions.background;
      }
      finalTools.push(imageGenerationTool);
    }

    const reasoningParam = this.getReasoningParam(options.budgetTokens);

    console.log(
      `SNOWGANDER: sending this request: ${JSON.stringify({
        model: model,
        instructions: systemPrompt,
        previous_response_id: previousResponseId, // Pass state ID
        input: apiInput as any,
        tools: finalTools,
        store: store,
        stream: true,
        ...(reasoningParam as any),
      })}`
    );

    try {
      // First cast to unknown, then to AsyncIterable to satisfy TypeScript's type safety
      const stream = (await this.client.responses.create({
        model: model,
        instructions: systemPrompt,
        previous_response_id: previousResponseId, // Pass state ID
        input: apiInput as any,
        tools: finalTools,
        store: store,
        stream: true,
        ...(reasoningParam as any),
      })) as unknown as AsyncIterable<any>;

      for await (const event of stream) {
        console.log(
          `SNOWGANDER: Received stream event: ${JSON.stringify(event, null, 2)}`
        );
        switch (event.type) {
          case "response.output_text.delta":
            if (event.delta) {
              yield {
                type: "text",
                text: event.delta,
              };
            }
            break;

          case "response.image_generation_call.in_progress":
            console.log(`Image in progress: ${JSON.stringify(event)}`);
            if (event.item_id) {
              currentImageGenerationId = event.item_id;
              /*
              console.log(
                `[SNOWGANDER] Captured Image Generation ID: ${currentImageGenerationId}`
              );
              */
            }
            break;

          case "response.image_generation_call.partial_image":
            console.log(`Image Partial Received: Item id: ${event.item_id}`);
            if (event.partial_image_b64) {
              yield {
                type: "image_data",
                id: currentImageGenerationId, // Add the captured ID here
                mimeType: "image/png", // The API consistently generates PNGs
                base64Data: event.partial_image_b64,
              };
            }
            break;

          case "response.web_search_call.searching":
            yield {
              type: "thinking",
              thinking: "Searching the web...",
              signature: "openai",
            };
            break;

          // START of new code to add
          case "response.reasoning_summary_text.delta":
            if (event.delta) {
              yield {
                type: "thinking",
                thinking: event.delta,
                signature: "openai",
              };
            }
            break;
          // END of new code to add

          case "response.completed": // Capture final data
            finalResponseId = event.response.id;
            if (
              event.response.usage &&
              this.inputTokenCost &&
              this.outputTokenCost
            ) {
              const didGenerateImage = event.response.output?.some(
                (o: any) => o.type === "image_generation_call"
              );
              const didWebSearch = event.response.output?.some(
                (o: any) => o.type === "web_search_call"
              );
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
              const webSearchCost =
                didWebSearch && this.modelConfig.webSearchCost
                  ? this.modelConfig.webSearchCost
                  : 0;
              finalUsage = {
                inputCost,
                outputCost,
                webSearchCost: webSearchCost > 0 ? webSearchCost : undefined,
                didGenerateImage,
                didWebSearch,
                totalCost: inputCost + outputCost + webSearchCost,
              };
            }

            break;

          case "response.failed":
            const errorMessage =
              event.response?.error?.message || "Response failed in stream.";
            console.error(`OpenAI stream failed: ${errorMessage}`);
            yield {
              type: "error",
              publicMessage: "The request failed during streaming.",
              privateMessage: errorMessage,
            };
            return; // Terminate the generator on a failure event
        }
      }
      if (finalResponseId) {
        yield { type: "meta", responseId: finalResponseId, usage: finalUsage };
      }
    } catch (error: any) {
      console.error("Error during OpenAI stream:", error);
      yield {
        type: "error",
        publicMessage: "An error occurred while streaming the response.",
        privateMessage: error.message || String(error),
      };
    }
  }
}
