import {
  GoogleGenAI,
  // Types likely needed from the new SDK - adjust based on actual SDK structure
  // GenerateContentRequest, // Removed import
  GenerationConfig,
  Content,
  Part,
  GenerateContentResponse, // Assuming this type exists
  UsageMetadata,
  Modality, // Assuming this type exists
  // Add other necessary types like FunctionDeclaration, Tool, etc. if needed later
} from "@google/genai"; // Updated import
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  ModelConfig,
  Chat,
  ChatResponse,
  ContentBlock,
  ImageDataBlock,
  Message,
  UsageResponse,
  NotImplementedError,
  ImageGenerationResponse,
} from "../types";
// Import the new utility function and remove axios import if no longer needed directly
import { computeResponseCost, getImageDataFromUrl } from "../utils";
// import axios from "axios"; // Removed as it's now handled in utils

export class GoogleAIAdapter implements AIVendorAdapter {
  private client: GoogleGenAI; // Keep client instance
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    // Initialization seems similar based on README
    this.client = new GoogleGenAI({ apiKey: config.apiKey });
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;
    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  // Helper to convert our Message format to Google's Content format
  // Structure might be slightly different in the new SDK, adjust if needed
  private mapMessagesToGoogleContent(messages: Message[]): Content[] {
    return messages
      .filter((msg) => msg.role !== "system")
      .map((msg) => {
        const role = msg.role === "assistant" ? "model" : "user";
        let parts: Part[] = [];

        if (typeof msg.content === "string") {
          parts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) {
          msg.content.forEach((block: ContentBlock) => {
            if (block.type === "text") {
              parts.push({ text: block.text });
            } else if (block.type === "image_data" && this.isVisionCapable) {
              parts.push({
                inlineData: {
                  mimeType: block.mimeType,
                  data: block.base64Data,
                },
              });
            } else if (block.type === "thinking") {
              // Represent thinking as text for now, until SDK confirms 'thought' part
              parts.push({ text: `${block.thinking}` });
            } else if (block.type === "redacted_thinking") {
              parts.push({
                text: `<redacted_thinking>${block.data}</redacted_thinking>`,
              });
            } else if (block.type === "image" && this.isVisionCapable) {
              console.warn(
                "Google adapter received 'image' block (URL), needs pre-processing to base64 inlineData."
              );
              // Pre-processing logic (getImageDataFromUrl) will handle this before this map function
            }
          });
        }
        if (parts.length === 0) {
          parts.push({ text: "" });
        }
        return { role, parts };
      });
  }

  // Removed the private getImageDataFromUrl method

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const {
      model,
      messages,
      maxTokens,
      budgetTokens,
      systemPrompt,
      visionUrl,
    } = options;

    // Prepare generationConfig - structure might differ slightly
    const generationConfig: GenerationConfig = {}; // Use GenerationConfig from @google/genai
    if (maxTokens) {
      generationConfig.maxOutputTokens = maxTokens;
    }
    // Add thinkingConfig using 'as any' if SDK types don't support it yet
    if (this.isThinkingCapable) {
      (generationConfig as any).thinkingConfig = {
        includeThoughts: true,
      };
      if (budgetTokens && budgetTokens > 0) {
        (generationConfig as any).thinkingConfig.thinkingBudget = budgetTokens;
      } else if (budgetTokens === 0) {
        (generationConfig as any).thinkingConfig.thinkingBudget = 0;
      }
    }
    // Image generation modality - check if new SDK handles this differently
    if (this.isImageGenerationCapable) {
      // This might need adjustment based on the new SDK's capabilities/API
      (generationConfig as any).responseModalities = ["Text", "Image"];
    }

    // Get model instance using the new SDK structure
    // Note: The new SDK might not require getting a model instance first,
    // and might pass config directly to generateContent. Adjust if needed.
    // const genAIModel = this.client.getGenerativeModel({ model }); // Old way

    let formattedMessages = this.mapMessagesToGoogleContent(messages);

    // Prepend system prompt if provided (new SDK might have a dedicated field)
    // Check if the new SDK has a specific systemInstruction field in generateContent options
    let systemInstructionContent: Content | undefined = undefined;
    if (systemPrompt) {
      systemInstructionContent = {
        role: "system",
        parts: [{ text: systemPrompt }],
      };
      // If the new SDK uses a dedicated field instead of prepending:
      // formattedMessages = this.mapMessagesToGoogleContent(messages); // Don't prepend here
    } else {
      formattedMessages = this.mapMessagesToGoogleContent(messages);
    }

    // --- Vision Handling ---
    let visionPart: Part | null = null;
    if (visionUrl && this.isVisionCapable) {
      // Call the imported utility function
      const imageData = await getImageDataFromUrl(visionUrl);
      if (imageData) {
        visionPart = {
          inlineData: {
            mimeType: imageData.mimeType,
            data: imageData.base64Data,
          },
        };
      } else {
        console.warn(`Could not process image from visionUrl: ${visionUrl}`);
      }
    }

    // Add vision part to the last user message
    if (visionPart) {
      const lastMessageIndex = formattedMessages.length - 1;
      if (
        lastMessageIndex >= 0 &&
        formattedMessages[lastMessageIndex].role === "user" &&
        formattedMessages[lastMessageIndex].parts
      ) {
        formattedMessages[lastMessageIndex].parts.push(visionPart);
      } else {
        // If no user message exists, create one just for the image
        formattedMessages.push({ role: "user", parts: [visionPart] });
        console.warn(
          "Image data provided but no user message found. Created new user message for image."
        );
      }
    }
    // --- End Vision Handling ---

    // Prepare the arguments for the generateContent call directly
    const generateContentArgs: any = {
      // Use 'any' temporarily for easier property addition
      model: model,
      contents: formattedMessages,
      generationConfig: generationConfig,
    };
    // Conditionally add systemInstruction if it exists
    if (systemInstructionContent) {
      generateContentArgs.systemInstruction = systemInstructionContent;
    }

    // This is the new way that GenAI handles modalities. Old way left
    // for compatibility
    if (this.isImageGenerationCapable) {
      generateContentArgs.config = {
        responseModalities: [Modality.TEXT, Modality.IMAGE], // <-- Use Modality enum
      };
    }

    // Call generateContent using the new SDK structure: ai.models.generateContent
    // The response structure might also differ.
    let result: GenerateContentResponse; // Use type from @google/genai
    try {
      // Pass the constructed arguments object directly
      result = await this.client.models.generateContent(generateContentArgs);
    } catch (error) {
      console.error("Error calling Google GenAI generateContent:", error);
      throw error; // Re-throw the error
    }

    // Access response - structure might differ (e.g., result.candidates instead of result.response.candidates)
    // Check the actual response object structure from the new SDK
    const responseCandidate = result?.candidates?.[0]; // Use optional chaining
    const responseUsageMetadata = result?.usageMetadata; // Use optional chaining

    // Add check for responseCandidate before accessing its parts
    if (!responseCandidate || !responseCandidate.content?.parts) {
      console.error(
        "Invalid response structure from @google/genai API (missing candidate or parts):",
        JSON.stringify(result) // Log the whole result for debugging
      );
      // Attempt to get text directly as a fallback
      try {
        // Explicitly check responseCandidate again inside the try block
        if (responseCandidate && responseCandidate.content?.parts) {
          const fallbackText =
            responseCandidate.content.parts.map((p) => p.text).join("") || "";
          if (fallbackText) {
            return {
              role: "assistant",
              content: [{ type: "text", text: fallbackText }],
              usage: undefined, // Usage might be unavailable in this error case
            };
          }
        }
      } catch (e) {
        // Ignore fallback error
      }
      // If fallback also failed or responseCandidate was null/undefined initially
      throw new Error("Invalid response structure from @google/genai API");
    }

    // --- Usage Calculation ---
    let usage: UsageResponse | undefined = undefined;
    // Check if usageMetadata exists and has the expected properties
    if (responseUsageMetadata && this.inputTokenCost && this.outputTokenCost) {
      // Property names might differ in the new SDK (e.g., promptTokenCount vs prompt_token_count)
      const promptTokens = responseUsageMetadata.promptTokenCount ?? 0;
      const candidateTokens = responseUsageMetadata.candidatesTokenCount ?? 0; // Or totalTokenCount - promptTokens? Check SDK docs.

      const inputCost = computeResponseCost(promptTokens, this.inputTokenCost);
      const outputCost = computeResponseCost(
        candidateTokens,
        this.outputTokenCost
      );

      usage = {
        inputCost: inputCost,
        outputCost: outputCost,
        totalCost: inputCost + outputCost,
      };
    }
    // --- End Usage Calculation ---

    // --- Content Parsing ---
    const contentBlocks: ContentBlock[] = [];
    for (const part of responseCandidate.content.parts) {
      // Check for thinking part (API reference confirms 'thought' exists on Part)
      if (part.thought === true) {
        contentBlocks.push({
          type: "thinking",
          thinking: part.text || "", // Assume thought content is in text
          signature: "google",
        });
      } else if (part.text) {
        contentBlocks.push({ type: "text", text: part.text });
      } else if (part.inlineData) {
        contentBlocks.push({
          type: "image_data",
          mimeType: part.inlineData.mimeType,
          base64Data: part.inlineData.data,
        } as ImageDataBlock);
      }
      // Add handling for other part types if needed (e.g., functionCall, functionResponse)
    }

    if (contentBlocks.length === 0) {
      // Fallback if no parts were processed but the structure seemed valid initially
      const fallbackText = responseCandidate.content.parts
        .map((p) => p.text)
        .join("");
      if (fallbackText) {
        contentBlocks.push({ type: "text", text: fallbackText });
      } else {
        console.error(
          "No processable content found in Gemini response parts:",
          responseCandidate.content.parts
        );
        throw new Error("No processable content found in Gemini response.");
      }
    }
    // --- End Content Parsing ---

    return {
      role: "assistant",
      content: contentBlocks,
      usage: usage,
    };
  }

  async generateImage(
    options: AIRequestOptions
  ): Promise<ImageGenerationResponse> {
    throw new NotImplementedError(
      "Google generates images inline with text, use generateResponse/sendChat."
    );
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    // This method should now correctly use the refactored generateResponse
    const options: AIRequestOptions = {
      model: chat.model,
      messages: [...chat.responseHistory], // Clone history
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined,
      systemPrompt: chat.systemPrompt,
      visionUrl: chat.visionUrl || undefined,
    };

    // Add current prompt to messages if it exists
    /*
    if (chat.prompt) {
      const currentMessage: Message = {
        role: "user",
        // Handle prompt as simple text for now, visionUrl is handled in generateResponse
        content: [{ type: "text", text: chat.prompt }],
      };
      options.messages.push(currentMessage);
    }
      */

    const response = await this.generateResponse(options);

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  // Function calling / MCP tool handling would need to be added here
  // by integrating with the `tools` and `toolConfig` parameters
  // in the `generateResponse` method, similar to how Anthropic adapter does it,
  // but using the @google/genai SDK's specific structures for Tool and FunctionDeclaration.
}
