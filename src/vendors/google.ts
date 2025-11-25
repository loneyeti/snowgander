// src/vendors/google.ts (Corrected Version)

import {
  GoogleGenAI,
  GenerationConfig,
  Content,
  Part,
  GenerateContentResponse,
  Modality,
  PartMediaResolutionLevel,
} from "@google/genai";
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
import { computeResponseCost, getImageDataFromUrl } from "../utils";

export class GoogleAIAdapter implements AIVendorAdapter {
  private client: GoogleGenAI;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number | undefined;
  public outputTokenCost?: number | undefined;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.client = new GoogleGenAI({ apiKey: config.apiKey });
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;
    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  private async mapMessagesToGoogleContent(
    messages: Message[]
  ): Promise<Content[]> {
    const googleContents: Content[] = [];

    for (const msg of messages) {
      if (msg.role === "system") continue;

      const role = msg.role === "assistant" ? "model" : "user";
      const parts: Part[] = [];

      const attachSignature = (signature: string) => {
        if (parts.length > 0) {
          (parts[parts.length - 1] as any).thoughtSignature = signature;
        } else {
          // Edge case: Signature came before content or standalone. 
          // We attach it to a dummy empty text part if needed, or hold it.
          // Gemini 3 docs say signature belongs to a Part.
          parts.push({ text: "", thoughtSignature: signature } as any);
        }
      };

      if (typeof msg.content === "string") {
        parts.push({ text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === "text") {
            parts.push({ text: block.text });
          } else if (block.type === "image_data" && this.isVisionCapable) {
            parts.push({
              inlineData: {
                mimeType: block.mimeType,
                data: block.base64Data,
              },
              mediaResolution: { level: PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH } 
            });
          } else if (block.type === "image" && this.isVisionCapable) {
            try {
              const imageData = await getImageDataFromUrl(block.url);
              if (imageData) {
                parts.push({
                  inlineData: {
                    mimeType: imageData.mimeType,
                    data: imageData.base64Data,
                  },
                  mediaResolution: { level: PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH } 
                });
              }
            } catch (error) {
              console.error(
                `Failed to fetch and process image URL ${block.url}:`,
                error
              );
              parts.push({ text: `[Error: Failed to load image from URL]` });
            }
          } else if (block.type === "thinking") {
            // CRITICAL FOR GEMINI 3:
            // If we have a signature preserved in the thinking block, we must send it back.
            // The signature belongs to the *content* part, so we attach it to the last added part.
            if (block.signature) {
              attachSignature(block.signature);
            }
            // We generally don't send the "thinking" text back to the model in the history
            // for Gemini, as it manages its own thought process, but the signature is mandatory.
          }
        }
      }

      if (parts.length > 0) {
        googleContents.push({ role, parts });
      }
    }
    return googleContents;
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const {
      model,
      messages,
      maxTokens,
      budgetTokens,
      systemPrompt,
      useImageGeneration,
    } = options;

    const generationConfig: GenerationConfig = {};
    if (maxTokens) {
      generationConfig.maxOutputTokens = maxTokens;
    }
    // --- GEMINI 3 THINKING CONFIGURATION ---
    if (this.isThinkingCapable) {
      (generationConfig as any).thinkingConfig = {
        includeThoughts: true,
      };

      // Map budgetTokens to thinking_level
      if (budgetTokens === 0) {
        // Explicitly disabled/low reasoning
        (generationConfig as any).thinkingConfig.thinking_level = "low";
      } else {
        // Default to high if thinking is enabled, as Gemini 3 is dynamic/high by default
        // We ignore the specific 'number' of budgetTokens as Gemini 3 uses levels.
        (generationConfig as any).thinkingConfig.thinking_level = "high";
      }
    }
    if (this.isImageGenerationCapable) {
      (generationConfig as any).responseModalities = ["Text", "Image"];
    }

    const formattedMessages = await this.mapMessagesToGoogleContent(messages);

    let systemInstructionContent: Content | undefined = undefined;
    if (systemPrompt) {
      systemInstructionContent = {
        role: "system",
        parts: [{ text: systemPrompt }],
      };
    }

    const generateContentArgs: any = {
      model: model,
      contents: formattedMessages,
      generationConfig: generationConfig,
    };
    if (systemInstructionContent) {
      generateContentArgs.systemInstruction = systemInstructionContent;
    }

    if (this.isImageGenerationCapable && useImageGeneration) {
      generateContentArgs.config = {
        responseModalities: [Modality.TEXT, Modality.IMAGE],
      };
    }

    let result: GenerateContentResponse;
    try {
      result = await this.client.models.generateContent(generateContentArgs);
    } catch (error) {
      console.error("Error calling Google GenAI generateContent:", error);
      throw error;
    }

    const responseCandidate = result?.candidates?.[0];
    const responseUsageMetadata = result?.usageMetadata;

    if (!responseCandidate || !responseCandidate.content?.parts) {
      console.error(
        "Invalid response structure from @google/genai API (missing candidate or parts):",
        JSON.stringify(result)
      );
      try {
        if (responseCandidate && responseCandidate.content?.parts) {
          const fallbackText =
            responseCandidate.content.parts.map((p) => p.text).join("") || "";
          if (fallbackText) {
            return {
              role: "assistant",
              content: [{ type: "text", text: fallbackText }],
              usage: undefined,
            };
          }
        }
      } catch (e) {
        // Ignore fallback error
      }
      throw new Error("Invalid response structure from @google/genai API");
    }

    let usage: UsageResponse | undefined = undefined;
    if (responseUsageMetadata && this.inputTokenCost && this.outputTokenCost) {
      const promptTokens = responseUsageMetadata.promptTokenCount ?? 0;
      const candidateTokens = responseUsageMetadata.candidatesTokenCount ?? 0;
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

    const contentBlocks: ContentBlock[] = [];
    for (const part of responseCandidate.content.parts) {
      const partAny = part as any;
      
      // 1. Capture Thought Signature
      // Gemini 3 attaches signatures to parts. We store this in a ThinkingBlock.
      // Even if there is no visible "thought" text, we need the signature.
      if (partAny.thoughtSignature) {
        contentBlocks.push({
          type: "thinking",
          thinking: partAny.thought === true ? part.text || "" : "", // Only show text if it's actual thought text
          signature: partAny.thoughtSignature,
        });
      }

      // 2. Handle Content
      if (partAny.thought === true) {
        // If we didn't capture the signature above (e.g. legacy thought), capture text here
        if (!partAny.thoughtSignature) {
           contentBlocks.push({
            type: "thinking",
            thinking: part.text || "",
            signature: "google-legacy",
          });
        }
      } else if (part.text) {
        contentBlocks.push({ type: "text", text: part.text });
      } else if (part.inlineData) {
        contentBlocks.push({
          type: "image_data",
          mimeType: part.inlineData.mimeType,
          base64Data: part.inlineData.data,
        } as ImageDataBlock);
      }
    }

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

  // **** THIS IS THE CORRECTED METHOD ****
  async sendChat(chat: Chat): Promise<ChatResponse> {
    // Start with a copy of the history
    const messages: Message[] = [...chat.responseHistory];

    // Create the new user message content
    const currentUserContent: ContentBlock[] = [];

    // Add the text prompt if it exists
    if (chat.prompt) {
      currentUserContent.push({ type: "text", text: chat.prompt });
    }

    // Add the image from visionUrl if it exists
    if (chat.visionUrl) {
      currentUserContent.push({ type: "image", url: chat.visionUrl });
    }

    // Add the new user message to the array if it has content
    if (currentUserContent.length > 0) {
      messages.push({
        role: "user",
        content: currentUserContent,
      });
    }

    // Prepare the options for generateResponse
    const options: AIRequestOptions = {
      model: chat.model,
      messages: messages, // Pass the newly constructed messages array
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined,
      systemPrompt: chat.systemPrompt,
      // visionUrl is no longer passed here
    };

    const response = await this.generateResponse(options);

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const {
      model,
      messages,
      maxTokens,
      budgetTokens,
      systemPrompt,
      useImageGeneration,
      temperature,
    } = options;

    const generationConfig: GenerationConfig = {};
    if (maxTokens) {
      generationConfig.maxOutputTokens = maxTokens;
    }

    // --- GEMINI 3 THINKING CONFIGURATION ---
    if (this.isThinkingCapable) {
      (generationConfig as any).thinkingConfig = {
        includeThoughts: true,
      };

      // Map budgetTokens to thinking_level
      if (budgetTokens === 0) {
        // Explicitly disabled/low reasoning (fastest)
        (generationConfig as any).thinkingConfig.thinking_level = "low";
      } else {
        // Default to high if thinking is enabled. 
        // Gemini 3 uses 'high' for deep reasoning, 'medium' is coming soon.
        (generationConfig as any).thinkingConfig.thinking_level = "high";
      }
    }

    // Gemini 3 recommends Temperature 1.0. 
    if (temperature !== undefined) {
      generationConfig.temperature = temperature;
    }

    if (this.isImageGenerationCapable) {
      (generationConfig as any).responseModalities = ["Text", "Image"];
    }

    // Use the updated mapping logic (includes media_resolution_high handling)
    const formattedMessages = await this.mapMessagesToGoogleContent(messages);

    let systemInstructionContent: Content | undefined = undefined;
    if (systemPrompt) {
      systemInstructionContent = {
        role: "system",
        parts: [{ text: systemPrompt }],
      };
    }

    const generateContentArgs: any = {
      model: model,
      contents: formattedMessages,
      generationConfig: generationConfig,
    };
    if (systemInstructionContent) {
      generateContentArgs.systemInstruction = systemInstructionContent;
    }
    if (this.isImageGenerationCapable && useImageGeneration) {
      generateContentArgs.config = {
        responseModalities: [Modality.TEXT, Modality.IMAGE],
      };
    }

    try {
      const responseStream = await this.client.models.generateContentStream(
        generateContentArgs
      );

      let finalPromptTokens = 0;
      let finalCandidateTokens = 0;
      let finalTotalTokens = 0;
      let responseId: string | undefined = undefined;

      for await (const chunk of responseStream) {
        // Track usage metadata if present
        if (chunk.usageMetadata) {
          finalPromptTokens =
            chunk.usageMetadata.promptTokenCount ?? finalPromptTokens;
          finalCandidateTokens =
            chunk.usageMetadata.candidatesTokenCount ?? finalCandidateTokens;
          finalTotalTokens =
            chunk.usageMetadata.totalTokenCount ?? finalTotalTokens;
        }

        const candidate = chunk.candidates?.[0];

        if (!candidate || !candidate.content?.parts) {
          continue;
        }

        for (const part of candidate.content.parts) {
          const partAny = part as any;

          // 1. CRITICAL: Capture Thought Signature
          // Gemini 3 sends signatures, sometimes in empty text chunks at the end.
          if (partAny.thoughtSignature) {
            yield {
              type: "thinking",
              // Only include text if 'thought' is also true, otherwise it's just a signature container
              thinking: partAny.thought === true ? part.text || "" : "",
              signature: partAny.thoughtSignature,
            };
          } 
          // 2. Standard Thinking (Text without explicit new signature field yet)
          else if (partAny.thought === true) {
            yield {
              type: "thinking",
              thinking: part.text || "",
              // Fallback if no signature provided in this chunk
              signature: "google", 
            };
          } 
          // 3. Standard Text
          else if (part.text) {
            yield {
              type: "text",
              text: part.text,
            };
          } 
          // 4. Inline Images
          else if (part.inlineData) {
            yield {
              type: "image_data",
              mimeType: part.inlineData.mimeType || "image/png",
              base64Data: part.inlineData.data || "",
            };
          }
        }
      }

      // Yield final meta block with usage data if available
      if (
        this.inputTokenCost &&
        this.outputTokenCost &&
        (finalPromptTokens > 0 || finalCandidateTokens > 0)
      ) {
        const inputCost = computeResponseCost(
          finalPromptTokens,
          this.inputTokenCost
        );
        const outputCost = computeResponseCost(
          finalCandidateTokens,
          this.outputTokenCost
        );
        const usage: UsageResponse = {
          inputCost: inputCost,
          outputCost: outputCost,
          totalCost: inputCost + outputCost,
        };

        yield {
          type: "meta",
          responseId: responseId || `google-stream-${Date.now()}`,
          usage: usage,
        };
      }
    } catch (error) {
      console.error("Error during Google GenAI stream:", error);
      yield {
        type: "error",
        publicMessage: "An error occurred while streaming the response.",
        privateMessage: error instanceof Error ? error.message : String(error),
      };
    }
  }
}
