import {
  GoogleGenAI,
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

  /**
   * Maps internal Message format to Google's Content format.
   * Handles downloading and converting images for vision models.
   */
  private async mapMessagesToGoogleContent(
    messages: Message[]
  ): Promise<Content[]> {
    const googleContents: Content[] = [];

    for (const msg of messages) {
      if (msg.role === "system") continue;

      const role = msg.role === "assistant" ? "model" : "user";
      const parts: Part[] = [];

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
              mediaResolution: { level: PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH },
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
                  mediaResolution: { level: PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH },
                });
              }
            } catch (error) {
              console.error(
                `Failed to fetch and process image URL ${block.url}:`,
                error
              );
              parts.push({ text: `[Error: Failed to load image from URL]` });
            }
          } else if (
            block.type === "thinking" ||
            block.type === "redacted_thinking"
          ) {
            // Generally we don't send thinking blocks back to the model unless
            // explicitly constructing a thought history, but usually we skip them
            // or send them as text if needed. Skipping for now.
            continue;
          }
        }
      }

      if (parts.length > 0) {
        googleContents.push({ role, parts });
      }
    }

    return googleContents;
  }

  /**
   * Helper to build the unified request arguments for @google/genai.
   * Correctly places configuration in 'config' and system instructions at the root.
   */
  private buildGenerateContentArgs(
    options: AIRequestOptions,
    formattedMessages: Content[],
    systemInstructionContent?: Content
  ) {
    const { model, maxTokens, budgetTokens, useImageGeneration } = options;

    // The 'config' object maps to 'generationConfig' in the API
    const config: any = {};

    if (maxTokens) {
      config.maxOutputTokens = maxTokens;
    }

    if (this.isThinkingCapable) {
      config.thinkingConfig = {
        includeThoughts: true,
      };
      if (budgetTokens && budgetTokens > 0) {
        config.thinkingConfig.thinkingBudget = budgetTokens;
      } else if (budgetTokens === 0) {
        // Budget of 0 effectively disables thinking if the model allows it
        config.thinkingConfig.thinkingBudget = 0;
      }
    }

    // Handle Image Generation Modalities
    // Gemini 3 Pro (Nano Banana Pro) requires explicitly asking for IMAGE modality
    if (this.isImageGenerationCapable && useImageGeneration) {
      config.responseModalities = [Modality.TEXT, Modality.IMAGE];
    }

    const args: any = {
      model: model,
      contents: formattedMessages,
      config: config, // Correct placement for generation parameters
    };

    if (systemInstructionContent) {
      args.systemInstruction = systemInstructionContent; // Correct placement for system instruction
    }

    return args;
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { messages, systemPrompt } = options;

    const formattedMessages = await this.mapMessagesToGoogleContent(messages);
    let systemInstructionContent: Content | undefined = undefined;
    if (systemPrompt) {
      systemInstructionContent = {
        role: "system",
        parts: [{ text: systemPrompt }],
      };
    }

    const generateContentArgs = this.buildGenerateContentArgs(
      options,
      formattedMessages,
      systemInstructionContent
    );

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
      // Attempt to salvage text if present in a weird format, otherwise throw
      throw new Error("Invalid response structure from @google/genai API");
    }

    // Calculate usage
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
      // Check for thinking content
      // Note: The SDK/API might return thought in different ways depending on version
      // but 'thought: true' or specific part types are standard indicators.
      if ((part as any).thought === true) {
        contentBlocks.push({
          type: "thinking",
          thinking: part.text || "",
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
    }

    // Fallback if no parts were processed (e.g., pure text without explicit types)
    if (contentBlocks.length === 0) {
      const fallbackText = responseCandidate.content.parts
        .map((p) => p.text)
        .join("");
      if (fallbackText) {
        contentBlocks.push({ type: "text", text: fallbackText });
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

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const messages: Message[] = [...chat.responseHistory];
    const currentUserContent: ContentBlock[] = [];

    if (chat.prompt) {
      currentUserContent.push({ type: "text", text: chat.prompt });
    }
    if (chat.visionUrl) {
      currentUserContent.push({ type: "image", url: chat.visionUrl });
    }

    if (currentUserContent.length > 0) {
      messages.push({
        role: "user",
        content: currentUserContent,
      });
    }

    const options: AIRequestOptions = {
      model: chat.model,
      messages: messages,
      maxTokens: chat.maxTokens || undefined,
      budgetTokens: chat.budgetTokens || undefined,
      systemPrompt: chat.systemPrompt,
      useImageGeneration: true, // Critical for "Nano Banana Pro" image features
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
    const { messages, systemPrompt, useImageGeneration } = options;

    const formattedMessages = await this.mapMessagesToGoogleContent(messages);
    let systemInstructionContent: Content | undefined = undefined;
    if (systemPrompt) {
      systemInstructionContent = {
        role: "system",
        parts: [{ text: systemPrompt }],
      };
    }

    const generateContentArgs = this.buildGenerateContentArgs(
      options,
      formattedMessages,
      systemInstructionContent
    );

    try {
      // -----------------------------------------------------------------------
      // FIX: Handle Image Generation via Non-Streaming Fallback
      // Google GenAI (Imagen 3/Gemini) typically does NOT support streaming
      // when generating images. We must use the standard generateContent
      // and simulate a stream.
      // -----------------------------------------------------------------------
      if (this.isImageGenerationCapable && useImageGeneration) {
        const result = await this.client.models.generateContent(
          generateContentArgs
        );

        const candidate = result?.candidates?.[0];
        const usageMetadata = result?.usageMetadata;

        if (candidate && candidate.content?.parts) {
          for (const part of candidate.content.parts) {
            if ((part as any).thought === true) {
              yield {
                type: "thinking",
                thinking: part.text || "",
                signature: "google",
              };
            } else if (part.text) {
              yield {
                type: "text",
                text: part.text,
              };
            } else if (part.inlineData) {
              yield {
                type: "image_data",
                mimeType: part.inlineData.mimeType || "image/png",
                base64Data: part.inlineData.data || "",
              };
            }
          }
        }

        // Yield usage at the end for the simulated stream
        if (usageMetadata && this.inputTokenCost && this.outputTokenCost) {
          const promptTokens = usageMetadata.promptTokenCount ?? 0;
          const candidateTokens = usageMetadata.candidatesTokenCount ?? 0;
          const inputCost = computeResponseCost(
            promptTokens,
            this.inputTokenCost
          );
          const outputCost = computeResponseCost(
            candidateTokens,
            this.outputTokenCost
          );

          yield {
            type: "meta",
            responseId: `google-gen-${Date.now()}`,
            usage: {
              inputCost,
              outputCost,
              totalCost: inputCost + outputCost,
              didGenerateImage: true,
            },
          };
        }
        return; // End generator
      }

      // -----------------------------------------------------------------------
      // Standard Streaming (Text/Thinking)
      // -----------------------------------------------------------------------
      const responseStream = await this.client.models.generateContentStream(
        generateContentArgs
      );

      let finalPromptTokens = 0;
      let finalCandidateTokens = 0;
      let responseId: string | undefined = undefined;

      for await (const chunk of responseStream) {
        if (chunk.usageMetadata) {
          finalPromptTokens =
            chunk.usageMetadata.promptTokenCount ?? finalPromptTokens;
          finalCandidateTokens =
            chunk.usageMetadata.candidatesTokenCount ?? finalCandidateTokens;
        }

        const candidate = chunk.candidates?.[0];
        if (!candidate || !candidate.content?.parts) {
          continue;
        }

        for (const part of candidate.content.parts) {
          if ((part as any).thought === true) {
            yield {
              type: "thinking",
              thinking: part.text || "",
              signature: "google",
            };
          } else if (part.text) {
            yield {
              type: "text",
              text: part.text,
            };
          } else if (part.inlineData) {
            // Unlikely to be hit in streaming mode, but good safety
            yield {
              type: "image_data",
              mimeType: part.inlineData.mimeType || "image/png",
              base64Data: part.inlineData.data || "",
            };
          }
        }
      }

      // Yield final meta block
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
        yield {
          type: "meta",
          responseId: responseId || `google-stream-${Date.now()}`,
          usage: {
            inputCost: inputCost,
            outputCost: outputCost,
            totalCost: inputCost + outputCost,
          },
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
