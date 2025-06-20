// src/vendors/google.ts (Corrected Version)

import {
  GoogleGenAI,
  GenerationConfig,
  Content,
  Part,
  GenerateContentResponse,
  Modality,
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
            });
          } else if (block.type === "image" && this.isVisionCapable) {
            // NOTE: The console.log statement has been removed from here.
            try {
              const imageData = await getImageDataFromUrl(block.url);
              if (imageData) {
                parts.push({
                  inlineData: {
                    mimeType: imageData.mimeType,
                    data: imageData.base64Data,
                  },
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

    if (contentBlocks.length === 0) {
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
    } = options;

    const generationConfig: GenerationConfig = {};
    if (maxTokens) {
      generationConfig.maxOutputTokens = maxTokens;
    }
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
    try {
      const responseStream = await this.client.models.generateContentStream(
        generateContentArgs
      );

      for await (const chunk of responseStream) {
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
            yield {
              type: "image_data",
              mimeType: part.inlineData.mimeType || "image/png",
              base64Data: part.inlineData.data || "",
            };
          }
        }
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
