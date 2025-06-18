import OpenAI, { toFile } from "openai"; // Import toFile
import fetch from "node-fetch"; // Need fetch for handling URL images
import { Buffer } from "buffer"; // Need Buffer for base64 decoding
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  Chat,
  ChatResponse,
  ContentBlock,
  ImageGenerationResponse,
  ImageEditResponse, // Keep this if used elsewhere, or remove if only OpenAIImageEditOptions is needed
  OpenAIImageEditOptions, // Add this back
  ImageDataBlock,
  ImageBlock,
  MCPAvailableTool,
  Message, // Added for constructing AIRequestOptions in sendChat
  ModelConfig,
  UsageResponse,
  VendorConfig,
  NotImplementedError,
  ErrorBlock,
} from "../types";
// Removed OpenAIImageEditOptions from import as it's defined in types.ts

// Helper function to create error responses
const createErrorResponse = (
  publicMsg: string,
  privateMsg: string,
  role: "assistant" | "error" = "error" // Default role to 'error' for clarity
): AIResponse | ChatResponse => {
  const errorBlock: ErrorBlock = {
    type: "error",
    publicMessage: publicMsg,
    privateMessage: privateMsg,
  };
  // Determine return type based on expected context (AIResponse vs ChatResponse)
  // For simplicity here, we'll return a structure compatible with both where possible
  return {
    role: role,
    content: [errorBlock],
    usage: { inputCost: 0, outputCost: 0, totalCost: 0 }, // Ensure usage is present
  };
};

export class OpenAIImageAdapter implements AIVendorAdapter {
  private client: OpenAI;
  readonly vendor = "openai-image";
  private modelConfig: ModelConfig;
  readonly isVisionCapable: boolean = false;
  readonly isImageGenerationCapable: boolean = true;
  readonly isThinkingCapable: boolean = false;

  constructor(vendorConfig: VendorConfig, modelConfig: ModelConfig) {
    if (!vendorConfig.apiKey) {
      throw new Error(`${this.vendor} API key is required.`);
    }
    this.client = new OpenAI({
      apiKey: vendorConfig.apiKey,
      organization: vendorConfig.organizationId,
      baseURL: vendorConfig.baseURL,
    });
    this.modelConfig = modelConfig;
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const errMsg = `${this.vendor} adapter does not support generateResponse. Use generateImage or editImage.`;
    console.error("generateResponse called:", errMsg);
    return createErrorResponse(
      "This model cannot generate text responses.",
      errMsg
    ) as AIResponse;
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const {
      prompt,
      model,
      openaiImageGenerationOptions,
      openaiImageEditOptions,
      responseHistory, // Needed to build messages for AIRequestOptions
      systemPrompt, // Needed for AIRequestOptions
    } = chat;
    console.log(`snowgander chat: ${chat}`);
    // Construct basic messages array from history and current prompt
    const messages: Message[] = responseHistory.map((res) => ({
      role: res.role,
      content: res.content,
    }));
    if (prompt) {
      messages.push({
        role: "user",
        content: [{ type: "text", text: prompt }],
      });
    }

    const baseOptions: Partial<AIRequestOptions> = {
      model: model || this.modelConfig.apiName,
      messages: messages,
      prompt: prompt, // Pass prompt along
      systemPrompt: systemPrompt, // Pass system prompt
      // Note: maxTokens, temperature etc. from Chat are not directly applicable here
      // but could be added if needed for future API versions or cost estimation.
    };

    try {
      // --- Handle visionUrl for Image Editing ---
      if (chat.visionUrl) {
        console.log(
          `${this.vendor}: visionUrl detected, attempting image edit.`
        );
        const editOptionsFromChat = chat.openaiImageEditOptions;

        if (!editOptionsFromChat) {
          const errMsg = `${
            this.vendor
          } adapter requires openaiImageEditOptions when visionUrl is provided for image editing. Chat: ${JSON.stringify(
            chat
          )}`;
          console.error("sendChat error:", errMsg);
          return createErrorResponse(
            "Image editing options missing for visionUrl.",
            errMsg
          ) as ChatResponse;
        }

        // Create ImageBlock from visionUrl
        const inputImageBlock: ImageBlock = {
          type: "image",
          url: chat.visionUrl,
        };

        // Use provided edit options but override the image source
        const finalEditOptions: OpenAIImageEditOptions = {
          ...editOptionsFromChat,
          image: [inputImageBlock], // Override with visionUrl image
        };

        const options: AIRequestOptions = {
          ...baseOptions, // Contains model, messages, prompt, systemPrompt
          openaiImageEditOptions: finalEditOptions,
        } as AIRequestOptions; // Cast needed because baseOptions is partial

        // Call editImage
        console.log(
          `${this.vendor}: Calling editImage with visionUrl as input.`
        );
        const result = await this.editImage(options);
        return {
          role: result.role,
          content: result.content,
          usage: result.usage,
        };
      }
      // --- End visionUrl Handling ---
      // Original logic if visionUrl is not present
      else if (openaiImageGenerationOptions) {
        const options: AIRequestOptions = {
          ...baseOptions, // baseOptions already defined above
          openaiImageGenerationOptions: openaiImageGenerationOptions,
        } as AIRequestOptions;

        // generateImage now returns AIResponse, potentially with ErrorBlock
        const result = await this.generateImage(options);
        // Map AIResponse to ChatResponse structure
        return {
          role: result.role,
          content: result.content,
          usage: result.usage,
        };
      } else if (openaiImageEditOptions && this.editImage) {
        const options: AIRequestOptions = {
          ...baseOptions,
          openaiImageEditOptions: openaiImageEditOptions,
        } as AIRequestOptions;

        // editImage now returns AIResponse, potentially with ErrorBlock
        const result = await this.editImage(options);
        // Map AIResponse (containing images or error) to ChatResponse structure
        return {
          role: result.role, // Will be 'error' if editImage returned an ErrorBlock
          content: result.content, // Will be ErrorBlock[] or ImageDataBlock[]
          usage: result.usage,
        };
      } else {
        const errMsg = `${this.vendor} adapter requires either openaiImageGenerationOptions or openaiImageEditOptions to be provided in the Chat object via sendChat. Chat: ${chat}`;
        console.error("sendChat error:", errMsg);
        return createErrorResponse(
          "Image generation/editing options missing.",
          errMsg
        ) as ChatResponse;
      }
    } catch (error: any) {
      // Catch unexpected errors during the process
      const errMsg = `Unexpected error in sendChat: ${error.message || error}`;
      console.error(errMsg, error);
      return createErrorResponse(
        "An unexpected error occurred.",
        errMsg
      ) as ChatResponse;
    }
  }

  async sendMCPChat(
    chat: Chat,
    tools: MCPAvailableTool[],
    options?: AIRequestOptions | undefined
  ): Promise<ChatResponse> {
    const errMsg = `${this.vendor} adapter does not support sendMCPChat.`;
    console.error("sendMCPChat called:", errMsg);
    return createErrorResponse(
      "Tool use is not supported by this model.",
      errMsg
    ) as ChatResponse;
  }

  // generateImage now returns AIResponse instead of ImageGenerationResponse
  // to accommodate potential ErrorBlocks
  async generateImage(options: AIRequestOptions): Promise<AIResponse> {
    const {
      prompt: textPrompt,
      messages,
      openaiImageGenerationOptions,
      model: modelNameFromOptions,
    } = options;
    const model = modelNameFromOptions || this.modelConfig.apiName;
    console.log(`Starting generateImage with ${model}`);

    // Extract prompt
    let prompt = textPrompt;
    if (!prompt && messages && messages.length > 0) {
      const lastUserMessage = messages.filter((m) => m.role === "user").pop();
      if (
        lastUserMessage &&
        lastUserMessage.content.length > 0 &&
        lastUserMessage.content[0].type === "text"
      ) {
        prompt = lastUserMessage.content[0].text;
      }
    }

    console.log(`generateImage prompt: ${prompt}`);

    if (!prompt) {
      const errMsg = "A text prompt is required for image generation.";
      console.error("generateImage error:", errMsg);
      return createErrorResponse("Prompt missing.", errMsg);
    }

    if (!openaiImageGenerationOptions) {
      const errMsg =
        "openaiImageGenerationOptions are required for OpenAI image generation.";
      console.error("generateImage error:", errMsg);
      return createErrorResponse("Image options missing.", errMsg);
    }

    try {
      // Map adapter options to OpenAI API params
      const apiOptions: OpenAI.Images.ImageGenerateParams = {
        model: model,
        prompt: prompt,
        n: openaiImageGenerationOptions.n,
        quality:
          openaiImageGenerationOptions.quality === "auto"
            ? undefined
            : openaiImageGenerationOptions.quality,
        //response_format: "b64_json", // Force b64_json
        size:
          openaiImageGenerationOptions.size === "auto"
            ? undefined
            : openaiImageGenerationOptions.size,
        user: openaiImageGenerationOptions.user,
        background:
          openaiImageGenerationOptions.background === "auto"
            ? undefined
            : openaiImageGenerationOptions.background,
        // style is removed as it's DALL-E 3 specific and caused TS error
      };

      // Remove undefined keys to avoid sending them
      Object.keys(apiOptions).forEach(
        (key) =>
          (apiOptions as any)[key] === undefined &&
          delete (apiOptions as any)[key]
      );

      console.log(
        `Snowgander: generateImage: apiOptions: ${JSON.stringify(apiOptions)}`
      );

      const result = await this.client.images.generate(apiOptions);

      console.log(
        `snowgander: generateImage: Result: ${JSON.stringify(result)}`
      );

      const images: ImageDataBlock[] = result.data
        ? result.data
            .filter((item) => item.b64_json)
            .map((item) => ({
              type: "image_data",
              mimeType: "image/png", // Assuming PNG, might need adjustment
              base64Data: item.b64_json!,
            }))
        : [];

      if (images.length === 0) {
        // Handle cases where API returns success but no images
        const errMsg = "OpenAI API returned success but no image data.";
        console.error("generateImage error:", errMsg);
        return createErrorResponse("Failed to retrieve image data.", errMsg);
      }

      // Use usage data directly from the API response
      const usage = this.mapApiUsageToUsageResponse(result.usage);

      return {
        role: "assistant",
        content: images,
        usage: usage,
      };
    } catch (error: any) {
      const errMsg = `Failed to generate image: ${error.message || error}`;
      console.error(`${this.vendor} generateImage error:`, error);
      return createErrorResponse("Image generation failed.", errMsg);
    }
  }

  // Note: editImage still needs implementation for image handling.
  // We'll make it return ErrorBlocks for missing options and the NotImplementedError.
  async editImage(options: AIRequestOptions): Promise<AIResponse> {
    const {
      prompt: textPrompt,
      messages,
      openaiImageEditOptions,
      model: modelNameFromOptions,
    } = options;

    if (!openaiImageEditOptions) {
      const errMsg =
        "openaiImageEditOptions are required for OpenAI image editing.";
      console.error("editImage error:", errMsg);
      return createErrorResponse("Image editing options missing.", errMsg);
    }

    // Placeholder check for prompt (similar to generateImage)
    let prompt = textPrompt;
    if (!prompt && messages && messages.length > 0) {
      // ... logic to extract prompt from messages ...
    }
    if (!prompt) {
      const errMsg = "A text prompt is required for image editing.";
      console.error("editImage error:", errMsg);
      return createErrorResponse("Prompt missing for editing.", errMsg);
    }

    if (
      !openaiImageEditOptions.image ||
      openaiImageEditOptions.image.length === 0
    ) {
      const errMsg = "An input image is required for editing.";
      console.error("editImage error:", errMsg);
      return createErrorResponse("Input image missing for editing.", errMsg);
    }

    const model = modelNameFromOptions || this.modelConfig.apiName;

    try {
      // Prepare image/mask inputs
      const imageFiles = await Promise.all(
        openaiImageEditOptions.image.map((img) => this.prepareImageInput(img))
      );
      const maskFile = openaiImageEditOptions.mask
        ? await this.prepareImageInput(openaiImageEditOptions.mask)
        : undefined;

      const apiOptions: OpenAI.Images.ImageEditParams = {
        model: model,
        prompt: prompt,
        image: imageFiles, // Pass the array of prepared files
        mask: maskFile,
        n: openaiImageEditOptions.n,
        //response_format: "b64_json", // Force b64_json
        size: (openaiImageEditOptions.size === "auto"
          ? undefined
          : openaiImageEditOptions.size) as any, // Cast to any to bypass SDK type mismatch for gpt-image-1 sizes
        user: openaiImageEditOptions.user,
        // moderation: openaiImageEditOptions.moderation, // Removed: Not valid in ImageEditParams type
      };

      // Remove undefined keys
      Object.keys(apiOptions).forEach(
        (key) =>
          (apiOptions as any)[key] === undefined &&
          delete (apiOptions as any)[key]
      );

      console.log(
        `Snowgander: editImage: apiOptions: ${JSON.stringify({
          ...apiOptions,
          image: "[Prepared Files]",
          mask: maskFile ? "[Prepared Mask]" : undefined,
        })}` // Avoid logging large file data
      );

      const result = await this.client.images.edit(apiOptions);

      console.log(`snowgander: editImage: Result: ${JSON.stringify(result)}`);

      const images: ImageDataBlock[] = result.data
        ? result.data
            .filter((item) => item.b64_json)
            .map((item) => ({
              type: "image_data",
              mimeType: "image/png", // Assuming PNG, might need adjustment based on API response or options
              base64Data: item.b64_json!,
            }))
        : [];

      if (images.length === 0) {
        const errMsg = "OpenAI API returned success but no edited image data.";
        console.error("editImage error:", errMsg);
        return createErrorResponse("Failed to retrieve edited image.", errMsg);
      }

      // Use usage data directly from the API response
      const usage = this.mapApiUsageToUsageResponse(result.usage);

      return {
        role: "assistant",
        content: images,
        usage: usage,
      };
    } catch (error: any) {
      const errMsg = `Failed to edit image: ${error.message || error}`;
      console.error(`${this.vendor} editImage error:`, error);
      return createErrorResponse("Image editing failed.", errMsg);
    }
  }

  // Helper to map OpenAI API usage object to our internal UsageResponse
  private mapApiUsageToUsageResponse(
    apiUsage: OpenAI.Images.ImagesResponse["usage"] | undefined
  ): UsageResponse {
    const inputCostPerMillion = this.modelConfig.inputTokenCost || 0;
    const outputCostPerMillion = this.modelConfig.outputTokenCost || 0;

    // Default to 0 if API doesn't provide usage data
    const inputTokens = apiUsage?.input_tokens ?? 0;
    const outputTokens = apiUsage?.output_tokens ?? 0;
    // Note: apiUsage.total_tokens might also be available, but we calculate totalCost from input/output

    const inputCost = (inputTokens / 1_000_000) * inputCostPerMillion;
    const outputCost = (outputTokens / 1_000_000) * outputCostPerMillion;
    const totalCost = inputCost + outputCost;

    console.log(
      `Mapped API Usage: InputTokens=${inputTokens}, OutputTokens=${outputTokens}, InputCost=${inputCost}, OutputCost=${outputCost}, TotalCost=${totalCost}`
    );

    return {
      inputCost: inputCost,
      outputCost: outputCost,
      totalCost: totalCost,
    };
  }

  // Helper to prepare image input for OpenAI SDK (handles URL and base64)
  private async prepareImageInput(
    imageBlock: ImageBlock | ImageDataBlock
  ): Promise<any> {
    // Reverted return type to any as Uploadable is not exported
    try {
      if (imageBlock.type === "image") {
        // Handle URL
        console.log(`Fetching image from URL: ${imageBlock.url}`);
        const response = await fetch(imageBlock.url);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch image URL: ${response.statusText} (${imageBlock.url})`
          );
        }
        // Use response.blob() which is compatible with toFile
        const blob = await response.blob();
        // Get the content type from the response header
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.startsWith("image/")) {
          console.warn(
            `Fetched image URL ${imageBlock.url} returned non-image content-type: ${contentType}. Attempting to proceed, but API may reject.`
          );
          // Consider throwing an error here or defaulting more carefully
        }
        // Determine filename from URL or use a default
        const filename = imageBlock.url.split("/").pop() || "image.png"; // Keep filename logic
        console.log(
          `Converting fetched blob to FileLike: ${filename} (Type: ${
            contentType || "unknown"
          })`
        );
        // Pass the inferred content type to toFile
        return await toFile(blob, filename, {
          type: contentType || undefined, // Pass contentType, let toFile handle if null/undefined
        });
      } else if (imageBlock.type === "image_data") {
        // Handle base64 data
        console.log(
          `Converting base64 data (mimeType: ${imageBlock.mimeType}) to FileLike`
        );
        const buffer = Buffer.from(imageBlock.base64Data, "base64");
        const filename = `image.${imageBlock.mimeType.split("/")[1] || "png"}`; // Create filename from mime type
        console.log(`Created buffer, converting to FileLike: ${filename}`);
        return await toFile(buffer, filename, { type: imageBlock.mimeType });
      } else {
        // Should not happen with correct typing, but handle defensively
        throw new Error(
          `Unsupported image block type: ${(imageBlock as any).type}`
        );
      }
    } catch (error: any) {
      console.error("Error preparing image input:", error);
      throw new Error(`Failed to prepare image input: ${error.message}`);
    }
  }
}
