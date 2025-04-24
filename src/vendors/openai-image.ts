import OpenAI from "openai";
import {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  Chat,
  ChatResponse,
  ContentBlock,
  ImageGenerationResponse,
  ImageEditResponse,
  ImageDataBlock,
  ImageBlock,
  MCPAvailableTool,
  Message, // Added for constructing AIRequestOptions in sendChat
  ModelConfig,
  UsageResponse,
  VendorConfig,
  NotImplementedError,
  ErrorBlock,
  OpenAIImageEditOptions,
} from "../types";

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
      if (openaiImageGenerationOptions) {
        const options: AIRequestOptions = {
          ...baseOptions,
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
        const errMsg = `${this.vendor} adapter requires either openaiImageGenerationOptions or openaiImageEditOptions to be provided in the Chat object via sendChat.`;
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
      const apiOptions: OpenAI.Images.ImageGenerateParams = {
        model: model,
        prompt: prompt,
        n: openaiImageGenerationOptions.n,
        quality:
          openaiImageGenerationOptions.quality === "auto"
            ? undefined
            : openaiImageGenerationOptions.quality,
        response_format: "b64_json", // Force b64_json for consistent handling
        size: openaiImageGenerationOptions.size,
        style: openaiImageGenerationOptions.style,
        user: openaiImageGenerationOptions.user,
        // background: openaiImageGenerationOptions.background, // GPT Image specific
        // output_compression: openaiImageGenerationOptions.output_compression, // GPT Image specific
      };

      // Remove undefined keys
      Object.keys(apiOptions).forEach(
        (key) =>
          apiOptions[key as keyof typeof apiOptions] === undefined &&
          delete apiOptions[key as keyof typeof apiOptions]
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

      const usage = this.calculateImageCost(
        openaiImageGenerationOptions.size,
        openaiImageGenerationOptions.quality
        // TODO: Add prompt token calculation if needed for cost
      );

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

    // Still not implemented part
    const errMsgNotImplemented = `${this.vendor} adapter editImage not fully implemented yet (image input handling needed).`;
    console.error("editImage error:", errMsgNotImplemented);
    return createErrorResponse(
      "Image editing is not fully supported yet.",
      errMsgNotImplemented
    );

    // --- Keep structure for future implementation ---
    /*
    try {
        // ... (API call logic) ...

        // ... (Process result) ...

        // return { role: 'assistant', content: images, usage: usage };

    } catch (error: any) {
        const errMsg = `Failed to edit image: ${error.message || error}`;
        console.error(`${this.vendor} editImage error:`, error);
        return createErrorResponse("Image editing failed.", errMsg);
    }
    const model = modelNameFromOptions || this.modelConfig.apiName;

    let prompt = textPrompt;
    // ... (extract prompt logic similar to generateImage) ...

    if (!prompt) {
      throw new Error('A text prompt is required for image editing.');
    }
    if (!openaiImageEditOptions.image || openaiImageEditOptions.image.length === 0) {
        throw new Error('An input image is required for editing.');
    }

    try {
        // Prepare image/mask inputs (requires async handling, fetching URLs, converting base64)
        const imageFile = await this.prepareImageInput(openaiImageEditOptions.image[0]); // Needs implementation
        const maskFile = openaiImageEditOptions.mask ? await this.prepareImageInput(openaiImageEditOptions.mask) : undefined; // Needs implementation

        const apiOptions: OpenAI.Images.ImageEditParams = {
            model: model,
            prompt: prompt,
            image: imageFile, // This needs to be a File-like object or Buffer
            mask: maskFile, // This needs to be a File-like object or Buffer
            n: openaiImageEditOptions.n,
            response_format: 'b64_json',
            size: openaiImageEditOptions.size,
            user: openaiImageEditOptions.user,
        };

        // Remove undefined keys
        Object.keys(apiOptions).forEach(key => apiOptions[key as keyof typeof apiOptions] === undefined && delete apiOptions[key as keyof typeof apiOptions]);

        const result = await this.client.images.edit(apiOptions);

        const images: ImageDataBlock[] = result.data
            ? result.data.filter(item => item.b64_json).map(item => ({
                type: 'image_data',
                mimeType: 'image/png',
                base64Data: item.b64_json!,
            }))
            : [];

        // Placeholder cost - adapt calculateImageCost if needed for edits
        const usage = this.calculateImageCost(
            openaiImageEditOptions.size,
            undefined // Quality not applicable to edits API
        );

        return {
            images: images,
            usage: usage,
        };

    } catch (error: any) {
        console.error(`${this.vendor} editImage error:`, error);
        throw new Error(`Failed to edit image: ${error.message || error}`);
    }
    */
  }

  // Placeholder Helper to calculate cost - Needs proper implementation using utils
  private calculateImageCost(
    size: OpenAI.Images.ImageGenerateParams["size"],
    quality: OpenAI.Images.ImageGenerateParams["quality"],
    promptTokens: number = 0 // Keep promptTokens for future use
  ): UsageResponse {
    // TODO: Implement actual cost calculation based on size, quality, prompt tokens,
    // and this.modelConfig.inputTokenCost / outputTokenCost.
    // This likely requires a new utility function in utils.ts.

    console.warn(
      "Placeholder cost calculation used for OpenAIImageAdapter. Implement proper cost logic."
    );
    // Return zero cost placeholder matching UsageResponse structure
    return {
      inputCost: 0,
      outputCost: 0,
      totalCost: 0,
    };
  }

  // --- Helper needed for editImage ---
  // private async prepareImageInput(imageBlock: ImageBlock | ImageDataBlock): Promise<Buffer | ???> {
  //   // If ImageBlock (URL), fetch the image, get buffer
  //   // If ImageDataBlock (base64), decode to buffer
  //   // Return Buffer or appropriate type for OpenAI SDK
  //   throw new NotImplementedError("prepareImageInput not implemented");
  // }
}
