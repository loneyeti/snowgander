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
  OpenAIImageEditOptions, // Needed for editImage
} from "../types";
// Removed computeResponseCost import as it's not used correctly here yet

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
    throw new NotImplementedError(
      `${this.vendor} adapter does not support generateResponse. Use generateImage or editImage.`
    );
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

    console.log(JSON.stringify(chat));
    console.log(JSON.stringify(messages));

    const baseOptions: Partial<AIRequestOptions> = {
      model: model || this.modelConfig.apiName,
      messages: messages,
      prompt: prompt, // Pass prompt along
      systemPrompt: systemPrompt, // Pass system prompt
      // Note: maxTokens, temperature etc. from Chat are not directly applicable here
      // but could be added if needed for future API versions or cost estimation.
    };

    console.log(JSON.stringify(baseOptions));

    if (openaiImageGenerationOptions && this.generateImage) {
      const options: AIRequestOptions = {
        ...baseOptions,
        openaiImageGenerationOptions: openaiImageGenerationOptions,
      } as AIRequestOptions; // Type assertion needed as baseOptions is partial

      const result = await this.generateImage(options);
      return {
        role: "assistant",
        content: result.images, // Map images to content blocks
        usage: result.usage,
      };
    } else if (openaiImageEditOptions && this.editImage) {
      const options: AIRequestOptions = {
        ...baseOptions,
        openaiImageEditOptions: openaiImageEditOptions,
      } as AIRequestOptions; // Type assertion needed as baseOptions is partial

      const result = await this.editImage(options);
      return {
        role: "assistant",
        content: result.images, // Map images to content blocks
        usage: result.usage,
      };
    } else {
      console.error(
        "adapter requires either openaiImageGenerationOptions or openaiImageEditOptions to be provided in the Chat object via sendChat."
      );
      throw new Error(
        `${this.vendor} adapter requires either openaiImageGenerationOptions or openaiImageEditOptions to be provided in the Chat object via sendChat.`
      );
    }
  }

  async sendMCPChat(
    chat: Chat,
    tools: MCPAvailableTool[],
    options?: AIRequestOptions | undefined
  ): Promise<ChatResponse> {
    throw new NotImplementedError(
      `${this.vendor} adapter does not support sendMCPChat.`
    );
  }

  async generateImage(
    options: AIRequestOptions
  ): Promise<ImageGenerationResponse> {
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
      console.error("A text prompt is required for image generation.");
      throw new Error("A text prompt is required for image generation.");
    }

    if (!openaiImageGenerationOptions) {
      console.error(
        "openaiImageGenerationOptions are required for OpenAI image generation."
      );
      throw new Error(
        "openaiImageGenerationOptions are required for OpenAI image generation."
      );
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

      console.log(
        `Snowgander: generateImage: apiOptions (pre-cleanup): ${JSON.stringify(
          apiOptions
        )}`
      );

      // Remove undefined keys
      Object.keys(apiOptions).forEach(
        (key) =>
          apiOptions[key as keyof typeof apiOptions] === undefined &&
          delete apiOptions[key as keyof typeof apiOptions]
      );

      console.log(
        `Snowgander: generateImage: apiOptions (post-cleanup): ${JSON.stringify(
          apiOptions
        )}`
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
              mimeType: "image/png",
              base64Data: item.b64_json!,
            }))
        : [];

      const usage = this.calculateImageCost(
        openaiImageGenerationOptions.size,
        openaiImageGenerationOptions.quality
        // TODO: Add prompt token calculation if needed for cost
      );

      return {
        images: images,
        usage: usage,
      };
    } catch (error: any) {
      console.error(`${this.vendor} generateImage error:`, error);
      throw new Error(`Failed to generate image: ${error.message || error}`);
    }
  }

  async editImage(options: AIRequestOptions): Promise<ImageEditResponse> {
    const {
      prompt: textPrompt,
      messages,
      openaiImageEditOptions,
      model: modelNameFromOptions,
    } = options;

    if (!openaiImageEditOptions) {
      throw new Error(
        "openaiImageEditOptions are required for OpenAI image editing."
      );
    }

    // TODO: Implement image fetching/conversion for ImageBlock/ImageDataBlock inputs
    // This requires handling URLs (fetching) and base64 data.
    // The OpenAI SDK expects file streams or Buffers.
    // Example placeholder:
    // const imageInput = await this.prepareImageInput(openaiImageEditOptions.image[0]); // Needs implementation
    // const maskInput = openaiImageEditOptions.mask ? await this.prepareImageInput(openaiImageEditOptions.mask) : undefined;

    console.log("editImage called with options:", options); // Placeholder
    throw new NotImplementedError(
      `${this.vendor} adapter editImage not fully implemented yet (image input handling needed).`
    );

    // --- Basic structure (needs image input handling) ---
    /*
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
