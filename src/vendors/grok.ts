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
  UsageResponse,
  ImageBlock,
  ImageDataBlock,
  GrokImageAspectRatio,
  OpenAIImageGenerationOptions,
} from "../types";
import { computeResponseCost, getImageDataFromUrl } from "../utils";

const DEFAULT_IMAGE_MODEL = "grok-imagine-image-2.0";
const MAX_EDIT_IMAGES = 5;
const USD_TICKS_PER_DOLLAR = 10_000_000_000; // 1 cent = 1e8 ticks

const GROK_ASPECT_RATIOS: Exclude<GrokImageAspectRatio, "auto">[] = [
  "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5",
  "19.5:9", "9:20", "20:9", "1:2", "2:1", "21:9", "5:2",
];

type GrokImageRef = { url: string };

// Response shape shared by /v1/images/generations and /v1/images/edits
interface GrokImageResponse {
  data?: {
    b64_json?: string | null;
    url?: string | null;
    mime_type?: string | null;
  }[];
  usage?: { cost_in_usd_ticks?: number } | null;
}

function toImageRef(block: ImageBlock | ImageDataBlock): GrokImageRef {
  return block.type === "image"
    ? { url: block.url }
    : { url: `data:${block.mimeType};base64,${block.base64Data}` };
}

// Explicit aspectRatio wins; otherwise map a "WxH" size to the nearest supported ratio.
function resolveAspectRatio(
  opts?: OpenAIImageGenerationOptions
): GrokImageAspectRatio | undefined {
  if (opts?.aspectRatio) return opts.aspectRatio;
  const match = opts?.size?.match(/^(\d+)x(\d+)$/);
  if (!match) return undefined;
  const target = Math.log(Number(match[1]) / Number(match[2]));
  let best = GROK_ASPECT_RATIOS[0];
  let bestDiff = Infinity;
  for (const ratio of GROK_ASPECT_RATIOS) {
    const [w, h] = ratio.split(":").map(Number);
    const diff = Math.abs(Math.log(w / h) - target);
    if (diff < bestDiff) {
      best = ratio;
      bestDiff = diff;
    }
  }
  return best;
}

// quality is only accepted by grok-imagine-image-2.0 (grok-imagine-image-quality / -pro
// redirect there as of Nov 2 2026). Grok has no tier above medium.
function resolveQuality(
  model: string,
  quality?: OpenAIImageGenerationOptions["quality"]
): "low" | "medium" | "auto" | undefined {
  if (!quality || !/^grok-imagine-image-(2|quality|pro)/.test(model)) {
    return undefined;
  }
  return quality === "low" || quality === "auto" ? quality : "medium";
}

function sniffImageMimeType(base64Data: string): string {
  if (base64Data.startsWith("iVBOR")) return "image/png";
  if (base64Data.startsWith("UklGR")) return "image/webp";
  return "image/jpeg";
}

function imageErrorResponse(
  publicMessage: string,
  privateMessage: string,
  code?: string
): AIResponse {
  return {
    role: "assistant",
    content: [{ type: "error", code, publicMessage, privateMessage }],
  };
}

// Interface for xAI specific delta which includes reasoning_content
interface GrokChatCompletionChunkDelta {
  content?: string | null;
  role?: "system" | "user" | "assistant" | "tool";
  reasoning_content?: string | null;
}

export class GrokAdapter implements AIVendorAdapter {
  private client: OpenAI;
  private modelConfig: ModelConfig;
  public isVisionCapable: boolean;
  public isImageGenerationCapable: boolean;
  public isThinkingCapable: boolean;
  public inputTokenCost?: number;
  public outputTokenCost?: number;

  constructor(config: VendorConfig, modelConfig: ModelConfig) {
    this.modelConfig = modelConfig;
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://api.x.ai/v1",
      dangerouslyAllowBrowser: true,
    });
    this.isVisionCapable = modelConfig.isVision;
    this.isImageGenerationCapable = modelConfig.isImageGeneration;
    this.isThinkingCapable = modelConfig.isThinking;

    if (modelConfig.inputTokenCost && modelConfig.outputTokenCost) {
      this.inputTokenCost = modelConfig.inputTokenCost;
      this.outputTokenCost = modelConfig.outputTokenCost;
    }
  }

  async generateResponse(options: AIRequestOptions): Promise<AIResponse> {
    const { model } = options;

    // Route to Image Generation if:
    // 1. Explicit options are provided
    // 2. The flag useImageGeneration is true
    // 3. The model name implies it is an image generation model (e.g., "grok-imagine-image-2.0")
    const isImageModel = model.toLowerCase().includes("-image");

    if (
      (options.openaiImageGenerationOptions ||
        options.useImageGeneration ||
        isImageModel) &&
      this.isImageGenerationCapable
    ) {
      return this.handleImageRequest(options);
    }

    // Otherwise, handle as Standard Chat Completion
    const { messages, systemPrompt } = options;
    const apiMessages = this.mapMessages(messages, systemPrompt);

    const response = await this.client.chat.completions.create({
      model: model,
      messages: apiMessages,
      stream: false,
    });

    const choice = response.choices[0];
    const contentBlocks: ContentBlock[] = [];

    // Extract Reasoning/Thinking
    const messageRaw = choice.message as any;
    if (messageRaw.reasoning_content) {
      contentBlocks.push({
        type: "thinking",
        thinking: messageRaw.reasoning_content,
        signature: "grok",
      });
    }

    // Extract Text
    if (choice.message.content) {
      contentBlocks.push({
        type: "text",
        text: choice.message.content,
      });
    }

    // Calculate Usage
    let usage: UsageResponse | undefined;
    if (response.usage && this.inputTokenCost && this.outputTokenCost) {
      const inputCost = computeResponseCost(
        response.usage.prompt_tokens,
        this.inputTokenCost
      );
      const outputCost = computeResponseCost(
        response.usage.completion_tokens,
        this.outputTokenCost
      );
      usage = {
        inputCost,
        outputCost,
        totalCost: inputCost + outputCost,
      };
    }

    return {
      role: "assistant",
      content: contentBlocks,
      usage,
    };
  }

  // --- Grok Imagine image generation / editing ---
  // xAI's /images/edits only accepts JSON (the OpenAI SDK's images.edit() sends
  // multipart), so both endpoints go through the SDK's generic client.post().
  private async handleImageRequest(
    options: AIRequestOptions
  ): Promise<AIResponse> {
    const genOptions = options.openaiImageGenerationOptions;
    const action = genOptions?.action ?? "auto";
    const model = options.model || DEFAULT_IMAGE_MODEL;

    const prompt = this.resolveImagePrompt(options);
    if (!prompt) {
      return imageErrorResponse(
        "A text prompt is required for image generation.",
        "No prompt provided for Grok image generation"
      );
    }

    const editImages =
      action === "generate" ? [] : this.collectEditImages(options);
    if (action === "edit" && editImages.length === 0) {
      return imageErrorResponse(
        "No source image to edit.",
        "Grok image edit requested but no source image was found in options or history"
      );
    }
    const isEdit = editImages.length > 0;

    const aspectRatio = resolveAspectRatio(genOptions);
    const body: Record<string, unknown> = {
      model,
      prompt,
      response_format: "b64_json",
    };
    if (genOptions?.n) body.n = genOptions.n;
    if (genOptions?.resolution) body.resolution = genOptions.resolution;
    const quality = resolveQuality(model, genOptions?.quality);
    if (quality) body.quality = quality;
    if (genOptions?.user) body.user = genOptions.user;

    if (isEdit) {
      if (editImages.length === 1) {
        // Single-image edits always take the input image's aspect ratio
        body.image = editImages[0];
      } else {
        body.images = editImages;
        if (aspectRatio) body.aspect_ratio = aspectRatio;
      }
    } else if (aspectRatio) {
      body.aspect_ratio = aspectRatio;
    }

    try {
      const response = await this.client.post<GrokImageResponse>(
        isEdit ? "/images/edits" : "/images/generations",
        { body }
      );
      return await this.mapImageResponse(response);
    } catch (error: any) {
      const message = error?.message || String(error);
      const isModeration =
        error?.code === "moderation_blocked" ||
        /moderat|content polic|safety/i.test(message);
      console.error(`Grok image ${isEdit ? "edit" : "generation"} error:`, error);
      return imageErrorResponse(
        isModeration
          ? "The request was blocked by content moderation."
          : `Failed to ${isEdit ? "edit" : "generate"} image.`,
        message,
        isModeration ? "moderation_blocked" : error?.code ?? undefined
      );
    }
  }

  private resolveImagePrompt(options: AIRequestOptions): string | undefined {
    if (options.prompt) return options.prompt;
    const lastUser = [...options.messages]
      .reverse()
      .find((m) => m.role === "user");
    const textBlock = lastUser?.content.find((c) => c.type === "text");
    return textBlock?.type === "text" ? textBlock.text : undefined;
  }

  // Source images for an edit, in the order they're referenced as <IMAGE_0>, <IMAGE_1>, ...
  // Explicit openaiImageEditOptions.image wins; otherwise the most recent assistant image
  // comes first (so it sets the output aspect ratio), followed by the latest user attachments.
  private collectEditImages(options: AIRequestOptions): GrokImageRef[] {
    const explicit = options.openaiImageEditOptions?.image;
    if (explicit && explicit.length > 0) {
      return explicit.slice(-MAX_EDIT_IMAGES).map(toImageRef);
    }

    const { messages } = options;
    let priorImage: GrokImageRef | undefined;
    let lastUserIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        lastUserIndex = i;
        break;
      }
    }

    for (let i = messages.length - 1; i >= 0 && !priorImage; i--) {
      if (messages[i].role !== "assistant") continue;
      const blocks = messages[i].content;
      for (let j = blocks.length - 1; j >= 0; j--) {
        const block = blocks[j];
        if (block.type === "image_data" && block.isPartial) continue;
        if (block.type === "image_data" || block.type === "image") {
          priorImage = toImageRef(block);
          break;
        }
      }
    }

    const userImages: GrokImageRef[] =
      lastUserIndex >= 0
        ? messages[lastUserIndex].content
            .filter(
              (b): b is ImageBlock | ImageDataBlock =>
                b.type === "image" || b.type === "image_data"
            )
            .map(toImageRef)
        : [];
    if (options.visionUrl) userImages.push({ url: options.visionUrl });

    if (!priorImage) return userImages.slice(-MAX_EDIT_IMAGES);
    return [priorImage, ...userImages.slice(-(MAX_EDIT_IMAGES - 1))];
  }

  private async mapImageResponse(
    response: GrokImageResponse
  ): Promise<AIResponse> {
    const contentBlocks: ContentBlock[] = [];
    const idBase = `grok_img_${Date.now()}`;

    for (const [i, img] of (response.data ?? []).entries()) {
      let base64Data = img.b64_json ?? undefined;
      let mimeType = img.mime_type ?? undefined;
      if (!base64Data && img.url) {
        const fetched = await getImageDataFromUrl(img.url);
        if (!fetched) continue;
        base64Data = fetched.base64Data;
        mimeType = mimeType ?? fetched.mimeType;
      }
      if (!base64Data) continue;
      contentBlocks.push({
        type: "image_data",
        id: `${idBase}_${i}`,
        mimeType: mimeType ?? sniffImageMimeType(base64Data),
        base64Data,
        isPartial: false,
      });
    }

    if (contentBlocks.length === 0) {
      return imageErrorResponse(
        "Failed to retrieve generated image.",
        "Grok image API returned no image data"
      );
    }

    const ticks = response.usage?.cost_in_usd_ticks;
    const cost = ticks ? ticks / USD_TICKS_PER_DOLLAR : 0;
    return {
      role: "assistant",
      content: contentBlocks,
      usage: {
        inputCost: 0,
        outputCost: cost,
        totalCost: cost,
        didGenerateImage: true,
      },
    };
  }

  async sendChat(chat: Chat): Promise<ChatResponse> {
    const response = await this.generateResponse({
      model: chat.model,
      messages: chat.responseHistory.map((r) => ({
        role: r.role,
        content: r.content,
      })),
      systemPrompt: chat.systemPrompt,
      openaiImageGenerationOptions: chat.openaiImageGenerationOptions,
      openaiImageEditOptions: chat.openaiImageEditOptions,
      visionUrl: chat.visionUrl ?? undefined,
      prompt: chat.prompt,
    });

    return {
      role: response.role,
      content: response.content,
      usage: response.usage,
    };
  }

  async *streamResponse(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown> {
    const { model, messages, systemPrompt } = options;

    // Check if this is actually an image request coming through the stream path
    const isImageModel = model.toLowerCase().includes("-image");
    if (
      (options.openaiImageGenerationOptions ||
        options.useImageGeneration ||
        isImageModel) &&
      this.isImageGenerationCapable
    ) {
      // The Imagine endpoints don't stream, so await the result and yield it.
      const response = await this.handleImageRequest(options);
      for (const block of response.content) {
        yield block;
      }
      if (response.usage) {
        yield {
          type: "meta",
          responseId: `grok-img-${Date.now()}`,
          usage: response.usage,
        };
      }
      return;
    }

    const apiMessages = this.mapMessages(messages, systemPrompt);

    const stream = await this.client.chat.completions.create({
      model: model,
      messages: apiMessages,
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta as GrokChatCompletionChunkDelta;

      if (!delta) continue;

      // Handle Reasoning (Thinking)
      if (delta.reasoning_content) {
        yield {
          type: "thinking",
          thinking: delta.reasoning_content,
          signature: "grok",
        };
      }

      // Handle Content
      if (delta.content) {
        yield {
          type: "text",
          text: delta.content,
        };
      }
    }
  }

  private mapMessages(messages: Message[], systemPrompt?: string): any[] {
    const apiMessages: any[] = [];

    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }

    for (const msg of messages) {
      if (msg.role === "system") continue;

      const contentParts: any[] = [];

      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === "text") {
            contentParts.push({ type: "text", text: block.text });
          } else if (
            (block.type === "image" || block.type === "image_data") &&
            this.isVisionCapable
          ) {
            const imageUrl =
              block.type === "image"
                ? block.url
                : `data:${block.mimeType};base64,${block.base64Data}`;
            contentParts.push({
              type: "image_url",
              image_url: { url: imageUrl },
            });
          }
        }
      }

      if (contentParts.length > 0) {
        apiMessages.push({
          role: msg.role,
          content: contentParts,
        });
      }
    }
    return apiMessages;
  }
}
