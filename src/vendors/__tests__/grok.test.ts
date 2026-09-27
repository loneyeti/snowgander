import { GrokAdapter } from "../grok";
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  Message,
  ContentBlock,
  Chat,
} from "../../types";

const mockPost = jest.fn();
const mockChatCreate = jest.fn();
const mockClient = {
  post: mockPost,
  chat: { completions: { create: mockChatCreate } },
};

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => mockClient);
});

const PNG_B64 = "iVBORw0KGgoAAAANSUhEUg";
const JPEG_B64 = "/9j/4AAQSkZJRgABAQ";

const imageModel: ModelConfig = {
  apiName: "grok-imagine-image-2.0",
  isVision: true,
  isImageGeneration: true,
  isThinking: false,
};

const chatModel: ModelConfig = {
  apiName: "grok-4.7",
  isVision: true,
  isImageGeneration: false,
  isThinking: true,
  inputTokenCost: 3,
  outputTokenCost: 15,
};

const config: VendorConfig = { apiKey: "test-xai-key" };

const userText = (text: string): Message => ({
  role: "user",
  content: [{ type: "text", text }],
});

const imageResponse = (overrides: Record<string, unknown> = {}) => ({
  data: [{ b64_json: PNG_B64, mime_type: "image/png" }],
  usage: { cost_in_usd_ticks: 200_000_000 }, // 2 cents
  ...overrides,
});

const lastBody = () => mockPost.mock.calls[mockPost.mock.calls.length - 1][1].body;
const lastPath = () => mockPost.mock.calls[mockPost.mock.calls.length - 1][0];

describe("GrokAdapter image generation", () => {
  let adapter: GrokAdapter;

  beforeEach(() => {
    jest.clearAllMocks();
    mockPost.mockResolvedValue(imageResponse());
    adapter = new GrokAdapter(config, imageModel);
  });

  it("generates via /images/generations with Imagine params", async () => {
    const options: AIRequestOptions = {
      model: "grok-imagine-image-2.0",
      messages: [userText("a lighthouse at dawn")],
      openaiImageGenerationOptions: {
        n: 2,
        aspectRatio: "16:9",
        resolution: "2k",
        quality: "low",
      },
    };

    const response = await adapter.generateResponse(options);

    expect(lastPath()).toBe("/images/generations");
    expect(lastBody()).toEqual({
      model: "grok-imagine-image-2.0",
      prompt: "a lighthouse at dawn",
      response_format: "b64_json",
      n: 2,
      aspect_ratio: "16:9",
      resolution: "2k",
      quality: "low",
    });
    expect(response.content[0]).toMatchObject({
      type: "image_data",
      mimeType: "image/png",
      base64Data: PNG_B64,
      isPartial: false,
    });
    expect((response.content[0] as any).id).toMatch(/^grok_img_\d+_0$/);
    expect(response.usage).toEqual({
      inputCost: 0,
      outputCost: 0.02,
      totalCost: 0.02,
      didGenerateImage: true,
    });
  });

  it("prefers the explicit prompt over message text", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("ignored")],
      prompt: "a red fox",
    });
    expect(lastBody().prompt).toBe("a red fox");
  });

  it.each([
    ["1024x1024", "1:1"],
    ["1536x1024", "3:2"],
    ["1024x1536", "2:3"],
    ["1920x1080", "16:9"],
  ])("maps size %s to aspect ratio %s", async (size, ratio) => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
      openaiImageGenerationOptions: { size: size as any },
    });
    expect(lastBody().aspect_ratio).toBe(ratio);
  });

  it("omits aspect_ratio for size auto", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
      openaiImageGenerationOptions: { size: "auto" },
    });
    expect(lastBody()).not.toHaveProperty("aspect_ratio");
  });

  it("maps high+ quality to medium and omits quality on grok-imagine-image 1.0", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
      openaiImageGenerationOptions: { quality: "high" },
    });
    expect(lastBody().quality).toBe("medium");

    await adapter.generateResponse({
      model: "grok-imagine-image",
      messages: [userText("x")],
      openaiImageGenerationOptions: { quality: "low" },
    });
    expect(lastBody()).not.toHaveProperty("quality");
  });

  it("sniffs the mime type when mime_type is missing", async () => {
    mockPost.mockResolvedValue(
      imageResponse({ data: [{ b64_json: JPEG_B64 }] })
    );
    const response = await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
    });
    expect((response.content[0] as any).mimeType).toBe("image/jpeg");
  });

  it("returns an ErrorBlock when no image data comes back", async () => {
    mockPost.mockResolvedValue({ data: [] });
    const response = await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
    });
    expect(response.content[0]).toMatchObject({ type: "error" });
  });

  it("maps moderation failures to moderation_blocked", async () => {
    mockPost.mockRejectedValue(
      new Error("400 Generated image rejected by content moderation.")
    );
    const response = await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("x")],
    });
    expect(response.content[0]).toMatchObject({
      type: "error",
      code: "moderation_blocked",
      publicMessage: "The request was blocked by content moderation.",
    });
  });
});

describe("GrokAdapter image editing", () => {
  let adapter: GrokAdapter;

  const priorImage: ContentBlock = {
    type: "image_data",
    id: "grok_img_1_0",
    mimeType: "image/png",
    base64Data: PNG_B64,
    isPartial: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockPost.mockResolvedValue(imageResponse());
    adapter = new GrokAdapter(config, imageModel);
  });

  it("edits an explicit source image with a single `image` and no aspect_ratio", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("make it a pencil sketch")],
      openaiImageGenerationOptions: { aspectRatio: "1:1" },
      openaiImageEditOptions: {
        image: [{ type: "image", url: "https://example.com/cat.png" }],
      },
    });

    expect(lastPath()).toBe("/images/edits");
    expect(lastBody().image).toEqual({ url: "https://example.com/cat.png" });
    expect(lastBody()).not.toHaveProperty("images");
    expect(lastBody()).not.toHaveProperty("aspect_ratio");
  });

  it("edits a user-attached image as a data URI", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "add a hat" },
            { type: "image_data", mimeType: "image/jpeg", base64Data: JPEG_B64 },
          ],
        },
      ],
    });

    expect(lastPath()).toBe("/images/edits");
    expect(lastBody().image).toEqual({
      url: `data:image/jpeg;base64,${JPEG_B64}`,
    });
  });

  it("multi-turn: edits the previous assistant image", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        userText("a living room"),
        { role: "assistant", content: [priorImage] },
        userText("make it night time"),
      ],
    });

    expect(lastPath()).toBe("/images/edits");
    expect(lastBody().prompt).toBe("make it night time");
    expect(lastBody().image).toEqual({
      url: `data:image/png;base64,${PNG_B64}`,
    });
  });

  it("multi-turn: ignores partial previews in history", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        userText("a living room"),
        {
          role: "assistant",
          content: [
            priorImage,
            {
              type: "image_data",
              id: "ig_x",
              mimeType: "image/png",
              base64Data: "PARTIAL",
              isPartial: true,
            },
          ],
        },
        userText("brighter"),
      ],
    });
    expect(lastBody().image.url).toContain(PNG_B64);
  });

  it("multi-turn: prior image first, then user attachments, with aspect_ratio", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        userText("a living room"),
        { role: "assistant", content: [priorImage] },
        {
          role: "user",
          content: [
            { type: "text", text: "put <IMAGE_1> on the sofa" },
            { type: "image", url: "https://example.com/cat.png" },
          ],
        },
      ],
      openaiImageGenerationOptions: { aspectRatio: "16:9" },
    });

    expect(lastBody().images).toEqual([
      { url: `data:image/png;base64,${PNG_B64}` },
      { url: "https://example.com/cat.png" },
    ]);
    expect(lastBody()).not.toHaveProperty("image");
    expect(lastBody().aspect_ratio).toBe("16:9");
  });

  it("caps source images at 5, keeping the prior image", async () => {
    const attachments: ContentBlock[] = Array.from({ length: 6 }, (_, i) => ({
      type: "image",
      url: `https://example.com/${i}.png`,
    }));
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        userText("start"),
        { role: "assistant", content: [priorImage] },
        { role: "user", content: [{ type: "text", text: "combine" }, ...attachments] },
      ],
    });

    const images = lastBody().images;
    expect(images).toHaveLength(5);
    expect(images[0].url).toContain(PNG_B64);
    expect(images[4].url).toBe("https://example.com/5.png");
  });

  it("action generate skips editing even with an image in history", async () => {
    await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [
        userText("a living room"),
        { role: "assistant", content: [priorImage] },
        userText("something completely different"),
      ],
      openaiImageGenerationOptions: { action: "generate" },
    });
    expect(lastPath()).toBe("/images/generations");
  });

  it("action edit without any source image returns an ErrorBlock", async () => {
    const response = await adapter.generateResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("edit it")],
      openaiImageGenerationOptions: { action: "edit" },
    });
    expect(mockPost).not.toHaveBeenCalled();
    expect(response.content[0]).toMatchObject({ type: "error" });
  });

  it("sendChat forwards visionUrl and edit options", async () => {
    const chat: Chat = {
      responseHistory: [{ role: "user", content: [{ type: "text", text: "add a hat" }] }],
      visionUrl: "https://example.com/dog.png",
      model: "grok-imagine-image-2.0",
      prompt: "add a hat",
      imageURL: null,
      maxTokens: null,
      budgetTokens: null,
    };
    await adapter.sendChat(chat);
    expect(lastPath()).toBe("/images/edits");
    expect(lastBody().image).toEqual({ url: "https://example.com/dog.png" });
  });

  it("streamResponse yields the image then a meta block", async () => {
    const blocks: ContentBlock[] = [];
    for await (const block of adapter.streamResponse({
      model: "grok-imagine-image-2.0",
      messages: [userText("a lighthouse")],
    })) {
      blocks.push(block);
    }
    expect(blocks.map((b) => b.type)).toEqual(["image_data", "meta"]);
    expect((blocks[1] as any).usage.totalCost).toBe(0.02);
  });
});

describe("GrokAdapter chat", () => {
  beforeEach(() => jest.clearAllMocks());

  it("still uses chat completions for non-image models", async () => {
    mockChatCreate.mockResolvedValue({
      choices: [{ message: { content: "hello", reasoning_content: "hmm" } }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    const adapter = new GrokAdapter(config, chatModel);
    const response = await adapter.generateResponse({
      model: "grok-4.7",
      messages: [userText("hi")],
    });
    expect(mockPost).not.toHaveBeenCalled();
    expect(response.content).toEqual([
      { type: "thinking", thinking: "hmm", signature: "grok" },
      { type: "text", text: "hello" },
    ]);
  });
});
