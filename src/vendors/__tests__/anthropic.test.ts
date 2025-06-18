import { AnthropicAdapter } from "../anthropic";
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  Message,
  TextBlock,
  ImageDataBlock,
  AIResponse,
  UsageResponse,
  Chat,
  ImageBlock,
  ThinkingBlock,
  ChatResponse,
  ContentBlock,
} from "../../types";
import Anthropic from "@anthropic-ai/sdk";
import { computeResponseCost } from "../../utils";

// Mock the Anthropic SDK
const mockMessagesCreate = jest.fn();
jest.mock("@anthropic-ai/sdk", () => {
  return jest.fn().mockImplementation(() => ({
    messages: {
      create: mockMessagesCreate,
    },
  }));
});

// Helper to get mock constructor and methods
const getMockAnthropicClient = () => {
  const MockAnthropicConstructor = Anthropic as unknown as jest.Mock;
  return {
    mockMessagesCreate,
    MockAnthropicConstructor,
  };
};

describe("AnthropicAdapter", () => {
  let adapter: AnthropicAdapter;
  const mockConfig: VendorConfig = { apiKey: "test-anthropic-key" };
  const mockClaude3Opus: ModelConfig = {
    apiName: "claude-3-opus-20240229",
    isVision: true,
    isImageGeneration: false,
    isThinking: true,
    inputTokenCost: 15 / 1_000_000,
    outputTokenCost: 75 / 1_000_000,
  };
  const mockClaude3Sonnet: ModelConfig = {
    apiName: "claude-3-sonnet-20240229",
    isVision: true,
    isImageGeneration: false,
    isThinking: true,
    inputTokenCost: 3 / 1_000_000,
    outputTokenCost: 15 / 1_000_000,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new AnthropicAdapter(mockConfig, mockClaude3Opus);
    const { MockAnthropicConstructor } = getMockAnthropicClient();
    expect(MockAnthropicConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
      baseURL: mockConfig.baseURL,
    });
  });

  it("should initialize Anthropic client with correct config", () => {
    const { MockAnthropicConstructor } = getMockAnthropicClient();
    expect(MockAnthropicConstructor).toHaveBeenCalledTimes(1);
    expect(MockAnthropicConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
      baseURL: undefined,
    });
  });

  it("should reflect capabilities from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(true);
    expect(adapter.isImageGenerationCapable).toBe(false);
    expect(adapter.isThinkingCapable).toBe(true);
    expect(adapter.inputTokenCost).toBe(mockClaude3Opus.inputTokenCost);
    expect(adapter.outputTokenCost).toBe(mockClaude3Opus.outputTokenCost);
  });

  describe("generateResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello Claude!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockClaude3Opus.apiName,
      messages: basicMessages,
    };

    it("should call Anthropic messages.create with correct basic parameters", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const mockApiResponse = {
        id: "msg_123",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hi there!" }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(basicOptions);

      expect(mockMessagesCreate).toHaveBeenCalledTimes(1);
      expect(mockMessagesCreate).toHaveBeenCalledWith({
        model: mockClaude3Opus.apiName,
        messages: [
          { role: "user", content: [{ type: "text", text: "Hello Claude!" }] },
        ],
        system: undefined,
        max_tokens: 1024,
        temperature: undefined,
      });
    });

    it("should map Anthropic response to AIResponse format including calculated usage", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const mockApiResponse = {
        id: "msg_abc",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Response from Anthropic." }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 25, output_tokens: 15 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const response = await adapter.generateResponse(basicOptions);

      const expectedInputCost = computeResponseCost(
        mockApiResponse.usage.input_tokens,
        adapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        mockApiResponse.usage.output_tokens,
        adapter.outputTokenCost!
      );
      const expectedTotalCost = expectedInputCost + expectedOutputCost;

      expect(response).toEqual<AIResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Response from Anthropic." }],
        usage: {
          inputCost: expectedInputCost,
          outputCost: expectedOutputCost,
          totalCost: expectedTotalCost,
        },
      });
    });

    it("should handle system prompt", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const systemPrompt = "You are Claude.";
      const optionsWithSystem: AIRequestOptions = {
        ...basicOptions,
        systemPrompt,
      };
      const mockApiResponse = {
        id: "msg_sys",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Yes?" }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 1 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithSystem);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          system: systemPrompt,
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Hello Claude!" }],
            },
          ],
        })
      );
    });

    it("should warn and IGNORE ImageDataBlock (as mapping is not implemented/verified)", async () => {
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();
      const { mockMessagesCreate } = getMockAnthropicClient();
      const imageData: ImageDataBlock = {
        type: "image_data",
        mimeType: "image/jpeg",
        base64Data: "base64jpegstring",
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [imageData, { type: "text", text: "What is in this JPEG?" }],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
      };
      const mockApiResponse = {
        id: "msg_img",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "A JPEG image." }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 50, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImage);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "What is in this JPEG?" }],
            },
          ],
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Anthropic adapter received ImageDataBlock (base64)"
        )
      );
      consoleWarnSpy.mockRestore();
    });

    it("should map ImageBlock (URL) correctly for vision models", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const imageUrl = "http://example.com/image.png";
      const imageBlock: ImageBlock = {
        type: "image",
        url: imageUrl,
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [
            imageBlock,
            { type: "text", text: "What is in this image URL?" },
          ],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
      };
      const mockApiResponse = {
        id: "msg_img_url",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "An image URL." }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 50, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImage);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: { type: "url", url: imageUrl },
                },
                { type: "text", text: "What is in this image URL?" },
              ],
            },
          ],
        })
      );
    });

    it("should map ThinkingBlock input correctly", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const thinkingBlock: ThinkingBlock = {
        type: "thinking",
        thinking: "Let me think...",
        signature: "test-sig",
      };
      const messagesWithThinking: Message[] = [
        { role: "user", content: [thinkingBlock] },
      ];
      const optionsWithThinking: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithThinking,
      };
      const mockApiResponse = {
        id: "msg_think_in",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Okay, I thought." }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 50, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithThinking);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "thinking",
                  thinking: "Let me think...",
                  signature: "test-sig",
                },
              ],
            },
          ],
        })
      );
    });

    it("should map ThinkingBlock output correctly", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const mockApiResponse = {
        id: "msg_think_out",
        type: "message",
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: "Assistant is thinking...",
            signature: "anthropic",
          },
          { type: "text", text: "Here is the result." },
        ],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 25, output_tokens: 15 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const response = await adapter.generateResponse(basicOptions);

      expect(response.content).toEqual([
        {
          type: "thinking",
          thinking: "Assistant is thinking...",
          signature: "anthropic",
        },
        { type: "text", text: "Here is the result." },
      ]);
    });
  });

  describe("streamResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello stream!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockClaude3Opus.apiName,
      messages: basicMessages,
    };

    it("should stream text deltas as TextBlocks", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      async function* mockStream() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello" },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: " " },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "World!" },
        };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(basicOptions);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          stream: true,
        })
      );

      expect(chunks).toEqual([
        { type: "text", text: "Hello" },
        { type: "text", text: " " },
        { type: "text", text: "World!" },
      ]);
    });

    it("should stream thinking deltas as ThinkingBlocks", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const optionsWithThinking: AIRequestOptions = {
        ...basicOptions,
        thinkingMode: true,
        budgetTokens: 100,
      };

      async function* mockStream() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Let me think... " },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Okay, I have a plan." },
        };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(optionsWithThinking);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        {
          type: "thinking",
          thinking: "Let me think... ",
          signature: "anthropic",
        },
        {
          type: "thinking",
          thinking: "Okay, I have a plan.",
          signature: "anthropic",
        },
      ]);
    });

    it("should handle a mixed stream of thinking and text", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const optionsWithThinking: AIRequestOptions = {
        ...basicOptions,
        thinkingMode: true,
      };

      async function* mockStream() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Analyzing... " },
        };
        yield {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "Here is " },
        };
        yield {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "the answer." },
        };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(optionsWithThinking);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        {
          type: "thinking",
          thinking: "Analyzing... ",
          signature: "anthropic",
        },
        { type: "text", text: "Here is " },
        { type: "text", text: "the answer." },
      ]);
    });

    it("should ignore tool use deltas and log a warning", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      async function* mockStream() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"tool":' },
        };
        yield {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "Some text" },
        };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(basicOptions);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([{ type: "text", text: "Some text" }]);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining("tool_use delta (input_json_delta)")
      );

      consoleWarnSpy.mockRestore();
    });
  });

  describe("sendChat", () => {
    let generateResponseSpy: jest.SpyInstance;
    const baseChat: Partial<Chat> = {
      model: mockClaude3Opus.apiName,
      responseHistory: [],
      visionUrl: null,
      prompt: "User prompt",
      imageURL: null,
      maxTokens: 100,
      budgetTokens: null,
      systemPrompt: "System prompt",
    };

    beforeEach(() => {
      generateResponseSpy = jest
        .spyOn(adapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "Mocked chat response" }],
          usage: { inputCost: 0.015, outputCost: 0.075, totalCost: 0.09 },
        });
    });

    afterEach(() => {
      generateResponseSpy.mockRestore();
    });

    it("should call generateResponse with mapped messages and options (thinking disabled)", async () => {
      const chatWithHistory: Chat = {
        ...baseChat,
        responseHistory: [
          {
            role: "user",
            content: [{ type: "text", text: "Previous question" }],
          },
          {
            role: "assistant",
            content: [{ type: "text", text: "Previous answer" }],
          },
        ],
        prompt: "Current question",
      } as Chat;

      await adapter.sendChat(chatWithHistory);

      expect(generateResponseSpy).toHaveBeenCalledTimes(1);
      expect(generateResponseSpy).toHaveBeenCalledWith({
        model: chatWithHistory.model,
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "Previous question" }],
          },
          {
            role: "assistant",
            content: [{ type: "text", text: "Previous answer" }],
          },
          {
            role: "user",
            content: [{ type: "text", text: "Current question" }],
          },
        ],
        maxTokens: chatWithHistory.maxTokens,
        budgetTokens: undefined,
        systemPrompt: chatWithHistory.systemPrompt,
        thinkingMode: false,
        tools: undefined,
      });
    });

    it("should enable thinkingMode and pass budgetTokens if budgetTokens > 0", async () => {
      const chatWithThinking: Chat = {
        ...baseChat,
        budgetTokens: 500,
        prompt: "Think about this",
      } as Chat;

      await adapter.sendChat(chatWithThinking);

      expect(generateResponseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          thinkingMode: true,
          budgetTokens: 500,
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Think about this" }],
            },
          ],
        })
      );
    });

    it("should return the ChatResponse with content and usage from generateResponse", async () => {
      const chat: Chat = baseChat as Chat;
      const expectedUsage: UsageResponse = {
        inputCost: 0.015,
        outputCost: 0.075,
        totalCost: 0.09,
      };
      generateResponseSpy.mockResolvedValue({
        role: "assistant",
        content: [{ type: "text", text: "Specific chat response" }],
        usage: expectedUsage,
      });

      const result = await adapter.sendChat(chat);

      expect(result).toEqual<ChatResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Specific chat response" }],
        usage: expectedUsage,
      });
    });

    it("should include visionUrl as ImageBlock in messages passed to generateResponse", async () => {
      const chatWithVision: Chat = {
        ...baseChat,
        visionUrl: "http://example.com/vision.jpg",
        prompt: "Describe this vision image",
      } as Chat;
      adapter.isVisionCapable = true;
      await adapter.sendChat(chatWithVision);
      expect(generateResponseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: [
                { type: "image", url: "http://example.com/vision.jpg" },
                { type: "text", text: "Describe this vision image" },
              ],
            },
          ],
        })
      );
    });

    it("should ignore visionUrl if model is not vision capable", async () => {
      const nonVisionAdapter = new AnthropicAdapter(mockConfig, {
        ...mockClaude3Sonnet,
        isVision: false,
      });
      const nonVisionGenerateSpy = jest
        .spyOn(nonVisionAdapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "Mocked non-vision response" }],
          usage: { inputCost: 0.01, outputCost: 0.02, totalCost: 0.03 },
        });
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      const chatWithVision: Chat = {
        ...baseChat,
        visionUrl: "http://example.com/vision.jpg",
        prompt: "Describe this vision image",
      } as Chat;

      await nonVisionAdapter.sendChat(chatWithVision);

      expect(nonVisionGenerateSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Describe this vision image" }],
            },
          ],
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Vision URL provided for non-vision capable Anthropic model"
        )
      );

      nonVisionGenerateSpy.mockRestore();
      consoleWarnSpy.mockRestore();
    });
  });

  describe("generateImage", () => {
    it("should throw NotImplementedError", async () => {
      const dummyOptions: AIRequestOptions = {
        model: "claude-3-opus-20240229",
        messages: [
          { role: "user", content: [{ type: "text", text: "generate image" }] },
        ],
      };
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        "Image generation not supported by Anthropic"
      );
    });
  });
});
