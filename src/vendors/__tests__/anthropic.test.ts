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
            signature: "",
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
          signature: "",
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

    it("should handle a full SSE stream for a simple text response", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      // Mock the full, realistic event stream from Anthropic
      async function* mockStream() {
        yield {
          type: "message_start",
          message: { id: "msg_123", usage: { input_tokens: 10 } },
        };
        yield {
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "" },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hello" },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: " World!" },
        };
        yield { type: "content_block_stop", index: 0 };
        yield {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 5 },
        };
        yield { type: "message_stop" };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(basicOptions);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({ stream: true })
      );

      // Verify the output chunks are correct, with the meta block now at the end
      const expectedInputCost = computeResponseCost(
        10,
        adapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        5,
        adapter.outputTokenCost!
      );

      expect(chunks).toEqual([
        { type: "text", text: "Hello" },
        { type: "text", text: " World!" },
        {
          type: "meta",
          responseId: "msg_123",
          usage: {
            inputCost: expectedInputCost,
            outputCost: expectedOutputCost,
            totalCost: expectedInputCost + expectedOutputCost,
          },
        },
      ]);
    });

    it("should handle a mixed stream of thinking and text blocks", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const optionsWithThinking: AIRequestOptions = {
        ...basicOptions,
        thinkingMode: true,
      };

      // Mock a stream with a thinking block followed by a text block
      async function* mockStream() {
        yield {
          type: "message_start",
          message: { id: "msg_456", usage: { input_tokens: 15 } },
        };
        // Thinking block
        yield {
          type: "content_block_start",
          index: 0,
          content_block: { type: "thinking", thinking: "" },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Analyzing... " },
        };
        yield { type: "content_block_stop", index: 0 };
        // Text block
        yield {
          type: "content_block_start",
          index: 1,
          content_block: { type: "text", text: "" },
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
        yield { type: "content_block_stop", index: 1 };
        yield {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 10 },
        };
        yield { type: "message_stop" };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(optionsWithThinking);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const expectedInputCost = computeResponseCost(
        15,
        adapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        10,
        adapter.outputTokenCost!
      );

      expect(chunks).toEqual([
        {
          type: "thinking",
          thinking: "Analyzing... ",
          signature: "",
        },
        { type: "text", text: "Here is " },
        { type: "text", text: "the answer." },
        {
          type: "meta",
          responseId: "msg_456",
          usage: {
            inputCost: expectedInputCost,
            outputCost: expectedOutputCost,
            totalCost: expectedInputCost + expectedOutputCost,
          },
        },
      ]);
    });

    it("should yield an error block if the stream throws an error", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation();
      const testError = new Error("Stream connection failed");

      mockMessagesCreate.mockRejectedValue(testError);

      const stream = adapter.streamResponse(basicOptions);
      const chunks: ContentBlock[] = [];

      // We expect the function to throw, so we wrap the iteration in a try/catch
      try {
        for await (const chunk of stream) {
          chunks.push(chunk);
        }
      } catch (e) {
        expect(e).toBe(testError);
      }

      // The adapter should yield a structured error block before re-throwing
      expect(chunks).toEqual([
        {
          type: "error",
          publicMessage: "An error occurred while streaming from the provider.",
          privateMessage: "Stream connection failed",
        },
      ]);

      // And it should log the error server-side
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        "Error during Anthropic stream:",
        testError
      );
      consoleErrorSpy.mockRestore();
    });

    it("should accumulate and yield a complete tool_use block on content_block_stop", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      // Mock a stream with a tool_use block
      async function* mockStream() {
        yield {
          type: "message_start",
          message: { id: "msg_789", usage: { input_tokens: 20 } },
        };
        yield {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "tool_use",
            id: "tool_abc",
            name: "get_weather",
            input: {},
          },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"location":' },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "input_json_delta",
            partial_json: ' "San Francisco"}',
          },
        };
        yield { type: "content_block_stop", index: 0 };
        yield {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 15 },
        };
        yield { type: "message_stop" };
      }

      mockMessagesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(basicOptions);
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const expectedInputCost = computeResponseCost(
        20,
        adapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        15,
        adapter.outputTokenCost!
      );

      // The final output should contain the tool_use block followed by meta with usage
      expect(chunks).toEqual([
        {
          type: "tool_use",
          id: "tool_abc",
          name: "get_weather",
          input: '{"location": "San Francisco"}',
        },
        {
          type: "meta",
          responseId: "msg_789",
          usage: {
            inputCost: expectedInputCost,
            outputCost: expectedOutputCost,
            totalCost: expectedInputCost + expectedOutputCost,
          },
        },
      ]);

      // Ensure no warnings were logged for tool use, as it's now handled
      expect(consoleWarnSpy).not.toHaveBeenCalled();
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

  describe("Claude 4.6 support", () => {
    const mockClaude46: ModelConfig = {
      apiName: "claude-opus-4-6",
      isVision: true,
      isImageGeneration: false,
      isThinking: true,
      inputTokenCost: 15 / 1_000_000,
      outputTokenCost: 75 / 1_000_000,
    };

    const mockClaudeSonnet45: ModelConfig = {
      apiName: "claude-sonnet-4-5-20250929",
      isVision: true,
      isImageGeneration: false,
      isThinking: true,
      inputTokenCost: 3 / 1_000_000,
      outputTokenCost: 15 / 1_000_000,
    };

    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello Claude!" }] },
    ];

    it("should use adaptive thinking with effort for Claude 4.6+ models", async () => {
      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_46",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: mockClaude46.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: mockClaude46.apiName,
        messages: basicMessages,
        thinkingMode: true,
        budgetTokens: 10000,
      };

      await adapter46.generateResponse(options);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          thinking: { type: "adaptive" },
          output_config: { effort: "high" },
        })
      );
    });

    it("should use explicit effort parameter over budgetTokens for 4.6+ models", async () => {
      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_46_effort",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: mockClaude46.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: mockClaude46.apiName,
        messages: basicMessages,
        thinkingMode: true,
        budgetTokens: 10000, // Would map to "high"
        effort: "low", // But explicit effort should take precedence
      };

      await adapter46.generateResponse(options);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          thinking: { type: "adaptive" },
          output_config: { effort: "low" },
        })
      );
    });

    it("should use budget_tokens for Claude 3.x models (backward compatibility)", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_3x",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: mockClaude3Opus.apiName,
        messages: basicMessages,
        thinkingMode: true,
        budgetTokens: 5000,
      };

      await adapter.generateResponse(options);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          thinking: {
            type: "enabled",
            budget_tokens: 5000,
          },
        })
      );
    });

    it("should throw error when both temperature and topP are provided for Claude 4+ models", async () => {
      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);

      const options: AIRequestOptions = {
        model: mockClaude46.apiName,
        messages: basicMessages,
        temperature: 0.7,
        topP: 0.9,
      };

      await expect(adapter46.generateResponse(options)).rejects.toThrow(
        /do not support using both temperature and top_p/
      );
    });

    it("should allow topP for Claude 3.x models", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_3x_topp",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: mockClaude3Opus.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: mockClaude3Opus.apiName,
        messages: basicMessages,
        temperature: 0.7,
        topP: 0.9,
      };

      await adapter.generateResponse(options);

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          temperature: 0.7,
          top_p: 0.9,
        })
      );
    });

    it("should handle refusal stop reason", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_refuse",
        type: "message",
        role: "assistant",
        content: [],
        model: mockClaude46.apiName,
        stop_reason: "refusal" as any, // Type assertion since SDK types may not include this yet
        usage: { input_tokens: 10, output_tokens: 0 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);
      const options: AIRequestOptions = {
        model: mockClaude46.apiName,
        messages: basicMessages,
      };

      const response = await adapter46.generateResponse(options);

      expect(response.content).toContainEqual(
        expect.objectContaining({
          type: "error",
          publicMessage: expect.stringContaining("declined to respond"),
        })
      );
    });

    it("should handle model_context_window_exceeded stop reason", async () => {
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_context",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Partial response..." }],
        model: mockClaude46.apiName,
        stop_reason: "model_context_window_exceeded" as any,
        usage: { input_tokens: 100000, output_tokens: 50 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);
      const options: AIRequestOptions = {
        model: mockClaude46.apiName,
        messages: basicMessages,
      };

      const response = await adapter46.generateResponse(options);

      expect(response.content).toContainEqual(
        expect.objectContaining({
          type: "error",
          publicMessage: expect.stringContaining("context limit"),
        })
      );
    });

    it("should map budgetTokens to effort levels correctly", async () => {
      const testCases = [
        { budgetTokens: 0, expectedEffort: "low" },
        { budgetTokens: 4000, expectedEffort: "medium" },
        { budgetTokens: 10000, expectedEffort: "high" },
      ];

      const adapter46 = new AnthropicAdapter(mockConfig, mockClaude46);
      const { mockMessagesCreate } = getMockAnthropicClient();

      for (const { budgetTokens, expectedEffort } of testCases) {
        mockMessagesCreate.mockClear();

        const mockApiResponse = {
          id: "msg_effort",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "Hello!" }],
          model: mockClaude46.apiName,
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
        };
        mockMessagesCreate.mockResolvedValue(mockApiResponse);

        await adapter46.generateResponse({
          model: mockClaude46.apiName,
          messages: basicMessages,
          thinkingMode: true,
          budgetTokens,
        });

        expect(mockMessagesCreate).toHaveBeenCalledWith(
          expect.objectContaining({
            output_config: { effort: expectedEffort },
          })
        );
      }
    });

    it("should support Sonnet 4.5 with adaptive thinking", async () => {
      const adapterSonnet45 = new AnthropicAdapter(
        mockConfig,
        mockClaudeSonnet45
      );
      const { mockMessagesCreate } = getMockAnthropicClient();

      const mockApiResponse = {
        id: "msg_sonnet45",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: mockClaudeSonnet45.apiName,
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockMessagesCreate.mockResolvedValue(mockApiResponse);

      await adapterSonnet45.generateResponse({
        model: mockClaudeSonnet45.apiName,
        messages: basicMessages,
        thinkingMode: true,
        effort: "medium",
      });

      expect(mockMessagesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          thinking: { type: "adaptive" },
          output_config: { effort: "medium" },
        })
      );
    });
  });
});
