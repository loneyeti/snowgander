import OpenAI from "openai";
import { OpenRouterAdapter } from "../openrouter";
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  AIResponse,
  Chat,
  ChatResponse,
  ContentBlock,
  TextBlock,
  UsageResponse,
  Message,
  NotImplementedError,
  ImageGenerationResponse,
  ImageBlock,
  ImageDataBlock,
} from "../../types";
import { computeResponseCost } from "../../utils";

// Mock the OpenAI SDK
jest.mock("openai", () => {
  const mockCompletionsCreate = jest.fn();
  return jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: mockCompletionsCreate,
      },
    },
  }));
});

// Mock the utility function
jest.mock("../../utils", () => ({
  computeResponseCost: jest.fn(),
}));

const MockedOpenAI = OpenAI as jest.MockedClass<typeof OpenAI>;
const mockCompletionsCreate = new MockedOpenAI().chat.completions
  .create as jest.Mock;
const mockComputeResponseCost = computeResponseCost as jest.Mock;

describe("OpenRouterAdapter", () => {
  let adapter: OpenRouterAdapter;
  const mockVendorConfig: VendorConfig = {
    apiKey: "test-openrouter-key",
    baseURL: "https://custom.openrouter.ai/api/v1",
  };
  const mockModelConfig: ModelConfig = {
    apiName: "openai/gpt-4-turbo",
    isVision: true,
    isImageGeneration: false,
    isThinking: true,
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
  };
  const mockModelConfigNoCost: ModelConfig = {
    ...mockModelConfig,
    inputTokenCost: undefined,
    outputTokenCost: undefined,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new OpenRouterAdapter(mockVendorConfig, mockModelConfig);
  });

  it("should initialize OpenAI client with correct config", () => {
    expect(MockedOpenAI).toHaveBeenCalledTimes(1);
    expect(MockedOpenAI).toHaveBeenCalledWith({
      apiKey: mockVendorConfig.apiKey,
      baseURL: mockVendorConfig.baseURL,
    });
  });

  it("should initialize OpenAI client with default baseURL if not provided", () => {
    const configWithoutBaseURL: VendorConfig = { apiKey: "test-key-default" };
    new OpenRouterAdapter(configWithoutBaseURL, mockModelConfig);
    expect(MockedOpenAI).toHaveBeenCalledWith({
      apiKey: configWithoutBaseURL.apiKey,
      baseURL: "https://openrouter.ai/api/v1",
    });
  });

  it("should set capabilities and costs from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(mockModelConfig.isVision);
    expect(adapter.isImageGenerationCapable).toBe(
      mockModelConfig.isImageGeneration
    );
    expect(adapter.isThinkingCapable).toBe(mockModelConfig.isThinking);
    expect(adapter.inputTokenCost).toBe(mockModelConfig.inputTokenCost);
    expect(adapter.outputTokenCost).toBe(mockModelConfig.outputTokenCost);
  });

  describe("generateResponse", () => {
    const mockRequestOptions: AIRequestOptions = {
      model: mockModelConfig.apiName,
      messages: [
        { role: "user", content: [{ type: "text", text: "Hello there!" }] },
      ],
      maxTokens: 150,
      temperature: 0.7,
      systemPrompt: "You is a helpful assistant.",
      budgetTokens: 50,
    };

    const mockApiResponse: OpenAI.Chat.Completions.ChatCompletion = {
      id: "chatcmpl-123",
      object: "chat.completion",
      created: 1677652288,
      model: mockModelConfig.apiName,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "General Kenobi!",
            refusal: null,
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    };

    const mockApiResponseWithReasoning: OpenAI.Chat.Completions.ChatCompletion =
      {
        ...mockApiResponse,
        choices: [
          {
            ...mockApiResponse.choices[0],
            message: {
              ...(mockApiResponse.choices[0].message as any),
              reasoning: "Thinking step 1...",
            },
          },
        ],
      };

    const mockExpectedUsage: UsageResponse = {
      inputCost: 0.0001,
      outputCost: 0.00015,
      totalCost: 0.00025,
    };

    beforeEach(() => {
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);
      mockComputeResponseCost
        .mockReturnValueOnce(mockExpectedUsage.inputCost)
        .mockReturnValueOnce(mockExpectedUsage.outputCost);
    });

    it("should call OpenAI completions.create with correct parameters", async () => {
      await adapter.generateResponse(mockRequestOptions);

      expect(mockCompletionsCreate).toHaveBeenCalledTimes(1);
      expect(mockCompletionsCreate).toHaveBeenCalledWith({
        model: mockRequestOptions.model,
        messages: [
          { role: "system", content: mockRequestOptions.systemPrompt },
          { role: "user", content: "Hello there!" },
        ],
        max_tokens: mockRequestOptions.maxTokens,
        temperature: mockRequestOptions.temperature,
        reasoning: { max_tokens: mockRequestOptions.budgetTokens },
      });
    });

    it("should NOT pass reasoning param if budgetTokens is missing or zero", async () => {
      const optionsWithoutBudget: AIRequestOptions = {
        ...mockRequestOptions,
        budgetTokens: 0,
      };
      await adapter.generateResponse(optionsWithoutBudget);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.not.objectContaining({
          reasoning: expect.anything(),
        })
      );

      const optionsWithNullBudget: AIRequestOptions = {
        ...mockRequestOptions,
        budgetTokens: undefined,
      };
      await adapter.generateResponse(optionsWithNullBudget);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.not.objectContaining({
          reasoning: expect.anything(),
        })
      );
    });

    it("should NOT pass reasoning param if isThinkingCapable is false", async () => {
      adapter.isThinkingCapable = false;
      await adapter.generateResponse(mockRequestOptions);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.not.objectContaining({
          reasoning: expect.anything(),
        })
      );
      adapter.isThinkingCapable = mockModelConfig.isThinking;
    });

    it("should handle messages with simple string content", async () => {
      const optionsWithStringContent: AIRequestOptions = {
        ...mockRequestOptions,
        messages: [
          { role: "user", content: [{ type: "text", text: "Just a string" }] },
        ],
      };
      await adapter.generateResponse(optionsWithStringContent);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: "system", content: mockRequestOptions.systemPrompt },
            { role: "user", content: "Just a string" },
          ],
        })
      );
    });

    it("should correctly format multimodal content (text and image) for user messages", async () => {
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();
      const complexMessage: Message = {
        role: "user",
        content: [
          { type: "text", text: "First part." },
          { type: "image", url: "http://example.com/img.png" },
          { type: "text", text: "Second part." },
        ],
      };
      const optionsWithComplexContent: AIRequestOptions = {
        ...mockRequestOptions,
        messages: [complexMessage],
      };

      adapter.isVisionCapable = true;

      await adapter.generateResponse(optionsWithComplexContent);

      expect(consoleWarnSpy).not.toHaveBeenCalledWith(
        "OpenRouter adapter received complex content block, attempting simple text extraction."
      );

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: "system", content: mockRequestOptions.systemPrompt },
            {
              role: "user",
              content: [
                { type: "text", text: "First part." },
                {
                  type: "image_url",
                  image_url: { url: "http://example.com/img.png" },
                },
                { type: "text", text: "Second part." },
              ],
            },
          ],
        })
      );
      consoleWarnSpy.mockRestore();
    });

    it("should return AIResponse with correct content and role", async () => {
      const response = await adapter.generateResponse(mockRequestOptions);

      expect(response.role).toBe("assistant");
      expect(response.content).toEqual([
        { type: "text", text: "General Kenobi!" },
      ]);
    });

    it("should parse reasoning field into ThinkingBlock when present", async () => {
      mockCompletionsCreate.mockResolvedValue(mockApiResponseWithReasoning);
      const response = await adapter.generateResponse(mockRequestOptions);

      expect(response.role).toBe("assistant");
      expect(response.content).toEqual([
        {
          type: "thinking",
          thinking: "Thinking step 1...",
          signature: "openrouter",
        },
        { type: "text", text: "General Kenobi!" },
      ]);
    });

    it("should calculate and return usage costs", async () => {
      const response = await adapter.generateResponse(mockRequestOptions);

      expect(mockComputeResponseCost).toHaveBeenCalledTimes(2);
      expect(mockComputeResponseCost).toHaveBeenCalledWith(
        mockApiResponse.usage?.prompt_tokens,
        mockModelConfig.inputTokenCost
      );
      expect(mockComputeResponseCost).toHaveBeenCalledWith(
        mockApiResponse.usage?.completion_tokens,
        mockModelConfig.outputTokenCost
      );
      expect(response.usage).toEqual(mockExpectedUsage);
    });

    it("should return undefined usage if costs are not configured", async () => {
      const adapterNoCost = new OpenRouterAdapter(
        mockVendorConfig,
        mockModelConfigNoCost
      );
      const response = await adapterNoCost.generateResponse(mockRequestOptions);

      expect(mockComputeResponseCost).not.toHaveBeenCalled();
      expect(response.usage).toBeUndefined();
    });

    it("should return undefined usage if API response has no usage data", async () => {
      mockCompletionsCreate.mockResolvedValue({
        ...mockApiResponse,
        usage: undefined,
      });
      const response = await adapter.generateResponse(mockRequestOptions);

      expect(mockComputeResponseCost).not.toHaveBeenCalled();
      expect(response.usage).toBeUndefined();
    });

    it("should throw error if API response has no content", async () => {
      mockCompletionsCreate.mockResolvedValue({
        ...mockApiResponse,
        choices: [
          {
            ...mockApiResponse.choices[0],
            message: { role: "assistant", content: null },
          },
        ],
      });

      await expect(
        adapter.generateResponse(mockRequestOptions)
      ).rejects.toThrow("No content received from OpenRouter");
    });
  });

  describe("sendChat", () => {
    const mockChat: Chat = {
      model: mockModelConfig.apiName,
      systemPrompt: "Act like a pirate.",
      responseHistory: [
        { role: "user", content: [{ type: "text", text: "Ahoy!" }] },
        { role: "assistant", content: [{ type: "text", text: "Avast ye!" }] },
      ],
      prompt: "Where's the treasure?",
      maxTokens: 100,
      visionUrl: null,
      imageURL: null,
      budgetTokens: null,
    };

    const mockAIResponse: AIResponse = {
      role: "assistant",
      content: [{ type: "text", text: "On Treasure Island, matey!" }],
      usage: { inputCost: 0.1, outputCost: 0.2, totalCost: 0.3 },
    };

    beforeEach(() => {
      jest.spyOn(adapter, "generateResponse").mockResolvedValue(mockAIResponse);
    });

    it("should map Chat object to AIRequestOptions and call generateResponse", async () => {
      await adapter.sendChat(mockChat);

      expect(adapter.generateResponse).toHaveBeenCalledTimes(1);
      const expectedOptions: AIRequestOptions = {
        model: mockChat.model,
        messages: [...mockChat.responseHistory],
        maxTokens: mockChat.maxTokens === null ? undefined : mockChat.maxTokens,
        budgetTokens:
          mockChat.budgetTokens === null ? undefined : mockChat.budgetTokens,
        systemPrompt: mockChat.systemPrompt,
      };
      expect(adapter.generateResponse).toHaveBeenCalledWith(
        expect.objectContaining(expectedOptions)
      );
    });

    it("should handle Chat object without a prompt", async () => {
      const chatWithoutPrompt: Chat = { ...mockChat, prompt: "" };
      await adapter.sendChat(chatWithoutPrompt);

      expect(adapter.generateResponse).toHaveBeenCalledTimes(1);
      const expectedOptions: AIRequestOptions = {
        model: chatWithoutPrompt.model,
        messages: chatWithoutPrompt.responseHistory,
        maxTokens:
          chatWithoutPrompt.maxTokens === null
            ? undefined
            : chatWithoutPrompt.maxTokens,
        budgetTokens:
          chatWithoutPrompt.budgetTokens === null
            ? undefined
            : chatWithoutPrompt.budgetTokens,
        systemPrompt: chatWithoutPrompt.systemPrompt,
      };
      expect(adapter.generateResponse).toHaveBeenCalledWith(
        expect.objectContaining(expectedOptions)
      );
    });

    it("should return ChatResponse based on generateResponse result", async () => {
      const response = await adapter.sendChat(mockChat);

      expect(response).toEqual({
        role: mockAIResponse.role,
        content: mockAIResponse.content,
        usage: mockAIResponse.usage,
      });
    });
  });

  describe("generateImage", () => {
    it("should throw NotImplementedError with correct message", async () => {
      const dummyOptions: AIRequestOptions = { model: "test", messages: [] };
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        NotImplementedError
      );
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        "Image generation not directly supported by OpenRouter adapter. Use a specific image generation model/vendor."
      );
    });
  });

  describe("streamResponse", () => {
    const mockRequestOptions: AIRequestOptions = {
      model: mockModelConfig.apiName,
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Tell me a story." }],
        },
      ],
    };

    // This is our mock stream generator
    async function* mockStreamGenerator(chunks: any[]) {
      for (const chunk of chunks) {
        yield chunk;
      }
    }

    it("should call completions.create with stream: true", async () => {
      mockCompletionsCreate.mockResolvedValue(mockStreamGenerator([]));
      const stream = adapter.streamResponse(mockRequestOptions);
      await stream.next(); // Start the generator

      expect(mockCompletionsCreate).toHaveBeenCalledTimes(1);
      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          stream: true,
        })
      );
    });

    it("should yield TextBlocks for each content delta", async () => {
      const mockChunks = [
        { choices: [{ delta: { role: "assistant", content: null } }] },
        { choices: [{ delta: { content: "Once " } }] },
        { choices: [{ delta: { content: "upon " } }] },
        { choices: [{ delta: { content: "a time..." } }] },
        { choices: [{ delta: { content: null } }] }, // End token
      ];
      mockCompletionsCreate.mockResolvedValue(mockStreamGenerator(mockChunks));

      const stream = adapter.streamResponse(mockRequestOptions);
      const receivedBlocks: ContentBlock[] = [];
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      const expectedBlocks: TextBlock[] = [
        { type: "text", text: "Once " },
        { type: "text", text: "upon " },
        { type: "text", text: "a time..." },
      ];

      expect(receivedBlocks).toEqual(expectedBlocks);
    });

    it("should handle an empty stream gracefully", async () => {
      mockCompletionsCreate.mockResolvedValue(mockStreamGenerator([]));
      const stream = adapter.streamResponse(mockRequestOptions);
      const receivedBlocks: ContentBlock[] = [];
      for await (const block of stream) {
        receivedBlocks.push(block);
      }
      expect(receivedBlocks).toEqual([]);
    });

    it("should yield an ErrorBlock if the API call fails", async () => {
      const apiError = new Error("API Connection Failed");
      mockCompletionsCreate.mockRejectedValue(apiError);

      const stream = adapter.streamResponse(mockRequestOptions);
      const receivedBlocks: ContentBlock[] = [];
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks.length).toBe(1);
      expect(receivedBlocks[0].type).toBe("error");
      expect((receivedBlocks[0] as any).publicMessage).toContain(
        "An error occurred"
      );
      expect((receivedBlocks[0] as any).privateMessage).toBe(
        "API Connection Failed"
      );
    });
  });
});
