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
} from "../../types";
import { computeResponseCost } from "../../utils";

// Mock the OpenAI SDK
jest.mock("openai", () => {
  // Mock the constructor and specific methods needed
  const mockCompletionsCreate = jest.fn();
  return jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: mockCompletionsCreate,
      },
    },
    // Mock other methods if needed
  }));
});

// Mock the utility function
jest.mock("../../utils", () => ({
  computeResponseCost: jest.fn(),
}));

// Define mock instances and methods for clarity in tests
const MockedOpenAI = OpenAI as jest.MockedClass<typeof OpenAI>;
const mockCompletionsCreate = new MockedOpenAI().chat.completions
  .create as jest.Mock;
const mockComputeResponseCost = computeResponseCost as jest.Mock;

describe("OpenRouterAdapter", () => {
  let adapter: OpenRouterAdapter;
  const mockVendorConfig: VendorConfig = {
    apiKey: "test-openrouter-key",
    baseURL: "https://custom.openrouter.ai/api/v1", // Test custom baseURL
  };
  const mockModelConfig: ModelConfig = {
    apiName: "openai/gpt-4-turbo", // Example OpenRouter model name
    isVision: true,
    isImageGeneration: false, // Assume this model doesn't generate images
    isThinking: false, // Assume this model doesn't support thinking blocks
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
  };
  const mockModelConfigNoCost: ModelConfig = {
    ...mockModelConfig,
    inputTokenCost: undefined,
    outputTokenCost: undefined,
  };

  beforeEach(() => {
    // Clear mocks before each test
    jest.clearAllMocks();
    // Create a new adapter instance for each test
    adapter = new OpenRouterAdapter(mockVendorConfig, mockModelConfig);
  });

  it("should initialize OpenAI client with correct config", () => {
    expect(MockedOpenAI).toHaveBeenCalledTimes(1);
    expect(MockedOpenAI).toHaveBeenCalledWith({
      apiKey: mockVendorConfig.apiKey,
      baseURL: mockVendorConfig.baseURL, // Verify custom baseURL is used
    });
  });

  it("should initialize OpenAI client with default baseURL if not provided", () => {
    const configWithoutBaseURL: VendorConfig = { apiKey: "test-key-default" };
    new OpenRouterAdapter(configWithoutBaseURL, mockModelConfig);
    expect(MockedOpenAI).toHaveBeenCalledWith({
      apiKey: configWithoutBaseURL.apiKey,
      baseURL: "https://openrouter.ai/api/v1", // Verify default baseURL
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
      systemPrompt: "You are a helpful assistant.",
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
            refusal: null, // Added missing property
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
      // system_fingerprint: null, // Property might not exist depending on SDK version/response
    };

    const mockExpectedUsage: UsageResponse = {
      inputCost: 0.0001, // Mocked calculation result
      outputCost: 0.00015, // Mocked calculation result
      totalCost: 0.00025,
    };

    beforeEach(() => {
      // Setup mock return values for dependencies
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);
      mockComputeResponseCost
        .mockReturnValueOnce(mockExpectedUsage.inputCost) // First call for input cost
        .mockReturnValueOnce(mockExpectedUsage.outputCost); // Second call for output cost
    });

    it("should call OpenAI completions.create with correct parameters", async () => {
      await adapter.generateResponse(mockRequestOptions);

      expect(mockCompletionsCreate).toHaveBeenCalledTimes(1);
      expect(mockCompletionsCreate).toHaveBeenCalledWith({
        model: mockRequestOptions.model,
        messages: [
          { role: "system", content: mockRequestOptions.systemPrompt },
          { role: "user", content: "Hello there!" }, // Simple text extraction assumed for now
        ],
        max_tokens: mockRequestOptions.maxTokens,
        temperature: mockRequestOptions.temperature,
      });
    });

    it("should handle messages with simple string content", async () => {
      const optionsWithStringContent: AIRequestOptions = {
        ...mockRequestOptions,
        // Corrected content to be ContentBlock[]
        messages: [
          { role: "user", content: [{ type: "text", text: "Just a string" }] },
        ],
      };
      await adapter.generateResponse(optionsWithStringContent);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: "system", content: mockRequestOptions.systemPrompt },
            // Ensure the expected content matches the corrected input format
            { role: "user", content: "Just a string" },
          ],
        })
      );
    });

    // RENAME and REWRITE this test to reflect correct multimodal handling
    it("should correctly format multimodal content (text and image) for user messages", async () => {
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation(); // Keep spy to ensure NO warnings are issued for valid multimodal
      const complexMessage: Message = {
        role: "user",
        content: [
          { type: "text", text: "First part." },
          { type: "image", url: "http://example.com/img.png" }, // Use ImageBlock
          { type: "text", text: "Second part." },
        ],
      };
      const optionsWithComplexContent: AIRequestOptions = {
        ...mockRequestOptions,
        messages: [complexMessage],
      };

      // Ensure the adapter is vision capable for this test
      adapter.isVisionCapable = true;

      await adapter.generateResponse(optionsWithComplexContent);

      // Ensure the old warning is NOT called
      expect(consoleWarnSpy).not.toHaveBeenCalledWith(
        "OpenRouter adapter received complex content block, attempting simple text extraction."
      );

      // Check that completions.create was called with the correct multimodal format
      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: "system", content: mockRequestOptions.systemPrompt },
            {
              role: "user",
              content: [
                // Expect an array of content parts
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
        usage: undefined, // Simulate missing usage
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
        ], // Simulate null content
      });

      await expect(
        adapter.generateResponse(mockRequestOptions)
      ).rejects.toThrow("No content received from OpenRouter");
    });
  });

  describe("sendChat", () => {
    // Removed id and userId as they are not in the Chat type
    // Added missing required properties
    const mockChat: Chat = {
      model: mockModelConfig.apiName,
      systemPrompt: "Act like a pirate.",
      responseHistory: [
        { role: "user", content: [{ type: "text", text: "Ahoy!" }] },
        { role: "assistant", content: [{ type: "text", text: "Avast ye!" }] },
      ],
      prompt: "Where's the treasure?",
      maxTokens: 100,
      visionUrl: null, // Added null value
      imageURL: null,
      budgetTokens: null, // Added null value
    };

    const mockAIResponse: AIResponse = {
      role: "assistant",
      content: [{ type: "text", text: "On Treasure Island, matey!" }],
      usage: { inputCost: 0.1, outputCost: 0.2, totalCost: 0.3 }, // Example usage
    };

    beforeEach(() => {
      // Mock generateResponse as sendChat calls it internally
      jest.spyOn(adapter, "generateResponse").mockResolvedValue(mockAIResponse);
    });

    it("should map Chat object to AIRequestOptions and call generateResponse", async () => {
      await adapter.sendChat(mockChat);

      expect(adapter.generateResponse).toHaveBeenCalledTimes(1);
      const expectedOptions: AIRequestOptions = {
        model: mockChat.model,
        messages: [
          ...mockChat.responseHistory,
          {
            role: "user",
            content: [{ type: "text", text: mockChat.prompt! }],
          }, // Added prompt
        ],
        // Reflect adapter's mapping: null -> undefined
        maxTokens: mockChat.maxTokens === null ? undefined : mockChat.maxTokens,
        systemPrompt: mockChat.systemPrompt,
        visionUrl: undefined, // Correctly maps null from mockChat.visionUrl to undefined
        // Add other mapped properties if the adapter uses them (e.g., modelId, budgetTokens)
        // modelId: mockChat.modelId, // Example if needed
        // budgetTokens: undefined, // Example if budgetTokens: null maps to undefined
      };
      expect(adapter.generateResponse).toHaveBeenCalledWith(expectedOptions);
    });

    it("should handle Chat object without a prompt", async () => {
      // Set prompt to empty string "" as it's required by Chat type
      const chatWithoutPrompt: Chat = { ...mockChat, prompt: "" };
      await adapter.sendChat(chatWithoutPrompt);

      expect(adapter.generateResponse).toHaveBeenCalledTimes(1);
      const expectedOptions: AIRequestOptions = {
        model: chatWithoutPrompt.model,
        // Adapter logic adds prompt message only if chat.prompt is truthy.
        // Since it's "", it shouldn't be added here.
        messages: chatWithoutPrompt.responseHistory,
        // Reflect adapter's mapping: null -> undefined
        maxTokens:
          chatWithoutPrompt.maxTokens === null
            ? undefined
            : chatWithoutPrompt.maxTokens,
        systemPrompt: chatWithoutPrompt.systemPrompt,
        visionUrl: undefined, // Correctly maps null to undefined
        // modelId: chatWithoutPrompt.modelId, // Example if needed
        // budgetTokens: undefined, // Example if needed
      };
      expect(adapter.generateResponse).toHaveBeenCalledWith(expectedOptions);
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
    it("should throw an error as it is not supported", async () => {
      // Removed id and userId, added missing required properties
      const mockChat: Chat = {
        model: mockModelConfig.apiName,
        prompt: "Generate an image",
        responseHistory: [],
        visionUrl: null,
        imageURL: null,
        maxTokens: null, // Use null as per type def
        budgetTokens: null,
      };
      await expect(adapter.generateImage(mockChat)).rejects.toThrow(
        "Image generation not directly supported by OpenRouter adapter. Check specific model capabilities."
      );
    });
  });

  describe("sendMCPChat", () => {
    it("should throw an error as it is not supported", async () => {
      // Removed id and userId, added missing required properties
      const mockChat: Chat = {
        model: mockModelConfig.apiName,
        prompt: "Use a tool",
        responseHistory: [],
        visionUrl: null,
        imageURL: null,
        maxTokens: null, // Use null as per type def
        budgetTokens: null,
      };
      const mockMCPToolData: any = {}; // Placeholder for MCPTool data
      await expect(
        adapter.sendMCPChat(mockChat, mockMCPToolData)
      ).rejects.toThrow(
        "MCP tool support via OpenRouter depends on the routed model and is not explicitly implemented here."
      );
    });
  });
});
