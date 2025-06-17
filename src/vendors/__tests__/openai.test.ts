import { OpenAIAdapter } from "../openai";
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
  ChatResponse,
} from "../../types";
import OpenAI from "openai";
import { computeResponseCost } from "../../utils";

// Mock the entire OpenAI SDK
type MockedOpenAI = jest.Mocked<OpenAI>;

// Enhanced mock implementation to satisfy OpenAI type requirements
const mockResponsesCreate = jest.fn();
const mockClient = {
  responses: {
    create: mockResponsesCreate,
  },
  chat: {
    completions: {
      create: jest.fn(), // Keep this for any tests that might use it
    },
  },
};

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => mockClient);
});

const getMockOpenAIClient = () => {
  const MockOpenAIConstructor = OpenAI as unknown as jest.Mock;
  return {
    mockResponsesCreate,
    MockOpenAIConstructor,
    mockClient,
  };
};

describe("OpenAIAdapter", () => {
  let adapter: OpenAIAdapter;
  const mockConfig: VendorConfig = { apiKey: "test-openai-key" };
  const mockVisionModel: ModelConfig = {
    apiName: "gpt-4-turbo",
    isVision: true,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
  };

  const mockFullCostModel: ModelConfig = {
    apiName: "gpt-4-turbo-plus",
    isVision: true,
    isImageGeneration: true, // Assuming this model can do it all
    isThinking: false,
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
    imageOutputTokenCost: 100 / 1_000_000, // New cost
    webSearchCost: 0.05, // New flat fee
  };

  const mockApiResponseWithImage = {
    id: "resp-img-gen",
    model: mockFullCostModel.apiName,
    output_text: "Here is the image you requested.",
    output: [
      {
        type: "image_generation_call",
        id: "ig_123",
        status: "completed",
      },
      {
        type: "message",
        role: "assistant",
        content: [
          { type: "output_text", text: "Here is the image you requested." },
        ],
      },
    ],
    usage: { input_tokens: 50, output_tokens: 100 },
  };

  const mockApiResponseWithWebSearch = {
    id: "resp-web-search",
    model: mockFullCostModel.apiName,
    output_text: "According to my search...",
    output: [
      {
        type: "web_search_call",
        id: "ws_123",
        status: "completed",
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "According to my search..." }],
      },
    ],
    usage: { input_tokens: 20, output_tokens: 150 },
  };

  const mockApiResponseWithBoth = {
    id: "resp-both",
    model: mockFullCostModel.apiName,
    output_text: "I searched the web to create this image.",
    output: [
      {
        type: "web_search_call",
        id: "ws_456",
        status: "completed",
      },
      {
        type: "image_generation_call",
        id: "ig_789",
        status: "completed",
      },
      {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: "I searched the web to create this image.",
          },
        ],
      },
    ],
    usage: { input_tokens: 70, output_tokens: 120 },
  };
  // Removed mockDalleModel as generateImage is removed

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new OpenAIAdapter(mockConfig, mockVisionModel);
    const { MockOpenAIConstructor } = getMockOpenAIClient();
    expect(MockOpenAIConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
      organization: mockConfig.organizationId,
      baseURL: mockConfig.baseURL,
    });
  });

  it("should initialize OpenAI client with correct config", () => {
    const { MockOpenAIConstructor } = getMockOpenAIClient();
    expect(MockOpenAIConstructor).toHaveBeenCalledTimes(1);
    expect(MockOpenAIConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
      organization: undefined,
      baseURL: undefined,
    });
  });

  // --- generateResponse Tests ---
  describe("generateResponse", () => {
    // Define basicOptions here for use within this describe block
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockVisionModel.apiName,
      messages: basicMessages,
    };

    it("should call responses.create with mapped messages", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const mockApiResponse = {
        id: "chat-1",
        model: mockVisionModel.apiName,
        output_text: "Hi there!",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hi there!" }],
          },
        ],
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(basicOptions);

      expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
      expect(mockResponsesCreate).toHaveBeenCalledWith({
        model: mockVisionModel.apiName,
        instructions: undefined, // No system prompt in basicOptions
        input: [
          // Expect mapped input structure
          { role: "user", content: [{ type: "input_text", text: "Hello!" }] },
        ],
      });
    });

    it("should return mapped AIResponse with content and usage", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const mockApiResponse = {
        id: "chat-2",
        model: mockVisionModel.apiName,
        output_text: "Response text",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Response text" }],
          },
        ],
        usage: { input_tokens: 15, output_tokens: 8 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      const expectedUsage: UsageResponse = {
        inputCost: computeResponseCost(15, mockVisionModel.inputTokenCost!),
        outputCost: computeResponseCost(8, mockVisionModel.outputTokenCost!),
        totalCost:
          computeResponseCost(15, mockVisionModel.inputTokenCost!) +
          computeResponseCost(8, mockVisionModel.outputTokenCost!),
      };
      const expectedResponse: AIResponse = {
        role: "assistant",
        content: [{ type: "text", text: "Response text" }],
        usage: expectedUsage,
      };

      const response = await adapter.generateResponse(basicOptions);
      expect(response).toEqual(expectedResponse);
    });

    it("should include system prompt as instructions", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const systemPrompt = "You are a helpful assistant.";
      const optionsWithSystem: AIRequestOptions = {
        ...basicOptions,
        systemPrompt: systemPrompt,
      };
      const mockApiResponse = {
        id: "chat-sys",
        model: mockVisionModel.apiName,
        output_text: "Okay.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Okay." }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 2 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithSystem);

      expect(mockResponsesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          instructions: systemPrompt,
          input: [
            { role: "user", content: [{ type: "input_text", text: "Hello!" }] },
          ],
        })
      );
    });

    it("should map ImageDataBlock to data URL for vision input", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const imageData: ImageDataBlock = {
        type: "image_data",
        mimeType: "image/png",
        base64Data: "base64encodedstring",
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [imageData, { type: "text", text: "What is this?" }],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
      };
      const mockApiResponse = {
        id: "resp-img",
        model: mockVisionModel.apiName,
        output_text: "It's an image.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "It's an image." }],
          },
        ],
        usage: { input_tokens: 50, output_tokens: 5 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImage);

      expect(mockResponsesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              role: "user",
              content: [
                {
                  type: "input_image",
                  image_url: "data:image/png;base64,base64encodedstring",
                },
                { type: "input_text", text: "What is this?" },
              ],
            },
          ],
        })
      );
    });

    it("should map ImageBlock (URL) to image_url for vision input", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const imageUrl = "http://example.com/image.jpg";
      const imageBlock: ImageBlock = { type: "image", url: imageUrl };
      const messagesWithImageUrl: Message[] = [
        {
          role: "user",
          content: [imageBlock, { type: "text", text: "Describe this image." }],
        },
      ];
      const optionsWithImageUrl: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImageUrl,
      };
      const mockApiResponse = {
        id: "resp-url",
        model: mockVisionModel.apiName,
        output_text: "A description.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "A description." }],
          },
        ],
        usage: { input_tokens: 60, output_tokens: 6 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImageUrl);

      expect(mockResponsesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              role: "user",
              content: [
                {
                  type: "input_image",
                  image_url: imageUrl,
                },
                { type: "input_text", text: "Describe this image." },
              ],
            },
          ],
        })
      );
    });

    it("should skip image blocks if adapter is not vision capable", async () => {
      const nonVisionModel: ModelConfig = {
        ...mockVisionModel,
        isVision: false,
      };
      // Create non-vision adapter instance *within the test*
      const nonVisionAdapter = new OpenAIAdapter(mockConfig, nonVisionModel);
      const { mockResponsesCreate } = getMockOpenAIClient();

      const imageData: ImageDataBlock = {
        type: "image_data",
        mimeType: "image/jpeg",
        base64Data: "anotherbase64",
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [imageData, { type: "text", text: "Hello again" }],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
        model: nonVisionModel.apiName,
      };
      const mockApiResponse = {
        id: "resp-novision",
        model: nonVisionModel.apiName,
        output_text: "Hi.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hi." }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 1 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      await nonVisionAdapter.generateResponse(optionsWithImage);

      expect(mockResponsesCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              role: "user",
              content: [{ type: "input_text", text: "Hello again" }],
            },
          ],
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Image block provided for non-vision model")
      );
      consoleWarnSpy.mockRestore();
    });

    it("should handle API errors", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const apiError = new Error("API Failure");
      mockResponsesCreate.mockRejectedValue(apiError);

      await expect(adapter.generateResponse(basicOptions)).rejects.toThrow(
        apiError
      );
    });

    it("should use imageOutputTokenCost when an image is generated", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      mockResponsesCreate.mockResolvedValue(mockApiResponseWithImage);
      const fullCostAdapter = new OpenAIAdapter(mockConfig, mockFullCostModel);

      const response = await fullCostAdapter.generateResponse(basicOptions);

      const expectedInputCost = computeResponseCost(
        50,
        mockFullCostModel.inputTokenCost!
      );
      // Crucially, this uses the special image cost
      const expectedOutputCost = computeResponseCost(
        100,
        mockFullCostModel.imageOutputTokenCost!
      );
      const expectedTotalCost = expectedInputCost + expectedOutputCost;

      expect(response.usage).toBeDefined();
      expect(response.usage?.inputCost).toBeCloseTo(expectedInputCost);
      expect(response.usage?.outputCost).toBeCloseTo(expectedOutputCost);
      expect(response.usage?.webSearchCost).toBeUndefined();
      expect(response.usage?.totalCost).toBeCloseTo(expectedTotalCost);
    });

    it("should add webSearchCost when the web search tool is used", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      mockResponsesCreate.mockResolvedValue(mockApiResponseWithWebSearch);
      const fullCostAdapter = new OpenAIAdapter(mockConfig, mockFullCostModel);

      const response = await fullCostAdapter.generateResponse(basicOptions);

      const expectedInputCost = computeResponseCost(
        20,
        mockFullCostModel.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        150,
        mockFullCostModel.outputTokenCost!
      );
      const expectedWebSearchCost = mockFullCostModel.webSearchCost!;
      const expectedTotalCost =
        expectedInputCost + expectedOutputCost + expectedWebSearchCost;

      expect(response.usage).toBeDefined();
      expect(response.usage?.inputCost).toBeCloseTo(expectedInputCost);
      expect(response.usage?.outputCost).toBeCloseTo(expectedOutputCost);
      expect(response.usage?.webSearchCost).toBe(expectedWebSearchCost);
      expect(response.usage?.totalCost).toBeCloseTo(expectedTotalCost);
    });

    it("should apply both imageOutputTokenCost and webSearchCost when both tools are used", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      mockResponsesCreate.mockResolvedValue(mockApiResponseWithBoth);
      const fullCostAdapter = new OpenAIAdapter(mockConfig, mockFullCostModel);

      const response = await fullCostAdapter.generateResponse(basicOptions);

      const expectedInputCost = computeResponseCost(
        70,
        mockFullCostModel.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        120,
        mockFullCostModel.imageOutputTokenCost!
      ); // Image cost applies
      const expectedWebSearchCost = mockFullCostModel.webSearchCost!;
      const expectedTotalCost =
        expectedInputCost + expectedOutputCost + expectedWebSearchCost;

      expect(response.usage).toBeDefined();
      expect(response.usage?.inputCost).toBeCloseTo(expectedInputCost);
      expect(response.usage?.outputCost).toBeCloseTo(expectedOutputCost);
      expect(response.usage?.webSearchCost).toBe(expectedWebSearchCost);
      expect(response.usage?.totalCost).toBeCloseTo(expectedTotalCost);
    });

    it("should throw error if no content is received", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const mockApiResponse = {
        // Response missing output_text and output content
        id: "resp-nocontent",
        model: mockVisionModel.apiName,
        output: [], // Empty output array
        usage: { input_tokens: 5, output_tokens: 0 },
      };
      mockResponsesCreate.mockResolvedValue(mockApiResponse);

      await expect(adapter.generateResponse(basicOptions)).rejects.toThrow(
        "No content or output received from OpenAI"
      );
    });
  });

  // --- sendChat Tests ---
  describe("sendChat", () => {
    let generateResponseSpy: jest.SpyInstance;
    const baseChat: Partial<Chat> = {
      model: mockVisionModel.apiName,
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
          content: [{ type: "text", text: "Mocked response" }],
          usage: { inputCost: 0.01, outputCost: 0.02, totalCost: 0.03 },
        });
    });

    afterEach(() => {
      generateResponseSpy.mockRestore();
    });

    it("should call generateResponse with mapped messages and options", async () => {
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
        temperature: undefined,
        systemPrompt: chatWithHistory.systemPrompt,
      });
    });

    it("should include visionUrl as an ImageBlock in the last message if present", async () => {
      const visionUrl = "http://example.com/image.png";
      const chatWithVision: Chat = {
        ...baseChat,
        visionUrl: visionUrl,
        prompt: "What's in this image?",
      } as Chat;

      await adapter.sendChat(chatWithVision);

      expect(generateResponseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({
              role: "user",
              content: expect.arrayContaining([
                { type: "text", text: "What's in this image?" },
                { type: "image", url: visionUrl }, // Check for ImageBlock
              ]),
            }),
          ]),
        })
      );
    });

    it("should ignore visionUrl if adapter is not vision capable", async () => {
      const nonVisionModel: ModelConfig = {
        ...mockVisionModel,
        isVision: false,
      };
      const nonVisionAdapter = new OpenAIAdapter(mockConfig, nonVisionModel);
      const nonVisionSpy = jest
        .spyOn(nonVisionAdapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "No vision" }],
          usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
        });
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      const visionUrl = "http://example.com/image.png";
      const chatWithVision: Chat = {
        ...baseChat,
        visionUrl: visionUrl,
        prompt: "What's in this image?",
      } as Chat;

      await nonVisionAdapter.sendChat(chatWithVision);

      expect(nonVisionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: expect.arrayContaining([
            expect.objectContaining({
              role: "user",
              content: [
                // Image block should be excluded
                { type: "text", text: "What's in this image?" },
              ],
            }),
          ]),
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Image data provided to a non-vision capable model"
        )
      );

      nonVisionSpy.mockRestore();
      consoleWarnSpy.mockRestore();
    });

    it("should return the ChatResponse with content and usage from generateResponse", async () => {
      const chat: Chat = baseChat as Chat;
      const expectedUsage: UsageResponse = {
        inputCost: 0.01,
        outputCost: 0.02,
        totalCost: 0.03,
      };
      generateResponseSpy.mockResolvedValue({
        role: "assistant",
        content: [{ type: "text", text: "Specific response" }],
        usage: expectedUsage,
      });

      const result = await adapter.sendChat(chat);

      expect(result).toEqual<ChatResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Specific response" }],
        usage: expectedUsage,
      });
    });
  });

  // Removed generateImage tests as this method was removed from OpenAIAdapter

  // Removed sendMCPChat tests as this method was removed from OpenAIAdapter
});

describe("OpenAIAdapter Reasoning Logic", () => {
  let adapter: OpenAIAdapter;
  const vendorConfig: VendorConfig = { apiKey: "test-key" };
  const modelConfig: ModelConfig = {
    apiName: "o4-mini",
    isVision: false,
    isImageGeneration: false,
    isThinking: true,
  };
  const baseRequestOptions: AIRequestOptions = {
    model: modelConfig.apiName,
    messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new OpenAIAdapter(vendorConfig, modelConfig);
    // @ts-ignore
    adapter.client = mockClient;
  });

  it("should not include the reasoning parameter when budgetTokens is null", async () => {
    mockResponsesCreate.mockResolvedValue({
      id: "chat-123",
      model: "o4-mini",
      output_text: "Hi!",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hi!" }],
        },
      ],
      usage: { input_tokens: 5, output_tokens: 2 },
    });

    await adapter.generateResponse({
      ...baseRequestOptions,
      budgetTokens: undefined,
    });

    const calledParams = mockResponsesCreate.mock.calls[0][0];
    expect(calledParams).not.toHaveProperty("reasoning");
  });

  it('should set reasoning effort to "low" when budgetTokens is 0', async () => {
    mockClient.chat.completions.create.mockResolvedValue({
      id: "chat-123",
      model: "o4-mini",
      output_text: "Hi!",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hi!" }],
        },
      ],
      usage: { input_tokens: 5, output_tokens: 2 },
    });

    await adapter.generateResponse({ ...baseRequestOptions, budgetTokens: 0 });

    const calledParams = mockResponsesCreate.mock.calls[0][0];
    expect(calledParams).toHaveProperty("reasoning");
    // @ts-ignore
    expect(calledParams.reasoning).toEqual({ effort: "low" });
  });

  it('should set reasoning effort to "medium" when budgetTokens is between 1 and 8191', async () => {
    mockResponsesCreate.mockResolvedValue({
      id: "chat-123",
      model: "o4-mini",
      output_text: "Hi!",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hi!" }],
        },
      ],
      usage: { input_tokens: 5, output_tokens: 2 },
    });

    await adapter.generateResponse({
      ...baseRequestOptions,
      budgetTokens: 4096,
    });

    const calledParams = mockResponsesCreate.mock.calls[0][0];
    expect(calledParams).toHaveProperty("reasoning");
    // @ts-ignore
    expect(calledParams.reasoning).toEqual({ effort: "medium" });
  });

  it('should set reasoning effort to "high" when budgetTokens is 8192 or greater', async () => {
    mockResponsesCreate.mockResolvedValue({
      id: "chat-123",
      model: "o4-mini",
      output_text: "Hi!",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hi!" }],
        },
      ],
      usage: { input_tokens: 5, output_tokens: 2 },
    });

    await adapter.generateResponse({
      ...baseRequestOptions,
      budgetTokens: 10000,
    });

    const calledParams = mockResponsesCreate.mock.calls[0][0];
    expect(calledParams).toHaveProperty("reasoning");
    // @ts-ignore
    expect(calledParams.reasoning).toEqual({ effort: "high" });
  });
});
