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

const mockCompletionsCreate = jest.fn();
const mockImagesGenerate = jest.fn(); // Keep mock even if method removed, for safety

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => {
    return {
      responses: {
        create: mockCompletionsCreate,
      },
      images: {
        generate: mockImagesGenerate, // Keep mock definition
      },
    };
  });
});

const getMockOpenAIClient = () => {
  const MockOpenAIConstructor = OpenAI as unknown as jest.Mock;
  const mockClientInstance = MockOpenAIConstructor.mock.instances[0];
  return {
    mockCompletionsCreate,
    mockImagesGenerate,
    MockOpenAIConstructor,
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
      const { mockCompletionsCreate } = getMockOpenAIClient();
      const mockApiResponse = {
        // Basic response structure is enough
        id: "resp-1",
        model: mockVisionModel.apiName,
        output_text: "Hi there!", // Use output_text for simplicity
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hi there!" }],
          },
        ],
        usage: { input_tokens: 10, output_tokens: 5 },
      };
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(basicOptions);

      expect(mockCompletionsCreate).toHaveBeenCalledTimes(1);
      expect(mockCompletionsCreate).toHaveBeenCalledWith({
        model: mockVisionModel.apiName,
        instructions: undefined, // No system prompt in basicOptions
        input: [
          // Expect mapped input structure
          { role: "user", content: [{ type: "input_text", text: "Hello!" }] },
        ],
      });
    });

    it("should return mapped AIResponse with content and usage", async () => {
      const { mockCompletionsCreate } = getMockOpenAIClient();
      const mockApiResponse = {
        id: "resp-2",
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
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

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
      const { mockCompletionsCreate } = getMockOpenAIClient();
      const systemPrompt = "You are a helpful assistant.";
      const optionsWithSystem: AIRequestOptions = {
        ...basicOptions,
        systemPrompt: systemPrompt,
      };
      const mockApiResponse = {
        id: "resp-sys",
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
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithSystem);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          instructions: systemPrompt,
          input: [
            { role: "user", content: [{ type: "input_text", text: "Hello!" }] },
          ],
        })
      );
    });

    it("should map ImageDataBlock to data URL for vision input", async () => {
      const { mockCompletionsCreate } = getMockOpenAIClient();
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
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImage);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
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
      const { mockCompletionsCreate } = getMockOpenAIClient();
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
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImageUrl);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
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
      const { mockCompletionsCreate } = getMockOpenAIClient();

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
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      await nonVisionAdapter.generateResponse(optionsWithImage);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              role: "user",
              content: [
                { type: "input_text", text: "Hello again" },
              ],
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
        const { mockCompletionsCreate } = getMockOpenAIClient();
        const apiError = new Error("API Failure");
        mockCompletionsCreate.mockRejectedValue(apiError);

        await expect(adapter.generateResponse(basicOptions)).rejects.toThrow(apiError);
    });

    it("should throw error if no content is received", async () => {
        const { mockCompletionsCreate } = getMockOpenAIClient();
        const mockApiResponse = { // Response missing output_text and output content
            id: "resp-nocontent",
            model: mockVisionModel.apiName,
            output: [], // Empty output array
            usage: { input_tokens: 5, output_tokens: 0 },
        };
        mockCompletionsCreate.mockResolvedValue(mockApiResponse);

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
              content: [ // Image block should be excluded
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
