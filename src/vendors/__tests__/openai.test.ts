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
  ChatResponse, // Added missing import
} from "../../types";
import OpenAI from "openai";
import { computeResponseCost } from "../../utils"; // Import for calculating expected costs

// Mock the entire OpenAI SDK
// Use jest.Mocked<T> for better type safety with mocks
type MockedOpenAI = jest.Mocked<OpenAI>;

// Define the mock structure more explicitly matching the adapter's usage
const mockCompletionsCreate = jest.fn();
const mockImagesGenerate = jest.fn();

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => {
    // Return the structure the adapter actually interacts with
    return {
      responses: {
        create: mockCompletionsCreate,
      },
      images: {
        generate: mockImagesGenerate,
      },
      // Add other parts of the client if the adapter uses them
    }; // Simplified return, relying on Jest's mock capabilities
  });
});

// Helper function to get the mocked SDK methods and constructor
const getMockOpenAIClient = () => {
  // Cast via unknown to satisfy TypeScript when dealing with complex mocks
  const MockOpenAIConstructor = OpenAI as unknown as jest.Mock;
  // Get the *instance* created by the mock constructor
  const mockClientInstance = MockOpenAIConstructor.mock.instances[0];
  // Return the original top-level mocks which have the mock methods (.mockResolvedValue etc.)
  return {
    mockCompletionsCreate, // Return the original mock function
    mockImagesGenerate, // Return the original mock function
    MockOpenAIConstructor, // Expose the mock constructor
  };
};

describe("OpenAIAdapter", () => {
  let adapter: OpenAIAdapter;
  const mockConfig: VendorConfig = { apiKey: "test-openai-key" };
  const mockVisionModel: ModelConfig = {
    apiName: "gpt-4-turbo",
    isVision: true,
    isImageGeneration: false, // DALL-E handles this, not the chat model itself
    isThinking: false,
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
  };
  const mockDalleModel: ModelConfig = {
    // Separate config for image generation model
    apiName: "dall-e-3",
    isVision: false,
    isImageGeneration: true,
    isThinking: false,
    // Costs for DALL-E are often per-image, not per-token, handle separately if needed
  };

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    // Create a new adapter instance for each test, assuming it uses the vision model by default
    adapter = new OpenAIAdapter(mockConfig, mockVisionModel);
    // Ensure the mock client is ready for inspection
    const { MockOpenAIConstructor } = getMockOpenAIClient(); // Use correct name from helper
    expect(MockOpenAIConstructor).toHaveBeenCalledWith({
      // Check constructor call
      apiKey: mockConfig.apiKey,
      organization: mockConfig.organizationId, // Will be undefined, which is fine
      baseURL: mockConfig.baseURL, // Will be undefined
    });
  });

  it("should initialize OpenAI client with correct config", () => {
    const { MockOpenAIConstructor } = getMockOpenAIClient(); // Use correct name from helper
    expect(MockOpenAIConstructor).toHaveBeenCalledTimes(1); // Called once in beforeEach
    expect(MockOpenAIConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
      organization: undefined,
      baseURL: undefined,
    });
  });

  it("should reflect capabilities from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(true);
    expect(adapter.isImageGenerationCapable).toBe(false); // Based on mockVisionModel
    expect(adapter.isThinkingCapable).toBe(false);
    expect(adapter.inputTokenCost).toBe(mockVisionModel.inputTokenCost);
    expect(adapter.outputTokenCost).toBe(mockVisionModel.outputTokenCost);

    // Test with a non-vision model config
    const nonVisionModel: ModelConfig = {
      ...mockVisionModel,
      apiName: "gpt-3.5-turbo",
      isVision: false,
    };
    const nonVisionAdapter = new OpenAIAdapter(mockConfig, nonVisionModel);
    expect(nonVisionAdapter.isVisionCapable).toBe(false);
  });

  // --- generateResponse Tests ---

  describe("generateResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockVisionModel.apiName,
      messages: basicMessages,
    };

    it("should call OpenAI responses.create with correct basic parameters", async () => {
      const { mockCompletionsCreate } = getMockOpenAIClient();
      // Adjust mock response to match the structure the adapter expects for text extraction
      const mockApiResponse = {
        id: "resp-123", // Example ID
        model: mockVisionModel.apiName,
        output_text: "Hi there!", // Include the convenience property
        output: [
          // Include the detailed structure as well
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Hi there!" }],
          },
        ],
        usage: { input_tokens: 10, output_tokens: 5 }, // Use correct token names
      };
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(basicOptions);

      expect(mockCompletionsCreate).toHaveBeenCalledTimes(1);
      // Verify the input mapping matches the adapter's logic
      expect(mockCompletionsCreate).toHaveBeenCalledWith({
        model: mockVisionModel.apiName,
        input: [
          // Expect the mapped input structure
          {
            role: "user",
            content: [{ type: "input_text", text: "Hello!" }],
          },
        ],
        instructions: undefined, // System prompt wasn't provided in basicOptions
        // max_tokens and temperature are not params for responses.create
      });
      // Add assertion to ensure the error wasn't thrown
      await expect(
        adapter.generateResponse(basicOptions)
      ).resolves.toBeDefined();
    });

    it("should map OpenAI response to AIResponse format including calculated usage", async () => {
      const { mockCompletionsCreate } = getMockOpenAIClient();
      // Use the same corrected mock structure as the previous test
      const mockApiResponse = {
        id: "resp-xyz",
        model: mockVisionModel.apiName,
        output_text: "Response text.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Response text." }],
          },
        ],
        usage: { input_tokens: 20, output_tokens: 10 },
      };
      mockCompletionsCreate.mockResolvedValue(mockApiResponse);

      const response = await adapter.generateResponse(basicOptions);

      // Calculate expected costs based on the adapter's logic
      const expectedInputCost = computeResponseCost(
        mockApiResponse.usage.input_tokens,
        adapter.inputTokenCost! // Use non-null assertion as cost is defined in mockVisionModel
      );
      const expectedOutputCost = computeResponseCost(
        mockApiResponse.usage.output_tokens,
        adapter.outputTokenCost! // Use non-null assertion
      );
      const expectedTotalCost = expectedInputCost + expectedOutputCost;

      expect(response).toEqual<AIResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Response text." }],
        usage: {
          inputCost: expectedInputCost,
          outputCost: expectedOutputCost,
          totalCost: expectedTotalCost,
        },
        // max_tokens and temperature are not params for responses.create
      });
      // Add assertion to ensure the error wasn't thrown
      await expect(
        adapter.generateResponse(basicOptions)
      ).resolves.toBeDefined();
    });

    it("should include system prompt as instructions", async () => {
      const { mockCompletionsCreate } = getMockOpenAIClient();
      const systemPrompt = "You are a helpful assistant.";
      const optionsWithSystem: AIRequestOptions = {
        ...basicOptions,
        systemPrompt: systemPrompt,
      };
      const mockApiResponse = {
        // Basic response structure is enough
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
          instructions: systemPrompt, // Verify system prompt mapping
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
        // Basic response structure
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
                // Expect mapped content array
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
        // Basic response structure
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
                // Expect mapped content array
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
      // Create non-vision adapter instance
      const nonVisionModel: ModelConfig = {
        ...mockVisionModel,
        isVision: false,
      };
      const nonVisionAdapter = new OpenAIAdapter(mockConfig, nonVisionModel);
      const { mockCompletionsCreate } = getMockOpenAIClient(); // Get the mock associated with this instance potentially? Need to check mock scope. Assuming shared mock for simplicity.

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
        // Basic response structure
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
      // Spy on console.warn
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      await nonVisionAdapter.generateResponse(optionsWithImage);

      expect(mockCompletionsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              role: "user",
              content: [
                // Image block should be filtered out
                { type: "input_text", text: "Hello again" },
              ],
            },
          ],
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Image block provided for non-vision model")
      );
      consoleWarnSpy.mockRestore(); // Clean up spy
    });

    // Add more tests for generateResponse:
    // - Handling maxTokens, temperature (Note: Not applicable to responses.create)
    // - Handling vision input (ImageDataBlock, ImageBlock) mapping
    // - Handling API errors
    // - Handling responses with different finish_reasons
  });

  // --- sendChat Tests ---
  describe("sendChat", () => {
    let generateResponseSpy: jest.SpyInstance;
    const baseChat: Partial<Chat> = {
      // Use Partial for easier test setup
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
      // Mock the instance's generateResponse method before each sendChat test
      generateResponseSpy = jest
        .spyOn(adapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "Mocked response" }],
          usage: { inputCost: 0.01, outputCost: 0.02, totalCost: 0.03 }, // Example usage
        });
    });

    afterEach(() => {
      // Restore the original implementation after each test
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
      } as Chat; // Cast needed because baseChat is Partial

      await adapter.sendChat(chatWithHistory);

      expect(generateResponseSpy).toHaveBeenCalledTimes(1);
      expect(generateResponseSpy).toHaveBeenCalledWith({
        model: chatWithHistory.model,
        messages: [
          // Expect history + current prompt
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
          }, // Current prompt added as user message
        ],
        maxTokens: chatWithHistory.maxTokens,
        temperature: undefined, // Not set in baseChat
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

      // Expect the exact AIRequestOptions object passed to generateResponse
      expect(generateResponseSpy).toHaveBeenCalledWith({
        model: chatWithVision.model,
        messages: [
          {
            role: "user",
            content: [
              // Expect text block first, then image block (matches adapter implementation)
              { type: "text", text: "What's in this image?" },
              { type: "image", url: visionUrl },
            ],
          },
        ],
        maxTokens: chatWithVision.maxTokens,
        temperature: undefined, // As it wasn't set in baseChat or chatWithVision
        systemPrompt: chatWithVision.systemPrompt,
        // visionUrl is correctly omitted here by the sendChat implementation
      });
    });

    it("should ignore visionUrl if adapter is not vision capable", async () => {
      // Create non-vision adapter instance and spy on its method
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
          messages: [
            {
              role: "user",
              content: [
                // Image block should be excluded
                { type: "text", text: "What's in this image?" },
              ],
            },
          ],
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
      // Ensure the spy returns the expected usage
      generateResponseSpy.mockResolvedValue({
        role: "assistant",
        content: [{ type: "text", text: "Specific response" }],
        usage: expectedUsage,
      });

      const result = await adapter.sendChat(chat);

      expect(result).toEqual<ChatResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Specific response" }],
        usage: expectedUsage, // Verify usage is passed through
      });
    });
  });

  // --- generateImage Tests ---
  describe("generateImage", () => {
    const imageGenChat: Partial<Chat> = {
      // Base chat for image gen
      prompt: "A futuristic cityscape",
      // Other fields aren't strictly needed by the current generateImage implementation
    };

    it("should call OpenAI images.generate with correct parameters", async () => {
      // Need an adapter configured for image generation
      const imageAdapter = new OpenAIAdapter(mockConfig, mockDalleModel); // Use DALL-E config
      const { mockImagesGenerate } = getMockOpenAIClient();
      const imageUrl = "http://example.com/generated-image.png";
      mockImagesGenerate.mockResolvedValue({
        created: Date.now() / 1000,
        data: [{ url: imageUrl }],
      });

      await imageAdapter.generateImage(imageGenChat as Chat);

      expect(mockImagesGenerate).toHaveBeenCalledTimes(1);
      expect(mockImagesGenerate).toHaveBeenCalledWith({
        model: "dall-e-3", // Hardcoded in the adapter currently
        prompt: imageGenChat.prompt,
        n: 1,
        size: "1024x1024",
        quality: "standard",
      });
    });

    it("should return the image URL from the response", async () => {
      const imageAdapter = new OpenAIAdapter(mockConfig, mockDalleModel);
      const { mockImagesGenerate } = getMockOpenAIClient();
      const imageUrl = "http://example.com/another-image.png";
      mockImagesGenerate.mockResolvedValue({
        created: Date.now() / 1000,
        data: [{ url: imageUrl }],
      });

      const result = await imageAdapter.generateImage(imageGenChat as Chat);
      expect(result).toBe(imageUrl);
    });

    it("should throw an error if the adapter model is not image generation capable", async () => {
      // Use the default 'adapter' which is configured with mockVisionModel (not image capable)
      await expect(adapter.generateImage(imageGenChat as Chat)).rejects.toThrow(
        "This model is not capable of image generation."
      );
    });

    it("should throw an error if the API response is missing image data", async () => {
      const imageAdapter = new OpenAIAdapter(mockConfig, mockDalleModel);
      const { mockImagesGenerate } = getMockOpenAIClient();
      // Mock response with missing data
      mockImagesGenerate.mockResolvedValue({
        created: Date.now() / 1000,
        data: [],
      });

      await expect(
        imageAdapter.generateImage(imageGenChat as Chat)
      ).rejects.toThrow("No image URL received from OpenAI");

      // Mock response with missing url
      mockImagesGenerate.mockResolvedValue({
        created: Date.now() / 1000,
        data: [{ url: undefined as any }],
      });
      await expect(
        imageAdapter.generateImage(imageGenChat as Chat)
      ).rejects.toThrow("No image URL received from OpenAI");
    });

    it("should handle API errors during image generation", async () => {
      const imageAdapter = new OpenAIAdapter(mockConfig, mockDalleModel);
      const { mockImagesGenerate } = getMockOpenAIClient();
      const apiError = new Error("OpenAI API Error");
      mockImagesGenerate.mockRejectedValue(apiError);

      await expect(
        imageAdapter.generateImage(imageGenChat as Chat)
      ).rejects.toThrow(apiError);
    });
  });

  // --- sendMCPChat Tests ---
  describe("sendMCPChat", () => {
    it("should throw NotImplementedError", async () => {
      const dummyChat: Partial<Chat> = { prompt: "test" };
      const dummyTool: any = { id: 1, name: "dummy", path: "" };
      await expect(
        adapter.sendMCPChat(dummyChat as Chat, dummyTool)
      ).rejects.toThrow(
        // Match the exact error message from the adapter
        "MCP tools require specific formatting not yet implemented for OpenAI adapter using client.responses.create."
      );
    });
  });
});
