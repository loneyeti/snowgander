import OpenAI from "openai";
import { OpenAIImageAdapter } from "../openai-image";
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  NotImplementedError, // Keep for checking specific error messages if needed
  ImageGenerationResponse,
  Chat,
  ChatResponse,
  AIResponse, // Added for generateResponse/editImage return type
  ErrorBlock, // Added for checking error content
} from "../../types";

// Mock the OpenAI SDK
jest.mock("openai", () => {
  const mockImages = {
    generate: jest.fn(),
    edit: jest.fn(),
  };
  return jest.fn().mockImplementation(() => ({
    images: mockImages,
  }));
});

// Helper to create expected error response structure
const createExpectedErrorResponse = (
  publicMsg: string,
  privateMsg: string,
  role: "assistant" | "error" = "error"
): Partial<AIResponse | ChatResponse> => ({
  role: role,
  content: [
    expect.objectContaining<Partial<ErrorBlock>>({
      type: "error",
      publicMessage: publicMsg,
      privateMessage: expect.stringContaining(privateMsg), // Use stringContaining for flexibility
    }),
  ],
  usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
});

describe("OpenAIImageAdapter", () => {
  let mockClient: OpenAI;
  let vendorConfig: VendorConfig;
  let modelConfig: ModelConfig;
  let adapter: OpenAIImageAdapter;

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    mockClient = new OpenAI({ apiKey: "test-key" }); // Instantiated mock

    vendorConfig = { apiKey: "test-key" };
    modelConfig = {
      apiName: "gpt-image-1", // Example model name
      isVision: false,
      isImageGeneration: true,
      isThinking: false,
      inputTokenCost: 0.01 / 1000000, // Example costs
      outputTokenCost: 0.03 / 1000000,
    };
    adapter = new OpenAIImageAdapter(vendorConfig, modelConfig);
  });

  it("should throw error if API key is missing in constructor", () => {
    expect(() => new OpenAIImageAdapter({ apiKey: "" }, modelConfig)).toThrow(
      "openai-image API key is required."
    );
  });

  describe("generateImage", () => {
    // ... (successful generateImage test remains the same) ...
    it("should call OpenAI images.generate with correct parameters and return mapped response", async () => {
      const mockApiResponse = {
        created: Date.now() / 1000,
        data: [
          {
            b64_json: "base64encodedstring1",
            revised_prompt: "revised prompt 1",
          },
          {
            b64_json: "base64encodedstring2",
            revised_prompt: "revised prompt 2",
          },
        ],
      };
      // Setup the mock implementation for generate
      (mockClient.images.generate as jest.Mock).mockResolvedValue(
        mockApiResponse
      );

      const options: AIRequestOptions = {
        model: "gpt-image-1", // Can be overridden by openaiImageGenerationOptions
        messages: [], // Not directly used, prompt extracted
        prompt: "a cat wearing a hat",
        openaiImageGenerationOptions: {
          n: 2,
          size: "1024x1024",
          quality: "hd",
          response_format: "b64_json", // Adapter forces this
          style: "vivid",
          user: "test-user",
        },
      };

      // Expected structure for successful response (AIResponse)
      const expectedResponse: Partial<AIResponse> = {
        role: "assistant",
        content: [
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "base64encodedstring1",
          },
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "base64encodedstring2",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 }, // Placeholder usage
      };

      const response = await adapter.generateImage(options);

      expect(mockClient.images.generate).toHaveBeenCalledTimes(1);
      expect(mockClient.images.generate).toHaveBeenCalledWith({
        model: "gpt-image-1",
        prompt: "a cat wearing a hat",
        n: 2,
        quality: "hd",
        response_format: "b64_json",
        size: "1024x1024",
        style: "vivid",
        user: "test-user",
      });
      // Use objectContaining because the actual response might have more fields
      expect(response).toMatchObject(expectedResponse);
    });

    it("should extract prompt from last user message if options.prompt is missing", async () => {
      const mockApiResponse = { data: [{ b64_json: "test" }] };
      (mockClient.images.generate as jest.Mock).mockResolvedValue(
        mockApiResponse
      );

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [
          { role: "user", content: [{ type: "text", text: "first message" }] },
          { role: "assistant", content: [{ type: "text", text: "response" }] },
          {
            role: "user",
            content: [{ type: "text", text: "last user prompt" }],
          },
        ],
        // No explicit prompt here
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      await adapter.generateImage(options);

      expect(mockClient.images.generate).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: "last user prompt" })
      );
    });

    it("should return an ErrorBlock if no prompt is available", async () => {
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [], // No messages
        // No prompt
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      const expectedError = createExpectedErrorResponse(
        "Prompt missing.",
        "A text prompt is required for image generation."
      );
      const response = await adapter.generateImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return an ErrorBlock if openaiImageGenerationOptions are missing", async () => {
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [],
        prompt: "test prompt",
        // Missing openaiImageGenerationOptions
      };

      const expectedError = createExpectedErrorResponse(
        "Image options missing.",
        "openaiImageGenerationOptions are required"
      );
      const response = await adapter.generateImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return an ErrorBlock if API call fails", async () => {
      const apiError = new Error("OpenAI API Error");
      (mockClient.images.generate as jest.Mock).mockRejectedValue(apiError);

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        prompt: "test",
        messages: [],
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      const expectedError = createExpectedErrorResponse(
        "Image generation failed.",
        `Failed to generate image: ${apiError.message}`
      );
      const response = await adapter.generateImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return an ErrorBlock if API returns success but no image data", async () => {
      const mockApiResponse = {
        created: Date.now() / 1000,
        data: [], // Empty data array
      };
      (mockClient.images.generate as jest.Mock).mockResolvedValue(
        mockApiResponse
      );

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        prompt: "a cat wearing a hat",
        messages: [],
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      const expectedError = createExpectedErrorResponse(
        "Failed to retrieve image data.",
        "OpenAI API returned success but no image data."
      );
      const response = await adapter.generateImage(options);
      expect(response).toMatchObject(expectedError);
    });
  });

  describe("sendChat", () => {
    let baseChat: Chat;
    let mockGeneratedImageResponse: AIResponse;

    beforeEach(() => {
      baseChat = {
        model: "gpt-image-1",
        responseHistory: [
          {
            role: "user",
            content: [{ type: "text", text: "previous prompt" }],
          },
        ],
        prompt: "generate a dog image",
        visionUrl: null,
        imageURL: null,
        maxTokens: null,
        budgetTokens: null,
        systemPrompt: "You are an image generator.",
      };

      mockGeneratedImageResponse = {
        role: "assistant",
        content: [
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "generated-image-data",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
      };

      // Mock the internal methods on the adapter instance
      jest
        .spyOn(adapter, "generateImage")
        .mockResolvedValue(mockGeneratedImageResponse);
      // Mock editImage to return its specific ErrorBlock
      jest
        .spyOn(adapter, "editImage")
        .mockResolvedValue(
          createExpectedErrorResponse(
            "Image editing is not fully supported yet.",
            "editImage not fully implemented"
          ) as AIResponse
        );
    });

    it("should call generateImage and return ChatResponse when generation options provided", async () => {
      const chatWithOptions: Chat = {
        ...baseChat,
        openaiImageGenerationOptions: { n: 1, size: "512x512" },
      };

      // Expected ChatResponse structure matching the mocked generateImage result
      const expectedResponse: ChatResponse = {
        role: "assistant",
        content: mockGeneratedImageResponse.content,
        usage: mockGeneratedImageResponse.usage,
      };

      const response = await adapter.sendChat(chatWithOptions);

      expect(adapter.generateImage).toHaveBeenCalledTimes(1);
      // ... (keep the check for arguments passed to generateImage) ...
      expect(response).toEqual(expectedResponse);
      expect(adapter.editImage).not.toHaveBeenCalled();
    });

    it("should call editImage and return ChatResponse (with ErrorBlock) when edit options provided", async () => {
      const chatWithOptions: Chat = {
        ...baseChat,
        prompt: "edit the dog image",
        openaiImageEditOptions: {
          image: [{ type: "image", url: "http://example.com/image.png" }],
          size: "512x512",
        },
      };

      // Expect the ErrorBlock response from the mocked editImage
      const expectedError = createExpectedErrorResponse(
        "Image editing is not fully supported yet.",
        "editImage not fully implemented"
      );

      const response = await adapter.sendChat(chatWithOptions);

      expect(adapter.editImage).toHaveBeenCalledTimes(1);
      // ... (keep the check for arguments passed to editImage) ...
      expect(response).toMatchObject(expectedError);
      expect(adapter.generateImage).not.toHaveBeenCalled();
    });

    it("should return ErrorBlock if neither generation nor edit options are provided", async () => {
      const expectedError = createExpectedErrorResponse(
        "Image generation/editing options missing.",
        "adapter requires either openaiImageGenerationOptions or openaiImageEditOptions"
      );

      const response = await adapter.sendChat(baseChat); // baseChat has no options

      expect(response).toMatchObject(expectedError);
      expect(adapter.generateImage).not.toHaveBeenCalled();
      expect(adapter.editImage).not.toHaveBeenCalled();
    });

    it("should return ErrorBlock if generateImage itself returns an error", async () => {
      const generateImageError = createExpectedErrorResponse(
        "Image generation failed.",
        "API Error"
      ) as AIResponse;
      (adapter.generateImage as jest.Mock).mockResolvedValue(
        generateImageError
      );

      const chatWithOptions: Chat = {
        ...baseChat,
        openaiImageGenerationOptions: { n: 1, size: "512x512" },
      };

      const response = await adapter.sendChat(chatWithOptions);
      expect(response).toMatchObject(generateImageError); // Should propagate the error response
    });
  });

  describe("Other Methods", () => {
    it("generateResponse should return ErrorBlock", async () => {
      const options: AIRequestOptions = { model: "test", messages: [] };
      const expectedError = createExpectedErrorResponse(
        "This model cannot generate text responses.",
        "adapter does not support generateResponse"
      );
      const response = await adapter.generateResponse(options);
      expect(response).toMatchObject(expectedError);
    });

    it("editImage should return ErrorBlock for missing options", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        prompt: "edit test",
        // Missing openaiImageEditOptions
      };
      const expectedError = createExpectedErrorResponse(
        "Image editing options missing.",
        "openaiImageEditOptions are required"
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("editImage should return ErrorBlock for missing prompt", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        // Missing prompt
        openaiImageEditOptions: { image: [{ type: "image", url: "test.png" }] },
      };
      const expectedError = createExpectedErrorResponse(
        "Prompt missing for editing.",
        "A text prompt is required for image editing."
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("editImage should return ErrorBlock for missing image", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        prompt: "edit test",
        openaiImageEditOptions: { image: [] }, // Empty image array
      };
      const expectedError = createExpectedErrorResponse(
        "Input image missing for editing.",
        "An input image is required for editing."
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("editImage should return ErrorBlock for not implemented image handling", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        prompt: "edit test",
        openaiImageEditOptions: { image: [{ type: "image", url: "test.png" }] },
      };
      const expectedError = createExpectedErrorResponse(
        "Image editing is not fully supported yet.",
        "editImage not fully implemented yet (image input handling needed)"
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("sendMCPChat should return ErrorBlock", async () => {
      const chat: Chat = {
        model: "test",
        responseHistory: [],
        prompt: "hi",
        visionUrl: null,
        imageURL: null,
        maxTokens: null,
        budgetTokens: null,
      };
      const expectedError = createExpectedErrorResponse(
        "Tool use is not supported by this model.",
        "adapter does not support sendMCPChat"
      );
      // sendMCPChat is optional, but we implemented it to return an error
      const response = await adapter.sendMCPChat(chat, []);
      expect(response).toMatchObject(expectedError);
    });
  });
});
