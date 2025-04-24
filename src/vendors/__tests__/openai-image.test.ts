import OpenAI from "openai";
import { OpenAIImageAdapter } from "../openai-image";
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  NotImplementedError,
  ImageGenerationResponse,
  Chat,
  ChatResponse, // Added missing import
} from "../../types";

// Mock the OpenAI SDK
jest.mock("openai", () => {
  const mockImages = {
    generate: jest.fn(),
    edit: jest.fn(), // Mock edit even if not implemented yet
  };
  return jest.fn().mockImplementation(() => ({
    images: mockImages,
  }));
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

  it("should throw error if API key is missing", () => {
    expect(() => new OpenAIImageAdapter({ apiKey: "" }, modelConfig)).toThrow(
      "openai-image API key is required."
    );
  });

  describe("generateImage", () => {
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

      const expectedResponse: ImageGenerationResponse = {
        images: [
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
      expect(response).toEqual(expectedResponse);
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

    it("should throw error if no prompt is available", async () => {
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [], // No messages
        // No prompt
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        "A text prompt is required for image generation."
      );
    });

    it("should throw error if openaiImageGenerationOptions are missing", async () => {
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [],
        prompt: "test prompt",
        // Missing openaiImageGenerationOptions
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        "openaiImageGenerationOptions are required for OpenAI image generation."
      );
    });

    it("should handle API errors gracefully", async () => {
      const apiError = new Error("OpenAI API Error");
      (mockClient.images.generate as jest.Mock).mockRejectedValue(apiError);

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        prompt: "test",
        messages: [],
        openaiImageGenerationOptions: { size: "1024x1024" },
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        `Failed to generate image: ${apiError.message}`
      );
    });
  });

  describe("sendChat", () => {
    let baseChat: Chat;

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

      // Mock the internal methods since sendChat calls them
      // We need to mock the methods on the *instance* of the adapter
      jest.spyOn(adapter, "generateImage").mockResolvedValue({
        images: [
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "generated-image-data",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
      });
      // Mock editImage to throw its specific NotImplementedError for now
      jest
        .spyOn(adapter, "editImage")
        .mockRejectedValue(
          new NotImplementedError(
            "openai-image adapter editImage not fully implemented yet (image input handling needed)."
          )
        );
    });

    it("should call generateImage when openaiImageGenerationOptions are provided", async () => {
      const chatWithOptions: Chat = {
        ...baseChat,
        openaiImageGenerationOptions: {
          n: 1,
          size: "512x512",
        },
      };

      const expectedResponse: ChatResponse = {
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

      const response = await adapter.sendChat(chatWithOptions);

      expect(adapter.generateImage).toHaveBeenCalledTimes(1);
      expect(adapter.generateImage).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-image-1",
          prompt: "generate a dog image",
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "previous prompt" }],
            },
            {
              role: "user",
              content: [{ type: "text", text: "generate a dog image" }],
            },
          ],
          systemPrompt: "You are an image generator.",
          openaiImageGenerationOptions: { n: 1, size: "512x512" },
        })
      );
      expect(response).toEqual(expectedResponse);
      expect(adapter.editImage).not.toHaveBeenCalled();
    });

    it("should call editImage when openaiImageEditOptions are provided", async () => {
      const chatWithOptions: Chat = {
        ...baseChat,
        prompt: "edit the dog image", // Change prompt for clarity
        openaiImageEditOptions: {
          image: [{ type: "image", url: "http://example.com/image.png" }], // Placeholder input
          size: "512x512",
        },
      };

      // Expect the NotImplementedError from the mocked editImage
      await expect(adapter.sendChat(chatWithOptions)).rejects.toThrow(
        NotImplementedError
      );
      await expect(adapter.sendChat(chatWithOptions)).rejects.toThrow(
        "openai-image adapter editImage not fully implemented yet (image input handling needed)."
      );

      expect(adapter.editImage).toHaveBeenCalledTimes(2); // Called twice due to expect().rejects structure
      expect(adapter.editImage).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-image-1",
          prompt: "edit the dog image",
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "previous prompt" }],
            },
            {
              role: "user",
              content: [{ type: "text", text: "edit the dog image" }],
            },
          ],
          systemPrompt: "You are an image generator.",
          openaiImageEditOptions: {
            image: [{ type: "image", url: "http://example.com/image.png" }],
            size: "512x512",
          },
        })
      );
      expect(adapter.generateImage).not.toHaveBeenCalled();
    });

    it("should throw error if neither generation nor edit options are provided", async () => {
      // baseChat has no image options
      await expect(adapter.sendChat(baseChat)).rejects.toThrow(Error);
      await expect(adapter.sendChat(baseChat)).rejects.toThrow(
        "openai-image adapter requires either openaiImageGenerationOptions or openaiImageEditOptions to be provided in the Chat object via sendChat."
      );
      expect(adapter.generateImage).not.toHaveBeenCalled();
      expect(adapter.editImage).not.toHaveBeenCalled();
    });
  });

  // Test other unimplemented/unchanged methods
  describe("Other Methods", () => {
    it("generateResponse should throw NotImplementedError", async () => {
      const options: AIRequestOptions = { model: "test", messages: [] };
      await expect(adapter.generateResponse(options)).rejects.toThrow(
        NotImplementedError
      );
      await expect(adapter.generateResponse(options)).rejects.toThrow(
        "openai-image adapter does not support generateResponse. Use generateImage or editImage."
      );
    });

    // editImage is tested indirectly via sendChat now, but we can keep a direct test
    // for its specific NotImplementedError regarding image input handling.
    it("editImage should throw NotImplementedError for image input handling", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        prompt: "edit test",
        openaiImageEditOptions: { image: [{ type: "image", url: "test.png" }] }, // Provide minimal required options
      };
      // Note: This calls the *real* editImage, not the mock set up in sendChat tests
      const freshAdapter = new OpenAIImageAdapter(vendorConfig, modelConfig);
      await expect(freshAdapter.editImage(options)).rejects.toThrow(
        NotImplementedError
      );
      await expect(freshAdapter.editImage(options)).rejects.toThrow(
        "openai-image adapter editImage not fully implemented yet (image input handling needed)."
      );
    });

    it("sendMCPChat should throw NotImplementedError", async () => {
      const chat: Chat = {
        model: "test",
        responseHistory: [],
        prompt: "hi",
        visionUrl: null,
        imageURL: null,
        maxTokens: null,
        budgetTokens: null,
      };
      // sendMCPChat is optional, so need to check if it exists before calling
      if (adapter.sendMCPChat) {
        await expect(adapter.sendMCPChat(chat, [])).rejects.toThrow(
          NotImplementedError
        );
        await expect(adapter.sendMCPChat(chat, [])).rejects.toThrow(
          "openai-image adapter does not support sendMCPChat."
        );
      } else {
        // If sendMCPChat is not even defined on the adapter, the test passes implicitly.
        // Or, explicitly pass:
        expect(adapter.sendMCPChat).toBeUndefined();
      }
    });

    // Note: The original comment about sendMCPChat being optional is kept,
    // but the test itself correctly handles the case where it might exist.
  });
});
