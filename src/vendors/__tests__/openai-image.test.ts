import OpenAI, { toFile } from "openai"; // Import toFile for mocking
import fetch from "node-fetch"; // Import fetch for mocking
const { Response } = jest.requireActual("node-fetch"); // Get Response for mocking fetch
import { OpenAIImageAdapter } from "../openai-image";
import {
  VendorConfig,
  ImageBlock, // Added for edit tests
  ImageDataBlock, // Added for edit tests
  ModelConfig,
  AIRequestOptions,
  Chat,
  ChatResponse,
  AIResponse, // Added for generateResponse/editImage return type
  ErrorBlock,
} from "../../types";

// --- Mocks ---
// Mock node-fetch
jest.mock("node-fetch");
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock OpenAI SDK and toFile
const mockImagesGenerate = jest.fn();
const mockImagesEdit = jest.fn();
// We will define the implementation inside the mock factory

jest.mock("openai", () => {
  // Define the mock implementation for toFile *inside* the factory
  const mockToFileImplementation = jest.fn(
    async (streamOrBlobOrBuffer, filename) => {
      // Simulate creating a File-like object for testing purposes
      console.log(`Mock toFile called with filename: ${filename}`);
      return {
        name: filename || "mocked_file.png",
      } as any;
    }
  );

  // Mock the default export (the constructor)
  const mockConstructor = jest.fn().mockImplementation(() => ({
    images: {
      generate: mockImagesGenerate,
      edit: mockImagesEdit,
    },
  }));

  // Return an object simulating the module's exports
  return {
    __esModule: true, // Indicate this is an ES module mock
    default: mockConstructor, // Mock the default export
    OpenAI: mockConstructor, // Mock the named export if used like `new OpenAI()`
    toFile: mockToFileImplementation, // Mock the named export `toFile`
  };
});

// Now get the mocked toFile after jest.mock has run
const mockToFile = toFile as jest.MockedFunction<typeof toFile>;
// --- End Mocks ---

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
  let mockGeneratedImage: OpenAI.Images.Image;
  let mockEditedImage: OpenAI.Images.Image;

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    mockFetch.mockClear(); // Clear fetch mock specifically
    mockImagesGenerate.mockClear();
    mockImagesEdit.mockClear();
    mockToFile.mockClear(); // Clear toFile mock

    mockClient = new OpenAI({ apiKey: "test-key" }); // Instantiated mock

    vendorConfig = { apiKey: "test-key" };
    modelConfig = {
      apiName: "gpt-image-1",
      isVision: false, // Image models are not vision models in this context
      isImageGeneration: true,
      isThinking: false,
      // Costs would depend on the specific model (gpt-image-1 pricing)
      inputTokenCost: 0, // Placeholder
      outputTokenCost: 0, // Placeholder - based on output tokens per image size/quality
    };
    adapter = new OpenAIImageAdapter(vendorConfig, modelConfig);

    // Common mock image data for responses
    mockGeneratedImage = {
      b64_json: "base64encodedstring_generated",
      revised_prompt: "revised prompt generated",
    };
    mockEditedImage = {
      b64_json: "base64encodedstring_edited",
      revised_prompt: "revised prompt edited",
    };
  });

  it("should throw error if API key is missing in constructor", () => {
    expect(() => new OpenAIImageAdapter({ apiKey: "" }, modelConfig)).toThrow(
      "openai-image API key is required."
    );
  });

  describe("generateImage", () => {
    it("should call OpenAI images.generate with correct parameters and return mapped response", async () => {
      const mockApiResponse = {
        created: Date.now() / 1000,
        data: [mockGeneratedImage],
      };
      mockImagesGenerate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [],
        prompt: "a futuristic cityscape",
        openaiImageGenerationOptions: {
          n: 1,
          size: "1024x1024",
          quality: "high",
          background: "transparent",
          output_compression: 80,
          user: "test-user-gen",
          moderation: "low",
        },
      };

      const expectedResponse: Partial<AIResponse> = {
        role: "assistant",
        content: [
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "base64encodedstring_generated",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 }, // Placeholder
      };

      const response = await adapter.generateImage(options);

      expect(mockImagesGenerate).toHaveBeenCalledTimes(1);
      expect(mockImagesGenerate).toHaveBeenCalledWith({
        model: "gpt-image-1",
        prompt: "a futuristic cityscape",
        n: 1,
        quality: "high",
        //response_format: "b64_json",
        size: "1024x1024",
        background: "transparent",
        output_compression: 80,
        user: "test-user-gen",
        moderation: "low",
      });
      expect(response).toMatchObject(expectedResponse);
    });

    it("should handle 'auto' options correctly by omitting them", async () => {
      const mockApiResponse = { data: [mockGeneratedImage] };
      mockImagesGenerate.mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [],
        prompt: "another image",
        openaiImageGenerationOptions: {
          size: "auto",
          quality: "auto",
          background: "auto",
        },
      };

      await adapter.generateImage(options);

      expect(mockImagesGenerate).toHaveBeenCalledTimes(1);
      // Check that size, quality, background are NOT in the called options
      expect(mockImagesGenerate).toHaveBeenCalledWith({
        model: "gpt-image-1",
        prompt: "another image",
        //response_format: "b64_json",
        // n defaults to 1 if not provided
        // size: undefined, // Omitted
        // quality: undefined, // Omitted
        // background: undefined, // Omitted
      });
    });

    it("should extract prompt from last user message if options.prompt is missing", async () => {
      mockImagesGenerate.mockResolvedValue({ data: [mockGeneratedImage] });
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [
          { role: "user", content: [{ type: "text", text: "first message" }] },
          { role: "assistant", content: [{ type: "text", text: "response" }] },
          {
            role: "user",
            content: [{ type: "text", text: "last user prompt for image" }],
          },
        ],
        openaiImageGenerationOptions: { size: "1024x1024" },
      };
      await adapter.generateImage(options);
      expect(mockImagesGenerate).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: "last user prompt for image" })
      );
    });

    it("should return an ErrorBlock if no prompt is available", async () => {
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        messages: [],
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
      };
      const expectedError = createExpectedErrorResponse(
        "Image options missing.",
        "openaiImageGenerationOptions are required"
      );
      const response = await adapter.generateImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return an ErrorBlock if API call fails", async () => {
      const apiError = new Error("Gen API Error");
      mockImagesGenerate.mockRejectedValue(apiError);
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
      mockImagesGenerate.mockResolvedValue({ data: [] }); // Empty data
      const options: AIRequestOptions = {
        model: "gpt-image-1",
        prompt: "a cat",
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

  // --- editImage Tests ---
  describe("editImage", () => {
    let editOptions: AIRequestOptions;
    let inputImageBlockUrl: ImageBlock;
    let inputImageBlockData: ImageDataBlock;
    let maskBlockData: ImageDataBlock;

    beforeEach(() => {
      inputImageBlockUrl = {
        type: "image",
        url: "http://example.com/image.png",
      };
      inputImageBlockData = {
        type: "image_data",
        mimeType: "image/jpeg",
        base64Data: "base64_image_data",
      };
      maskBlockData = {
        type: "image_data",
        mimeType: "image/png",
        base64Data: "base64_mask_data",
      };

      editOptions = {
        model: "gpt-image-1",
        messages: [],
        prompt: "add a hat to the person",
        openaiImageEditOptions: {
          image: [inputImageBlockUrl], // Start with one URL image
          size: "1024x1024", // Use valid size
          user: "test-user-edit",
          // moderation: "auto", // Removed from call, but keep in type def for now
        },
      };

      // Mock fetch response
      mockFetch.mockResolvedValue(
        new Response(Buffer.from("fetched-image-data"), { status: 200 })
      );
      // Mock API response
      mockImagesEdit.mockResolvedValue({ data: [mockEditedImage] });
    });

    it("should call OpenAI images.edit with correct parameters (URL image) and return mapped response", async () => {
      const expectedResponse: Partial<AIResponse> = {
        role: "assistant",
        content: [
          {
            type: "image_data",
            mimeType: "image/png", // Adapter assumes PNG for now
            base64Data: "base64encodedstring_edited",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 }, // Placeholder
      };

      const response = await adapter.editImage(editOptions);

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(mockFetch).toHaveBeenCalledWith(inputImageBlockUrl.url);
      expect(mockToFile).toHaveBeenCalledTimes(1); // Called once for the input image
      expect(mockImagesEdit).toHaveBeenCalledTimes(1);
      expect(mockImagesEdit).toHaveBeenCalledWith({
        model: "gpt-image-1",
        prompt: "add a hat to the person",
        image: [expect.objectContaining({ name: "image.png" })], // Check mocked file
        //response_format: "b64_json",
        size: "1024x1024", // Check casted size
        user: "test-user-edit",
        // moderation removed from call
        // mask: undefined (not provided in this test)
        // n: undefined (defaults to 1)
      });
      expect(response).toMatchObject(expectedResponse);
    });

    it("should call OpenAI images.edit with base64 image and mask", async () => {
      editOptions.openaiImageEditOptions!.image = [inputImageBlockData];
      editOptions.openaiImageEditOptions!.mask = maskBlockData;

      await adapter.editImage(editOptions);

      expect(mockFetch).not.toHaveBeenCalled(); // No URL fetching
      expect(mockToFile).toHaveBeenCalledTimes(2); // Called for image and mask
      expect(mockImagesEdit).toHaveBeenCalledTimes(1);
      expect(mockImagesEdit).toHaveBeenCalledWith(
        expect.objectContaining({
          image: [expect.objectContaining({ name: "image.jpeg" })], // Check mocked file from base64
          mask: expect.objectContaining({ name: "image.png" }), // Check mocked mask file
        })
      );
    });

    it("should call OpenAI images.edit with multiple input images", async () => {
      editOptions.openaiImageEditOptions!.image = [
        inputImageBlockUrl,
        inputImageBlockData,
      ];

      await adapter.editImage(editOptions);

      expect(mockFetch).toHaveBeenCalledTimes(1); // Fetch the URL image
      expect(mockToFile).toHaveBeenCalledTimes(2); // Called for URL image and base64 image
      expect(mockImagesEdit).toHaveBeenCalledTimes(1);
      expect(mockImagesEdit).toHaveBeenCalledWith(
        expect.objectContaining({
          image: [
            expect.objectContaining({ name: "image.png" }), // From URL
            expect.objectContaining({ name: "image.jpeg" }), // From base64
          ],
        })
      );
    });

    it("should return ErrorBlock if prepareImageInput fails (e.g., fetch error)", async () => {
      mockFetch.mockResolvedValue(new Response(null, { status: 404 })); // Simulate fetch error

      const expectedError = createExpectedErrorResponse(
        "Image editing failed.",
        "Failed to prepare image input: Failed to fetch image URL: Not Found"
      );
      const response = await adapter.editImage(editOptions);
      expect(response).toMatchObject(expectedError);
      expect(mockImagesEdit).not.toHaveBeenCalled();
    });

    it("should return ErrorBlock if API call fails", async () => {
      const apiError = new Error("Edit API Error");
      mockImagesEdit.mockRejectedValue(apiError);

      const expectedError = createExpectedErrorResponse(
        "Image editing failed.",
        `Failed to edit image: ${apiError.message}`
      );
      const response = await adapter.editImage(editOptions);
      expect(response).toMatchObject(expectedError);
    });

    it("should return ErrorBlock if API returns success but no image data", async () => {
      mockImagesEdit.mockResolvedValue({ data: [] }); // Empty data

      const expectedError = createExpectedErrorResponse(
        "Failed to retrieve edited image.",
        "OpenAI API returned success but no edited image data."
      );
      const response = await adapter.editImage(editOptions);
      expect(response).toMatchObject(expectedError);
    });

    // Add tests for missing prompt, missing image, missing options (similar to generateImage)
    it("should return ErrorBlock for missing options", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        prompt: "edit test",
      };
      const expectedError = createExpectedErrorResponse(
        "Image editing options missing.",
        "openaiImageEditOptions are required"
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return ErrorBlock for missing prompt", async () => {
      const options: AIRequestOptions = {
        model: "test",
        messages: [],
        openaiImageEditOptions: { image: [inputImageBlockData] },
      };
      const expectedError = createExpectedErrorResponse(
        "Prompt missing for editing.",
        "A text prompt is required for image editing."
      );
      const response = await adapter.editImage(options);
      expect(response).toMatchObject(expectedError);
    });

    it("should return ErrorBlock for missing image", async () => {
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
  });
  // --- End editImage Tests ---

  describe("sendChat", () => {
    let baseChat: Chat;
    let mockGeneratedImageResponse: AIResponse;
    let mockEditedImageResponse: AIResponse; // Added for edit path

    beforeEach(() => {
      baseChat = {
        model: "gpt-image-1",
        responseHistory: [],
        prompt: "generate a dog image",
        visionUrl: null,
        imageURL: null,
        maxTokens: null,
        budgetTokens: null,
      };

      // Mock responses for internal calls
      mockGeneratedImageResponse = {
        role: "assistant",
        content: [
          { type: "image_data", mimeType: "image/png", base64Data: "gen_data" },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
      };
      mockEditedImageResponse = {
        role: "assistant",
        content: [
          {
            type: "image_data",
            mimeType: "image/png",
            base64Data: "edit_data",
          },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 },
      };

      // Spy on adapter methods
      jest
        .spyOn(adapter, "generateImage")
        .mockResolvedValue(mockGeneratedImageResponse);
      jest
        .spyOn(adapter, "editImage")
        .mockResolvedValue(mockEditedImageResponse);
    });

    it("should call generateImage and return ChatResponse when generation options provided", async () => {
      const chatWithOptions: Chat = {
        ...baseChat,
        openaiImageGenerationOptions: { n: 1, size: "1024x1024" }, // Use valid size
      };
      const expectedResponse: ChatResponse = {
        role: "assistant",
        content: mockGeneratedImageResponse.content,
        usage: mockGeneratedImageResponse.usage,
      };

      const response = await adapter.sendChat(chatWithOptions);

      expect(adapter.generateImage).toHaveBeenCalledTimes(1);
      expect(adapter.generateImage).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt: baseChat.prompt,
          openaiImageGenerationOptions:
            chatWithOptions.openaiImageGenerationOptions,
        })
      );
      expect(response).toEqual(expectedResponse);
      expect(adapter.editImage).not.toHaveBeenCalled();
    });

    it("should call editImage and return ChatResponse when edit options provided", async () => {
      const inputImage: ImageBlock = { type: "image", url: "img.png" };
      const chatWithOptions: Chat = {
        ...baseChat,
        prompt: "edit the dog image",
        openaiImageEditOptions: { image: [inputImage], size: "1024x1024" }, // Use valid size
      };
      const expectedResponse: ChatResponse = {
        role: "assistant",
        content: mockEditedImageResponse.content,
        usage: mockEditedImageResponse.usage,
      };

      const response = await adapter.sendChat(chatWithOptions);

      expect(adapter.editImage).toHaveBeenCalledTimes(1);
      expect(adapter.editImage).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt: chatWithOptions.prompt,
          openaiImageEditOptions: chatWithOptions.openaiImageEditOptions,
        })
      );
      expect(response).toEqual(expectedResponse);
      expect(adapter.generateImage).not.toHaveBeenCalled();
    });

    it("should return ErrorBlock if neither generation nor edit options are provided", async () => {
      const expectedError = createExpectedErrorResponse(
        "Image generation/editing options missing.",
        "adapter requires either openaiImageGenerationOptions or openaiImageEditOptions"
      );
      const response = await adapter.sendChat(baseChat); // No options
      expect(response).toMatchObject(expectedError);
      expect(adapter.generateImage).not.toHaveBeenCalled();
      expect(adapter.editImage).not.toHaveBeenCalled();
    });

    it("should return ErrorBlock if generateImage itself returns an error", async () => {
      const generateError = createExpectedErrorResponse(
        "Gen Fail",
        "API Err"
      ) as AIResponse;
      (adapter.generateImage as jest.Mock).mockResolvedValue(generateError);
      const chatWithOptions: Chat = {
        ...baseChat,
        openaiImageGenerationOptions: { n: 1, size: "1024x1024" }, // Use valid size
      };
      const response = await adapter.sendChat(chatWithOptions);
      expect(response).toMatchObject(generateError);
    });

    it("should return ErrorBlock if editImage itself returns an error", async () => {
      const editError = createExpectedErrorResponse(
        "Edit Fail",
        "API Err"
      ) as AIResponse;
      (adapter.editImage as jest.Mock).mockResolvedValue(editError);
      const inputImage: ImageBlock = { type: "image", url: "img.png" };
      const chatWithOptions: Chat = {
        ...baseChat,
        prompt: "edit",
        openaiImageEditOptions: { image: [inputImage], size: "1024x1024" }, // Use valid size
      };
      const response = await adapter.sendChat(chatWithOptions);
      expect(response).toMatchObject(editError);
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
      const response = await adapter.sendMCPChat(chat, []);
      expect(response).toMatchObject(expectedError);
    });
  });
});
