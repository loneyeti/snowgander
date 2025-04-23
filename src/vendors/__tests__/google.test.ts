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
  NotImplementedError, // Added missing import
} from "../../types";
import {
  GoogleGenerativeAI, // Import the original class
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";
import { computeResponseCost } from "../../utils";

// --- Define Mocks FIRST ---
const mockGenerateContent = jest.fn();
const mockSendMessage = jest.fn();
const mockStartChat = jest
  .fn()
  .mockReturnValue({ sendMessage: mockSendMessage });
const mockGetGenerativeModel = jest.fn().mockReturnValue({
  generateContent: mockGenerateContent,
  startChat: mockStartChat,
});
// IMPORTANT: Define the constructor mock implementation referencing the others
const mockGoogleAIConstructor = jest.fn().mockImplementation(() => ({
  getGenerativeModel: mockGetGenerativeModel,
}));

// --- Mock the SDK using jest.doMock (not hoisted) ---
jest.doMock("@google/generative-ai", () => ({
  GoogleGenerativeAI: mockGoogleAIConstructor,
  HarmCategory: {
    HARM_CATEGORY_HARASSMENT: "HARM_CATEGORY_HARASSMENT",
    HARM_CATEGORY_HATE_SPEECH: "HARM_CATEGORY_HATE_SPEECH",
    HARM_CATEGORY_SEXUALLY_EXPLICIT: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    HARM_CATEGORY_DANGEROUS_CONTENT: "HARM_CATEGORY_DANGEROUS_CONTENT",
  },
  HarmBlockThreshold: {
    BLOCK_NONE: "BLOCK_NONE",
  },
}));

// --- Get Typed Mock Constructor AFTER jest.doMock ---
const MockedGoogleGenerativeAI = GoogleGenerativeAI as jest.MockedClass<
  typeof GoogleGenerativeAI
>;

describe("GoogleAIAdapter", () => {
  let GoogleAIAdapter: typeof import("../google").GoogleAIAdapter;
  let adapter: import("../google").GoogleAIAdapter;

  const mockConfig: VendorConfig = { apiKey: "test-google-key" };
  const mockGeminiProVision: ModelConfig = {
    apiName: "gemini-pro-vision",
    isVision: true,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 0.125 / 1_000_000,
    outputTokenCost: 0.375 / 1_000_000,
  };
  const mockGeminiPro: ModelConfig = {
    apiName: "gemini-pro",
    isVision: false,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 0.125 / 1_000_000,
    outputTokenCost: 0.375 / 1_000_000,
  };

  beforeAll(async () => {
    const module = await import("../google");
    GoogleAIAdapter = module.GoogleAIAdapter;
  });

  beforeEach(() => {
    mockGoogleAIConstructor.mockClear();
    mockGetGenerativeModel.mockClear();
    mockGenerateContent.mockClear();
    mockStartChat.mockClear();
    mockSendMessage.mockClear();

    mockGetGenerativeModel.mockReturnValue({
      generateContent: mockGenerateContent,
      startChat: mockStartChat,
    });

    adapter = new GoogleAIAdapter(mockConfig, mockGeminiProVision);
  });

  it("should initialize GoogleAI client with correct config", () => {
    expect(mockGoogleAIConstructor).toHaveBeenCalledTimes(1);
  });

  it("should reflect capabilities from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(true);
    expect(adapter.isImageGenerationCapable).toBe(false);
    expect(adapter.isThinkingCapable).toBe(false);
    expect(adapter.inputTokenCost).toBe(mockGeminiProVision.inputTokenCost);
    expect(adapter.outputTokenCost).toBe(mockGeminiProVision.outputTokenCost);
  });

  // --- generateResponse Tests ---
  describe("generateResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello Gemini!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockGeminiProVision.apiName,
      messages: basicMessages,
    };

    it("should call Google generateContent with correct basic parameters", async () => {
      const mockApiResponse = {
        response: {
          text: () => "Hi there!",
          candidates: [
            { content: { role: "model", parts: [{ text: "Hi there!" }] } },
          ],
        },
      };
      mockGenerateContent.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(basicOptions);

      expect(mockGenerateContent).toHaveBeenCalledTimes(1);
      expect(mockGenerateContent).toHaveBeenCalledWith({
        contents: [{ role: "user", parts: [{ text: "Hello Gemini!" }] }],
      });
    });

    it("should map Google response to AIResponse format", async () => {
      const mockApiResponse = {
        response: {
          text: () => "Response from Gemini.",
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "Response from Gemini." }],
              },
            },
          ],
        },
      };
      mockGenerateContent.mockResolvedValue(mockApiResponse);

      const response = await adapter.generateResponse(basicOptions);

      expect(response).toEqual<AIResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Response from Gemini." }],
        usage: undefined,
      });
    });

    it("should map ImageDataBlock to Google Part format", async () => {
      const imageData: ImageDataBlock = {
        type: "image_data",
        mimeType: "image/png",
        base64Data: "base64pngstring",
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [imageData, { type: "text", text: "What is this PNG?" }],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
      };
      const mockApiResponse = {
        response: {
          text: () => "A PNG.",
          candidates: [
            { content: { role: "model", parts: [{ text: "A PNG." }] } },
          ],
        },
      };
      mockGenerateContent.mockResolvedValue(mockApiResponse);

      await adapter.generateResponse(optionsWithImage);

      expect(mockGenerateContent).toHaveBeenCalledWith({
        contents: [
          {
            role: "user",
            parts: [
              {
                inlineData: { mimeType: "image/png", data: "base64pngstring" },
              },
              { text: "What is this PNG?" },
            ],
          },
        ],
      });
    });

    it("should warn and skip ImageBlock (URL)", async () => {
      const imageBlock: ImageBlock = {
        type: "image",
        url: "http://example.com/image.gif",
      };
      const messagesWithImage: Message[] = [
        {
          role: "user",
          content: [imageBlock, { type: "text", text: "What about this GIF?" }],
        },
      ];
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        messages: messagesWithImage,
      };
      const mockApiResponse = {
        response: {
          text: () => "A GIF URL.",
          candidates: [
            { content: { role: "model", parts: [{ text: "A GIF URL." }] } },
          ],
        },
      };
      mockGenerateContent.mockResolvedValue(mockApiResponse);
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation();

      await adapter.generateResponse(optionsWithImage);

      expect(mockGenerateContent).toHaveBeenCalledWith({
        contents: [
          {
            role: "user",
            parts: [{ text: "What about this GIF?" }],
          },
        ],
      });
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Google adapter received 'image' block (URL), needs pre-processing to base64 inlineData."
        )
      );
      consoleWarnSpy.mockRestore();
    });
  });

  // --- sendChat Tests ---
  describe("sendChat", () => {
    // Placeholder for future tests
  });

  // --- generateImage Tests ---
  describe("generateImage", () => {
    it("should throw NotImplementedError with correct message", async () => {
      // Use AIRequestOptions as required by the updated interface
      const dummyOptions: AIRequestOptions = { model: 'test', messages: [] };
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(NotImplementedError);
      // Check the specific error message from the adapter
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        "Google generates images inline with text, use generateResponse/sendChat."
      );
    });
  });

  // Removed sendMCPChat tests as the method was removed
});
