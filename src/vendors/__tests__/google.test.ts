import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  Message,
  Chat,
  ImageBlock,
  ContentBlock,
} from "../../types";
import { computeResponseCost, getImageDataFromUrl } from "../../utils";

// --- Define Mocks FIRST ---
const mockGenerateContent = jest.fn();
const mockGenerateContentStream = jest.fn();
const mockGoogleGenAIConstructor = jest.fn().mockImplementation(() => ({
  models: {
    generateContent: mockGenerateContent,
    generateContentStream: mockGenerateContentStream,
  },
}));

// --- Mock the SDK using jest.doMock (not hoisted) ---
jest.doMock("@google/genai", () => ({
  GoogleGenAI: mockGoogleGenAIConstructor,
  // CRITICAL FIX: Mock the enum used in the implementation
  PartMediaResolutionLevel: {
    MEDIA_RESOLUTION_HIGH: "MEDIA_RESOLUTION_HIGH",
  },
}));

// --- Mock the utility function ---
jest.mock("../../utils", () => ({
  ...jest.requireActual("../../utils"),
  getImageDataFromUrl: jest.fn(),
}));

// --- Get Typed Mocks AFTER jest.doMock ---
const MockedGetImageDataFromUrl = getImageDataFromUrl as jest.Mock;

describe("GoogleAIAdapter", () => {
  // Use let for the class to support lazy import after doMock
  let GoogleAIAdapter: typeof import("../google").GoogleAIAdapter;
  let adapter: import("../google").GoogleAIAdapter;

  const mockConfig: VendorConfig = { apiKey: "test-google-key" };
  const mockGeminiProVision: ModelConfig = {
    apiName: "gemini-pro-vision",
    isVision: true,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 0.125,
    outputTokenCost: 0.375,
  };
  const mockGeminiPro: ModelConfig = {
    apiName: "gemini-pro",
    isVision: false,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 0.125,
    outputTokenCost: 0.375,
  };

  beforeAll(async () => {
    // Import the module AFTER jest.doMock
    const module = await import("../google");
    GoogleAIAdapter = module.GoogleAIAdapter;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new GoogleAIAdapter(mockConfig, mockGeminiProVision);
  });

  it("should initialize GoogleGenAI client with correct config", () => {
    expect(mockGoogleGenAIConstructor).toHaveBeenCalledTimes(1);
    expect(mockGoogleGenAIConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
    });
  });

  it("should reflect capabilities from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(true);
  });

  const mockBasicApiResponse = {
    candidates: [
      {
        content: {
          role: "model",
          parts: [{ text: "Hi there!" }],
        },
      },
    ],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    },
  };

  describe("generateResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello Gemini!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockGeminiPro.apiName,
      messages: basicMessages,
    };

    it("should call Google models.generateContent with correct basic parameters", async () => {
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);
      await textAdapter.generateResponse(basicOptions);
      expect(mockGenerateContent).toHaveBeenCalledTimes(1);
      expect(mockGenerateContent).toHaveBeenCalledWith({
        model: mockGeminiPro.apiName,
        contents: [{ role: "user", parts: [{ text: "Hello Gemini!" }] }],
        generationConfig: {},
      });
    });

    // --- Vision Tests ---

    it("should handle ImageBlock by fetching the URL and converting to inlineData", async () => {
      const imageUrl = "http://example.com/image.gif";
      const imageBlock: ImageBlock = { type: "image", url: imageUrl };
      const messagesWithImageUrl: Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "What about this GIF?" }, imageBlock],
        },
      ];
      const optionsWithImageUrl: AIRequestOptions = {
        model: mockGeminiProVision.apiName,
        messages: messagesWithImageUrl,
      };

      const mockImageData = {
        mimeType: "image/gif",
        base64Data: "base64gifstring",
      };
      MockedGetImageDataFromUrl.mockResolvedValue(mockImageData);
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);

      await adapter.generateResponse(optionsWithImageUrl);

      expect(MockedGetImageDataFromUrl).toHaveBeenCalledWith(imageUrl);

      // Expect the call to include the new mediaResolution configuration
      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          contents: [
            {
              role: "user",
              parts: [
                { text: "What about this GIF?" },
                {
                  inlineData: {
                    mimeType: mockImageData.mimeType,
                    data: mockImageData.base64Data,
                  },
                  // Fix: This expectation was missing in the original test
                  mediaResolution: { level: "MEDIA_RESOLUTION_HIGH" },
                },
              ],
            },
          ],
        })
      );
    });

    it("should yield content chunks and a final meta block with usage", async () => {
      async function* mockStreamWithUsage() {
        yield {
          candidates: [{ content: { parts: [{ text: "This " }] } }],
        };
        yield {
          candidates: [{ content: { parts: [{ text: "is a " }] } }],
        };
        yield {
          candidates: [{ content: { parts: [{ text: "test." }] } }],
          usageMetadata: {
            promptTokenCount: 15,
            candidatesTokenCount: 25,
            totalTokenCount: 40,
          },
        };
      }
      mockGenerateContentStream.mockReturnValue(mockStreamWithUsage());
      const testAdapter = new GoogleAIAdapter(mockConfig, mockGeminiProVision);
      const stream = testAdapter.streamResponse({
        ...basicOptions,
        model: mockGeminiProVision.apiName,
      });

      const receivedBlocks: ContentBlock[] = [];
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks).toHaveLength(4);
      expect(receivedBlocks[0]).toEqual({ type: "text", text: "This " });
      expect(receivedBlocks[1]).toEqual({ type: "text", text: "is a " });
      expect(receivedBlocks[2]).toEqual({ type: "text", text: "test." });

      const metaBlock = receivedBlocks[3];
      expect(metaBlock.type).toBe("meta");
      expect((metaBlock as any).responseId).toBeDefined();

      const expectedInputCost = computeResponseCost(
        15,
        testAdapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        25,
        testAdapter.outputTokenCost!
      );
      const expectedTotalCost = expectedInputCost + expectedOutputCost;

      const receivedUsage = (metaBlock as any).usage;
      expect(receivedUsage).toBeDefined();
      expect(receivedUsage.inputCost).toBeCloseTo(expectedInputCost);
      expect(receivedUsage.outputCost).toBeCloseTo(expectedOutputCost);
      expect(receivedUsage.totalCost).toBeCloseTo(expectedTotalCost);
    });
  });

  describe("sendChat", () => {
    const basicChat: Chat = {
      model: mockGeminiPro.apiName,
      responseHistory: [],
      prompt: "Current question?",
      maxTokens: 100,
      systemPrompt: "Be concise.",
      visionUrl: null,
      imageURL: null,
      budgetTokens: null,
    };

    it("should handle chat with visionUrl by creating an ImageBlock", async () => {
      const chatWithVision: Chat = {
        ...basicChat,
        model: mockGeminiProVision.apiName,
        visionUrl: "http://example.com/image.png",
        prompt: "What's in this image?",
      };

      const generateResponseSpy = jest
        .spyOn(adapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "Mocked vision response" }],
        });

      await adapter.sendChat(chatWithVision);

      expect(generateResponseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockGeminiProVision.apiName,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "What's in this image?" },
                { type: "image", url: "http://example.com/image.png" },
              ],
            },
          ],
        })
      );
      generateResponseSpy.mockRestore();
    });
  });

  describe("streamResponse", () => {
    const basicOptions: AIRequestOptions = {
      model: mockGeminiPro.apiName,
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Stream me a story" }],
        },
      ],
    };

    it("should handle ImageBlock correctly in streaming", async () => {
      const imageUrl = "http://example.com/image.jpg";
      const optionsWithImage: AIRequestOptions = {
        ...basicOptions,
        model: mockGeminiProVision.apiName,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe this." },
              { type: "image", url: imageUrl },
            ],
          },
        ],
      };

      const mockImageData = {
        mimeType: "image/jpeg",
        base64Data: "base64jpegstring",
      };
      MockedGetImageDataFromUrl.mockResolvedValue(mockImageData);

      async function* mockVisionStream() {
        yield { candidates: [{ content: { parts: [{ text: "It's a..." }] } }] };
      }
      mockGenerateContentStream.mockReturnValue(mockVisionStream());

      const stream = adapter.streamResponse(optionsWithImage);
      await stream.next();

      expect(MockedGetImageDataFromUrl).toHaveBeenCalledWith(imageUrl);
      
      // Fix: Expectation must now include mediaResolution for the streaming test too
      expect(mockGenerateContentStream).toHaveBeenCalledWith(
        expect.objectContaining({
          contents: expect.arrayContaining([
            expect.objectContaining({
              parts: expect.arrayContaining([
                { text: "Describe this." },
                {
                  inlineData: {
                    mimeType: mockImageData.mimeType,
                    data: mockImageData.base64Data,
                  },
                  mediaResolution: { level: "MEDIA_RESOLUTION_HIGH" },
                },
              ]),
            }),
          ]),
        })
      );
    });
  });
});