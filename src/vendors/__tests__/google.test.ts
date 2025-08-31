import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  Message,
  UsageResponse,
  Chat,
  ImageBlock,
  ChatResponse,
  NotImplementedError,
  ContentBlock,
  AIResponse,
} from "../../types";
import { GoogleGenAI } from "@google/genai";
import { getImageDataFromUrl, computeResponseCost } from "../../utils";

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
  // Mock other exports if they were needed
}));

// --- Mock the utility function ---
jest.mock("../../utils", () => ({
  ...jest.requireActual("../../utils"),
  getImageDataFromUrl: jest.fn(),
}));
// --- Get Typed Mocks AFTER jest.doMock ---
const MockedGetImageDataFromUrl = getImageDataFromUrl as jest.Mock;

describe("GoogleAIAdapter", () => {
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
    // ... other capability checks
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

    // THIS TEST IS REWRITTEN. It no longer checks for a warning.
    // It now tests the SUCCESSFUL handling of an ImageBlock URL.
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

      // Mock the return value of our utility function
      const mockImageData = {
        mimeType: "image/gif",
        base64Data: "base64gifstring",
      };
      MockedGetImageDataFromUrl.mockResolvedValue(mockImageData);
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);

      await adapter.generateResponse(optionsWithImageUrl);

      // Verify the utility was called with the correct URL
      expect(MockedGetImageDataFromUrl).toHaveBeenCalledWith(imageUrl);

      // Verify the SDK was called with the converted base64 data
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
                },
              ],
            },
          ],
        })
      );
    });

    it("should yield content chunks and a final meta block with usage", async () => {
      // 1. Mock the stream from the Google SDK
      async function* mockStreamWithUsage() {
        // First chunk: just text
        yield {
          candidates: [{ content: { parts: [{ text: "This " }] } }],
        };
        // Second chunk: more text
        yield {
          candidates: [{ content: { parts: [{ text: "is a " }] } }],
        };
        // Final chunk: last bit of text AND the usage metadata
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

      // Because the adapter is initialized with mockGeminiProVision in beforeEach,
      // we need to use its costs for the assertion.
      const testAdapter = new GoogleAIAdapter(mockConfig, mockGeminiProVision);

      // 2. Call the adapter's streamResponse method
      const stream = testAdapter.streamResponse({
        ...basicOptions,
        model: mockGeminiProVision.apiName,
      });

      // 3. Collect all the chunks yielded by our adapter
      const receivedBlocks: ContentBlock[] = [];
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      // 4. Assert the results
      // We expect 3 text blocks and 1 meta block
      expect(receivedBlocks).toHaveLength(4);

      // Check the content blocks
      expect(receivedBlocks[0]).toEqual({ type: "text", text: "This " });
      expect(receivedBlocks[1]).toEqual({ type: "text", text: "is a " });
      expect(receivedBlocks[2]).toEqual({ type: "text", text: "test." });

      // --- THIS IS THE CORRECTED PART ---
      // Check the final meta block
      const metaBlock = receivedBlocks[3];
      expect(metaBlock.type).toBe("meta");
      expect((metaBlock as any).responseId).toBeDefined();

      // Calculate expected costs using the helper function
      const expectedInputCost = computeResponseCost(
        15,
        testAdapter.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        25,
        testAdapter.outputTokenCost!
      );
      const expectedTotalCost = expectedInputCost + expectedOutputCost;

      // Test the usage object properties individually for floating-point precision
      const receivedUsage = (metaBlock as any).usage;
      expect(receivedUsage).toBeDefined();
      expect(receivedUsage.inputCost).toBeCloseTo(expectedInputCost);
      expect(receivedUsage.outputCost).toBeCloseTo(expectedOutputCost);
      expect(receivedUsage.totalCost).toBeCloseTo(expectedTotalCost);
    });

    // THIS TEST IS REMOVED as it is now redundant.
    // The `visionUrl` parameter is deprecated, and its functionality
    // is fully covered by the `ImageBlock` test above.
    // it("should handle visionUrl correctly", async () => { ... });
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

    // THIS TEST IS UPDATED to check the new behavior of sendChat
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

      // Verify that sendChat constructed the correct `messages` array
      // with the prompt text and an ImageBlock from the visionUrl.
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
          // The top-level visionUrl in AIRequestOptions should be undefined
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

    // THIS TEST IS UPDATED to use an ImageBlock instead of `visionUrl`
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
      await stream.next(); // Execute up to the first yield to trigger the call

      // Verify the utility was called
      expect(MockedGetImageDataFromUrl).toHaveBeenCalledWith(imageUrl);
      // Verify the stream was initiated with the correct, converted data
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
                },
              ]),
            }),
          ]),
        })
      );
    });
  });
});
