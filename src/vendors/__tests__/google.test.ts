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
  NotImplementedError,
  ContentBlock,
} from "../../types";
import {
  GoogleGenAI, // Import the NEW class
  HarmCategory,
  HarmBlockThreshold,
} from "@google/genai"; // Use the NEW SDK path
import { computeResponseCost, getImageDataFromUrl } from "../../utils"; // Import the function to mock
// No longer need the GoogleAdapterModule type import
// import type * as GoogleAdapterModule from "../google";

// --- Define Mocks FIRST ---
// Mock for the new SDK structure: client.models.generateContent
const mockGenerateContent = jest.fn();
const mockGenerateContentStream = jest.fn();
// Mock for the constructor
const mockGoogleGenAIConstructor = jest.fn().mockImplementation(() => ({
  models: {
    generateContent: mockGenerateContent,
    generateContentStream: mockGenerateContentStream,
  },
}));

// --- Mock the SDK using jest.doMock (not hoisted) ---
// Mock the NEW SDK path
jest.doMock("@google/genai", () => ({
  GoogleGenAI: mockGoogleGenAIConstructor,
  // Mock other exports if needed by the adapter
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

// --- Mock the utility function that uses file-type ---
const mockGetImageDataFromUrl = jest.fn();
jest.mock("../../utils", () => ({
  // Keep other exports from utils if needed, mock only getImageDataFromUrl
  ...jest.requireActual("../../utils"), // Keep actual implementations of other utils
  getImageDataFromUrl: (...args: any[]) => mockGetImageDataFromUrl(...args),
}));

// --- Get Typed Mock Constructor AFTER jest.doMock ---
// Cast the mock constructor to the correct type from the NEW SDK
const MockedGoogleGenAI = GoogleGenAI as jest.MockedClass<typeof GoogleGenAI>;

describe("GoogleAIAdapter", () => {
  // Import the adapter normally now
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

  // No need for dynamic import in beforeAll anymore
  beforeAll(async () => {
    // Import normally
    const module = await import("../google");
    GoogleAIAdapter = module.GoogleAIAdapter;
  });

  beforeEach(() => {
    // Clear mocks before each test
    mockGoogleGenAIConstructor.mockClear();
    mockGenerateContent.mockClear();
    mockGenerateContentStream.mockClear();

    // Re-initialize adapter before each test
    adapter = new GoogleAIAdapter(mockConfig, mockGeminiProVision);
  });

  it("should initialize GoogleGenAI client with correct config", () => {
    // Adapter initialization happens in beforeEach
    expect(mockGoogleGenAIConstructor).toHaveBeenCalledTimes(1);
    // Check if the constructor was called with the expected API key config
    expect(mockGoogleGenAIConstructor).toHaveBeenCalledWith({
      apiKey: mockConfig.apiKey,
    });
  });

  it("should reflect capabilities from ModelConfig", () => {
    expect(adapter.isVisionCapable).toBe(true);
    expect(adapter.isImageGenerationCapable).toBe(false);
    expect(adapter.isThinkingCapable).toBe(false);
    expect(adapter.inputTokenCost).toBe(mockGeminiProVision.inputTokenCost);
    expect(adapter.outputTokenCost).toBe(mockGeminiProVision.outputTokenCost);
  });

  // --- Mocks needed by multiple test blocks ---
  // Mock response structure for the NEW SDK
  const mockBasicApiResponse = {
    candidates: [
      {
        content: {
          role: "model",
          parts: [{ text: "Hi there!" }],
        },
        // Other candidate properties like finishReason, safetyRatings etc. might exist
      },
    ],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    },
  }; // Removed extra };

  // Mock axios for visionUrl test - Define outside generateResponse block
  const mockAxiosGet = jest.fn();
  jest.mock("axios", () => ({
    get: (...args: any[]) => mockAxiosGet(...args),
  }));

  // utils.getImageDataFromUrl is mocked above
  // --- generateResponse Tests ---
  describe("generateResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hello Gemini!" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockGeminiPro.apiName, // Use non-vision model for basic text test
      messages: basicMessages,
    };

    it("should call Google models.generateContent with correct basic parameters", async () => {
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      // Use the non-vision adapter for this basic text test
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);

      await textAdapter.generateResponse(basicOptions);

      expect(mockGenerateContent).toHaveBeenCalledTimes(1);
      // Check the NEW argument structure (single object)
      expect(mockGenerateContent).toHaveBeenCalledWith({
        model: mockGeminiPro.apiName,
        contents: [{ role: "user", parts: [{ text: "Hello Gemini!" }] }],
        generationConfig: {}, // Empty by default
        // systemInstruction should be absent by default
      });
    });

    it("should map Google response to AIResponse format including usage", async () => {
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro); // Use model with costs defined

      const response = await textAdapter.generateResponse(basicOptions);

      const expectedInputCost = computeResponseCost(
        mockBasicApiResponse.usageMetadata.promptTokenCount,
        mockGeminiPro.inputTokenCost!
      );
      const expectedOutputCost = computeResponseCost(
        mockBasicApiResponse.usageMetadata.candidatesTokenCount,
        mockGeminiPro.outputTokenCost!
      );
      const expectedUsage: UsageResponse = {
        inputCost: expectedInputCost,
        outputCost: expectedOutputCost,
        totalCost: expectedInputCost + expectedOutputCost,
      }; // Removed extra };

      expect(response).toEqual<AIResponse>({
        role: "assistant",
        content: [{ type: "text", text: "Hi there!" }], // Text from the mock response part
        usage: expectedUsage,
      });
    });

    // --- Tests for specific features ---

    it("should handle systemPrompt correctly", async () => {
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      const optionsWithSystemPrompt: AIRequestOptions = {
        ...basicOptions,
        systemPrompt: "Be helpful.",
      };
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);

      await textAdapter.generateResponse(optionsWithSystemPrompt);

      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          systemInstruction: {
            role: "system",
            parts: [{ text: "Be helpful." }],
          },
          contents: [{ role: "user", parts: [{ text: "Hello Gemini!" }] }], // Contents remain separate
        })
      );
    });

    it("should handle maxTokens correctly", async () => {
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      const optionsWithMaxTokens: AIRequestOptions = {
        ...basicOptions,
        maxTokens: 150,
      };
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);

      await textAdapter.generateResponse(optionsWithMaxTokens);

      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          generationConfig: { maxOutputTokens: 150 },
        })
      );
    });

    // --- Vision Tests (using the vision-capable adapter initialized in beforeEach) ---
    it("should map ImageDataBlock to Google Part format (Vision)", async () => {
      // Use the vision-capable adapter initialized in beforeEach
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
      // Ensure options use the correct vision model
      const optionsWithImage: AIRequestOptions = {
        model: mockGeminiProVision.apiName,
        messages: messagesWithImage,
      };
      // Mock response structure for new SDK
      const mockVisionApiResponse = {
        candidates: [
          { content: { role: "model", parts: [{ text: "It's a PNG." }] } },
        ],
        usageMetadata: {
          promptTokenCount: 50,
          candidatesTokenCount: 5,
          totalTokenCount: 55,
        },
      };
      mockGenerateContent.mockResolvedValue(mockVisionApiResponse);

      await adapter.generateResponse(optionsWithImage); // adapter is vision capable

      // Assert call structure for new SDK
      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockGeminiProVision.apiName,
          contents: [
            {
              role: "user",
              parts: [
                {
                  inlineData: {
                    mimeType: "image/png",
                    data: "base64pngstring",
                  },
                },
                { text: "What is this PNG?" },
              ],
            },
          ],
          generationConfig: {}, // Default empty
        })
      );
    });

    // Clear axios/file-type mocks before each relevant test if needed,
    // but since they are defined within this describe block,
    // they should reset automatically. Add explicit clears if issues arise.
    // beforeEach(() => {
    //     mockAxiosGet.mockClear();
    // Clear the mock for the utility function before each test
    beforeEach(() => {
      mockAxiosGet.mockClear(); // Still need to clear axios mock if used elsewhere
      mockGetImageDataFromUrl.mockClear();
    });

    it("should handle visionUrl correctly", async () => {
      // Use the vision-capable adapter
      const visionUrl = "http://example.com/image.jpeg";
      const optionsWithVisionUrl: AIRequestOptions = {
        model: mockGeminiProVision.apiName,
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "Describe this image." }],
          },
        ],
        visionUrl: visionUrl,
      };
      const mockImageData = {
        mimeType: "image/jpeg",
        base64Data: "base64jpegstring",
      };
      const mockAxiosResponse = {
        data: Buffer.from("jpegdata", "binary"),
        headers: { "content-type": "image/jpeg" },
      };
      // Setup mocks for this specific test
      mockAxiosGet.mockResolvedValue(mockAxiosResponse);
      // Setup mocks for this specific test
      mockAxiosGet.mockResolvedValue(mockAxiosResponse); // Keep axios mock for completeness if adapter still uses it internally for other things
      // Set the return value for the mocked utility function
      mockGetImageDataFromUrl.mockResolvedValue(mockImageData);

      const mockVisionApiResponse = {
        candidates: [
          { content: { role: "model", parts: [{ text: "It's a JPEG." }] } },
        ],
        usageMetadata: {
          promptTokenCount: 60,
          candidatesTokenCount: 6,
          totalTokenCount: 66,
        },
      };
      mockGenerateContent.mockResolvedValue(mockVisionApiResponse);

      await adapter.generateResponse(optionsWithVisionUrl);

      // We no longer need to check mockAxiosGet because we mocked the function that calls it.
      // expect(mockAxiosGet).toHaveBeenCalledWith(visionUrl, {
      //   responseType: "arraybuffer",
      // });

      // Verify the mocked utility function was called
      expect(mockGetImageDataFromUrl).toHaveBeenCalledWith(visionUrl);
      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockGeminiProVision.apiName,
          contents: [
            {
              role: "user",
              parts: [
                { text: "Describe this image." },
                // The image part should be added here
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

    it("should warn and skip ImageBlock (URL) - still relevant", async () => {
      // Use the vision-capable adapter
      const imageBlock: ImageBlock = {
        type: "image",
        url: "http://example.com/image.gif",
      };
      const messagesWithImageUrl: Message[] = [
        {
          role: "user",
          content: [imageBlock, { type: "text", text: "What about this GIF?" }],
        },
      ];
      // Ensure options use the correct vision model
      const optionsWithImageUrl: AIRequestOptions = {
        model: mockGeminiProVision.apiName,
        messages: messagesWithImageUrl,
      };
      // Response structure for new SDK
      const mockApiResponse = {
        candidates: [
          { content: { role: "model", parts: [{ text: "A GIF URL." }] } },
        ],
        usageMetadata: {
          promptTokenCount: 12,
          candidatesTokenCount: 4,
          totalTokenCount: 16,
        },
      };
      mockGenerateContent.mockResolvedValue(mockApiResponse);
      const consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation(); // Keep spy

      await adapter.generateResponse(optionsWithImageUrl); // Use vision adapter

      // Check that the image block part was NOT included in the call to the SDK
      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockGeminiProVision.apiName,
          contents: [
            {
              role: "user",
              parts: [{ text: "What about this GIF?" }], // Only text part remains
            },
          ],
        })
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          "Google adapter received 'image' block (URL), needs pre-processing to base64 inlineData."
        )
      );
      consoleWarnSpy.mockRestore(); // Restore console.warn
    });

    it("should handle thinking config correctly", async () => {
      const thinkingModelConfig: ModelConfig = {
        ...mockGeminiPro,
        isThinking: true,
      };
      const thinkingAdapter = new GoogleAIAdapter(
        mockConfig,
        thinkingModelConfig
      );
      const optionsWithThinking: AIRequestOptions = {
        ...basicOptions,
        budgetTokens: 50, // Example budget
      };
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse); // Use basic response for simplicity

      await thinkingAdapter.generateResponse(optionsWithThinking);

      expect(mockGenerateContent).toHaveBeenCalledWith(
        expect.objectContaining({
          generationConfig: expect.objectContaining({
            thinkingConfig: { includeThoughts: true, thinkingBudget: 50 },
          }),
        })
      );
    });

    it("should parse thinking response part", async () => {
      const thinkingModelConfig: ModelConfig = {
        ...mockGeminiPro,
        isThinking: true,
      };
      const thinkingAdapter = new GoogleAIAdapter(
        mockConfig,
        thinkingModelConfig
      );
      const mockThinkingApiResponse = {
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                { thought: true, text: "Thinking step 1..." }, // Thought part
                { text: "Okay, here is the answer." },
              ],
            },
          },
        ],
        usageMetadata: {
          promptTokenCount: 20,
          candidatesTokenCount: 15,
          totalTokenCount: 35,
        },
      };
      mockGenerateContent.mockResolvedValue(mockThinkingApiResponse);

      const response = await thinkingAdapter.generateResponse(basicOptions);

      expect(response.content).toEqual([
        {
          type: "thinking",
          thinking: "Thinking step 1...",
          signature: "google",
        },
        { type: "text", text: "Okay, here is the answer." },
      ]);
    });

    it("should throw error for invalid response structure", async () => {
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);

      const invalidResponse1 = { candidates: null }; // Missing candidates
      mockGenerateContent.mockResolvedValue(invalidResponse1);
      await expect(textAdapter.generateResponse(basicOptions)).rejects.toThrow(
        "Invalid response structure from @google/genai API"
      );

      const invalidResponse2 = { candidates: [{ content: null }] }; // Missing content
      mockGenerateContent.mockResolvedValue(invalidResponse2);
      await expect(textAdapter.generateResponse(basicOptions)).rejects.toThrow(
        "Invalid response structure from @google/genai API"
      );

      const invalidResponse3 = { candidates: [{ content: { parts: null } }] }; // Missing parts
      mockGenerateContent.mockResolvedValue(invalidResponse3);
      await expect(textAdapter.generateResponse(basicOptions)).rejects.toThrow(
        "Invalid response structure from @google/genai API"
      );
    });
  });

  // --- sendChat Tests ---
  describe("sendChat", () => {
    // Define basic chat object with all optional fields explicitly undefined
    const basicChat: Chat = {
      model: mockGeminiPro.apiName,
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
      prompt: "Current question?",
      maxTokens: 100,
      systemPrompt: "Be concise.",
      visionUrl: null, // Use null for optional props
      imageURL: null, // Use null for optional props
      budgetTokens: null, // Use null for optional props
    };

    it("should call generateResponse with correctly mapped Chat options", async () => {
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);
      // Spy on the instance's generateResponse method
      const generateResponseSpy = jest
        .spyOn(textAdapter, "generateResponse")
        .mockResolvedValue({
          // Provide a minimal valid AIResponse for the spy
          role: "assistant",
          content: [{ type: "text", text: "Mocked response" }],
        });
      // Mock the underlying SDK call just in case, though the spy should prevent it
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);

      await textAdapter.sendChat(basicChat);

      expect(generateResponseSpy).toHaveBeenCalledTimes(1);
      // Explicitly map null -> undefined for properties where AIRequestOptions expects undefined
      // The prompt is no longer automatically added to messages in sendChat
      const expectedOptions: AIRequestOptions = {
        model: basicChat.model,
        messages: [
          ...basicChat.responseHistory,
          // The prompt message is no longer added here
        ],
        maxTokens:
          basicChat.maxTokens === null ? undefined : basicChat.maxTokens, // Map null to undefined
        budgetTokens:
          basicChat.budgetTokens === null ? undefined : basicChat.budgetTokens, // Map null to undefined
        systemPrompt: basicChat.systemPrompt, // AIRequestOptions allows null here
        visionUrl:
          basicChat.visionUrl === null ? undefined : basicChat.visionUrl, // Map null to undefined
      };
      expect(generateResponseSpy).toHaveBeenCalledWith(expectedOptions);

      generateResponseSpy.mockRestore(); // Clean up spy
    });

    it("should return the result from generateResponse", async () => {
      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);
      const mockAiResponse: AIResponse = {
        role: "assistant",
        content: [{ type: "text", text: "Chat response" }],
        usage: { inputCost: 0.1, outputCost: 0.2, totalCost: 0.3 },
      };
      // Mock generateResponse directly for this test
      const generateResponseSpy = jest
        .spyOn(textAdapter, "generateResponse")
        .mockResolvedValue(mockAiResponse);

      const response = await textAdapter.sendChat(basicChat);

      expect(response).toEqual<ChatResponse>({
        role: mockAiResponse.role,
        content: mockAiResponse.content,
        usage: mockAiResponse.usage,
      });

      generateResponseSpy.mockRestore();
    });

    it("should handle chat with visionUrl", async () => {
      // Use vision adapter (initialized in global beforeEach)
      const chatWithVision: Chat = {
        ...basicChat,
        model: mockGeminiProVision.apiName, // Use vision model
        visionUrl: "http://example.com/image.png",
        prompt: "What's in this image?",
      };
      const generateResponseSpy = jest
        .spyOn(adapter, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "Mocked vision response" }],
        });
      // Mock underlying calls just in case
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      // Mock underlying calls for sendChat test
      mockGenerateContent.mockResolvedValue(mockBasicApiResponse);
      mockAxiosGet.mockResolvedValue({
        data: Buffer.from("pngdata", "binary"),
        headers: { "content-type": "image/png" },
      });
      // Set the return value for the mocked utility function for this test
      mockGetImageDataFromUrl.mockResolvedValue({
        mimeType: "image/png",
        base64Data: "base64pngstring", // Example data consistent with the test
      });

      await adapter.sendChat(chatWithVision); // Use vision adapter instance

      expect(generateResponseSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          visionUrl: chatWithVision.visionUrl,
          model: mockGeminiProVision.apiName,
          messages: [
            ...chatWithVision.responseHistory,
            // The prompt message is no longer added here
          ],
          // visionUrl is passed separately and handled within generateResponse
        })
      );
      generateResponseSpy.mockRestore();
    });
  });

  // --- generateImage Tests ---
  // This test remains valid as the method signature/behavior hasn't changed
  describe("generateImage", () => {
    it("should throw NotImplementedError with correct message", async () => {
      // Use AIRequestOptions as required by the updated interface
      const dummyOptions: AIRequestOptions = { model: "test", messages: [] };
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        NotImplementedError
      );
      // Check the specific error message from the adapter
      await expect(adapter.generateImage(dummyOptions)).rejects.toThrow(
        "Google generates images inline with text, use generateResponse/sendChat."
      );
    });
  });

  // Removed sendMCPChat tests as the method was removed

  // --- streamResponse Tests ---
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

    it("should stream text, image, and thinking content blocks from parts", async () => {
      const mockImageBase64 = "fake-base64-png-data";
      async function* mockStream() {
        // Chunk with a thought
        yield {
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  { thought: true, text: "Okay, I will generate an image." },
                ],
              },
            },
          ],
        };
        // Chunk with text
        yield {
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ text: "Here is the image: " }],
              },
            },
          ],
        };
        // Chunk with an image
        yield {
          candidates: [
            {
              content: {
                role: "model",
                parts: [
                  {
                    inlineData: {
                      mimeType: "image/jpeg",
                      data: mockImageBase64,
                    },
                  },
                ],
              },
            },
          ],
        };
      }
      mockGenerateContentStream.mockReturnValue(mockStream());

      const visionAdapter = new GoogleAIAdapter(mockConfig, {
        ...mockGeminiProVision,
        isThinking: true,
      });
      const stream = visionAdapter.streamResponse({
        ...basicOptions,
        model: mockGeminiProVision.apiName,
      });
      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBe(3);
      expect(chunks[0]).toEqual({
        type: "thinking",
        thinking: "Okay, I will generate an image.",
        signature: "google",
      });
      expect(chunks[1]).toEqual({
        type: "text",
        text: "Here is the image: ",
      });
      expect(chunks[2]).toEqual({
        type: "image_data",
        mimeType: "image/jpeg",
        base64Data: mockImageBase64,
      });
      expect(mockGenerateContentStream).toHaveBeenCalledTimes(1);
    });

    it("should yield an error block if the stream fails", async () => {
      const errorMessage = "Stream failed!";
      mockGenerateContentStream.mockImplementation(() => {
        throw new Error(errorMessage);
      });

      const textAdapter = new GoogleAIAdapter(mockConfig, mockGeminiPro);
      const stream = textAdapter.streamResponse(basicOptions);

      const chunks: ContentBlock[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBe(1);
      expect(chunks[0].type).toBe("error");
      expect((chunks[0] as any).privateMessage).toBe(errorMessage);
    });

    it("should handle visionUrl correctly in streaming", async () => {
      const visionUrl = "http://example.com/image.jpg";
      const optionsWithVisionUrl: AIRequestOptions = {
        ...basicOptions,
        visionUrl: visionUrl,
        model: mockGeminiProVision.apiName,
      };

      // Mock image data processing
      const mockImageData = {
        mimeType: "image/jpeg",
        base64Data: "base64jpegstring",
      };
      mockGetImageDataFromUrl.mockResolvedValue(mockImageData);

      // Mock stream response
      async function* mockVisionStream() {
        yield { text: "This is an image description" };
      }
      mockGenerateContentStream.mockReturnValue(mockVisionStream());

      const visionAdapter = new GoogleAIAdapter(
        mockConfig,
        mockGeminiProVision
      );
      const stream = visionAdapter.streamResponse(optionsWithVisionUrl);
      await stream.next(); // This executes the generator's code until the first `yield`

      expect(mockGetImageDataFromUrl).toHaveBeenCalledWith(visionUrl);
      expect(mockGenerateContentStream).toHaveBeenCalledWith(
        expect.objectContaining({
          contents: expect.arrayContaining([
            expect.objectContaining({
              parts: expect.arrayContaining([
                expect.objectContaining({
                  inlineData: {
                    mimeType: mockImageData.mimeType,
                    data: mockImageData.base64Data,
                  },
                }),
              ]),
            }),
          ]),
        })
      );
    });
  });
});
