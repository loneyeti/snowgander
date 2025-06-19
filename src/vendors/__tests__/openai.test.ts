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
  ContentBlock,
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
      create: jest.fn().mockImplementation(() => {
        // Create a mock async generator for streaming
        async function* mockStreamGenerator() {
          yield { choices: [{ delta: { content: "Hello " } }] };
          yield { choices: [{ delta: { content: "world!" } }] };
        }
        return mockStreamGenerator();
      }),
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
  };

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

  // ... (all existing test cases remain unchanged) ...

  describe("streamResponse", () => {
    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Tell me a story" }] },
    ];
    const basicOptions: AIRequestOptions = {
      model: mockVisionModel.apiName,
      messages: basicMessages,
    };

    beforeEach(() => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      mockResponsesCreate.mockClear();
    });

    it("should call responses.create with stream: true and correct parameters", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      async function* mockStream() {}
      mockResponsesCreate.mockResolvedValue(mockStream());

      const stream = adapter.streamResponse(basicOptions);
      await stream.next();

      expect(mockResponsesCreate).toHaveBeenCalledTimes(1);
      const callArgs = mockResponsesCreate.mock.calls[0][0];
      expect(callArgs).toHaveProperty("model", mockVisionModel.apiName);
      expect(callArgs).toHaveProperty("stream", true);
      expect(callArgs.input).toEqual([
        {
          role: "user",
          content: [{ type: "input_text", text: "Tell me a story" }],
        },
      ]);
    });

    it("should yield TextBlocks for 'response.output_text.delta' events", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      async function* mockTextStream() {
        yield { type: "response.created", response: {} };
        yield { type: "response.output_text.delta", delta: "Hello " };
        yield { type: "response.output_text.delta", delta: "world!" };
        yield { type: "response.completed", response: {} };
      }
      mockResponsesCreate.mockResolvedValue(mockTextStream());

      const receivedBlocks: ContentBlock[] = [];
      const stream = adapter.streamResponse(basicOptions);
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks).toEqual([
        { type: "text", text: "Hello " },
        { type: "text", text: "world!" },
      ]);
    });

    it("should stream text and image data for mixed content events", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const mockImageBase64 = "base64-encoded-image-string";

      async function* mockImageStream() {
        yield {
          type: "response.output_text.delta",
          delta: "Here is your image: ",
        };
        yield {
          type: "response.image_generation_call.partial_image",
          partial_image_b64: mockImageBase64,
        };
        yield { type: "response.output_text.delta", delta: "!" };
      }
      mockResponsesCreate.mockResolvedValue(mockImageStream());

      const optionsWithImageGen: AIRequestOptions = {
        ...basicOptions,
        openaiImageGenerationOptions: { quality: "high", size: "1024x1024" },
      };

      const receivedBlocks: ContentBlock[] = [];
      const stream = adapter.streamResponse(optionsWithImageGen);
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks).toEqual([
        { type: "text", text: "Here is your image: " },
        {
          type: "image_data",
          id: null,
          mimeType: "image/png",
          base64Data: mockImageBase64,
        },
        { type: "text", text: "!" },
      ]);
    });

    it("should yield an ErrorBlock for 'response.failed' event", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const failureMessage = "Model overload";
      async function* mockFailureStream() {
        yield {
          type: "response.failed",
          response: { error: { message: failureMessage } },
        };
      }
      mockResponsesCreate.mockResolvedValue(mockFailureStream());

      // Temporarily suppress console.error for this specific test
      const consoleErrorSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      const receivedBlocks: ContentBlock[] = [];
      const stream = adapter.streamResponse(basicOptions);
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks.length).toBe(1);
      expect(receivedBlocks[0].type).toBe("error");
      const errorBlock = receivedBlocks[0] as any;
      expect(errorBlock.publicMessage).toBe(
        "The request failed during streaming."
      );
      expect(errorBlock.privateMessage).toBe(failureMessage);

      // Restore the original console.error function
      consoleErrorSpy.mockRestore();
    });
  });
});
