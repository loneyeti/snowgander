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
      mockClient.chat.completions.create.mockClear();
    });

    it("should call chat.completions.create with stream: true and correct parameters", async () => {
      const stream = adapter.streamResponse(basicOptions);
      await stream.next(); // Start the generator to trigger the API call

      expect(mockClient.chat.completions.create).toHaveBeenCalledTimes(1);
      expect(mockClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockVisionModel.apiName,
          messages: [
            {
              role: "user",
              content: [{ type: "text", text: "Tell me a story" }],
            },
          ],
          stream: true,
        })
      );
    });

    it("should yield TextBlocks for each content delta from the stream", async () => {
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

    it("should handle system prompt and vision content in the request", async () => {
      const visionMessages: Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            { type: "image", url: "http://example.com/image.png" },
          ],
        },
      ];
      const optionsWithVision: AIRequestOptions = {
        ...basicOptions,
        messages: visionMessages,
        systemPrompt: "You are a helpful assistant.",
      };

      const stream = adapter.streamResponse(optionsWithVision);
      await stream.next(); // Start the generator

      expect(mockClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            {
              role: "user",
              content: [
                { type: "text", text: "What's in this image?" },
                {
                  type: "image_url",
                  image_url: { url: "http://example.com/image.png" },
                },
              ],
            },
          ],
          stream: true,
        })
      );
    });

    it("should yield an ErrorBlock if the API call throws an error", async () => {
      const apiError = new Error("API Connection Failed");
      mockClient.chat.completions.create.mockRejectedValue(apiError);
      const receivedBlocks: ContentBlock[] = [];

      const stream = adapter.streamResponse(basicOptions);
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      expect(receivedBlocks.length).toBe(1);
      expect(receivedBlocks[0].type).toBe("error");
      const errorBlock = receivedBlocks[0] as any;
      expect(errorBlock.publicMessage).toBe(
        "An error occurred while streaming the response."
      );
      expect(errorBlock.privateMessage).toBe("API Connection Failed");
    });
  });
});
