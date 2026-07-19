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

  describe("GPT-5.x / reasoning model support", () => {
    const gpt56Model: ModelConfig = {
      apiName: "gpt-5.6-sol",
      isVision: false,
      isImageGeneration: false,
      isThinking: true,
    };
    const gpt5Model: ModelConfig = {
      apiName: "gpt-5",
      isVision: false,
      isImageGeneration: false,
      isThinking: true,
    };
    const o3Model: ModelConfig = {
      apiName: "o3",
      isVision: false,
      isImageGeneration: false,
      isThinking: true,
    };
    const gpt4oModel: ModelConfig = {
      apiName: "gpt-4o",
      isVision: true,
      isImageGeneration: false,
      isThinking: false,
    };

    const basicMessages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Hi" }] },
    ];

    const mockBasicResponse = {
      id: "resp-basic",
      output_text: "Hello there.",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello there." }],
        },
      ],
    };

    beforeEach(() => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      mockResponsesCreate.mockResolvedValue(mockBasicResponse);
    });

    it("sends explicit effort 'xhigh' on gpt-5.6 models", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a = new OpenAIAdapter(mockConfig, gpt56Model);
      await a.generateResponse({
        model: gpt56Model.apiName,
        messages: basicMessages,
        effort: "xhigh",
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning).toEqual({
        effort: "xhigh",
        summary: "auto",
      });
    });

    it("maps budgetTokens to effort levels (0->low, 4000->medium, 10000->high)", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a = new OpenAIAdapter(mockConfig, gpt5Model);

      await a.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        budgetTokens: 0,
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning.effort).toBe(
        "low"
      );

      mockResponsesCreate.mockClear();
      await a.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        budgetTokens: 4000,
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning.effort).toBe(
        "medium"
      );

      mockResponsesCreate.mockClear();
      await a.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        budgetTokens: 10000,
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning.effort).toBe(
        "high"
      );
    });

    it("prefers explicit effort over a budgetTokens-derived value", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a = new OpenAIAdapter(mockConfig, gpt5Model);
      await a.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        budgetTokens: 10000, // would map to "high"
        effort: "low",
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning.effort).toBe(
        "low"
      );
    });

    it("applies reasoning.mode 'pro' on gpt-5.6 but ignores it (with a warning) on gpt-5", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a56 = new OpenAIAdapter(mockConfig, gpt56Model);
      await a56.generateResponse({
        model: gpt56Model.apiName,
        messages: basicMessages,
        effort: "high",
        reasoningMode: "pro",
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning.mode).toBe("pro");

      mockResponsesCreate.mockClear();
      const consoleWarnSpy = jest
        .spyOn(console, "warn")
        .mockImplementation(() => {});
      const a5 = new OpenAIAdapter(mockConfig, gpt5Model);
      await a5.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        effort: "high",
        reasoningMode: "pro",
      });
      expect(
        mockResponsesCreate.mock.calls[0][0].reasoning.mode
      ).toBeUndefined();
      consoleWarnSpy.mockRestore();
    });

    it("forwards verbosity as text.verbosity on gpt-5.x but omits it (with a warning) on gpt-4o/o3", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a5 = new OpenAIAdapter(mockConfig, gpt5Model);
      await a5.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        verbosity: "low",
      });
      expect(mockResponsesCreate.mock.calls[0][0].text).toEqual({
        verbosity: "low",
      });

      mockResponsesCreate.mockClear();
      const consoleWarnSpy = jest
        .spyOn(console, "warn")
        .mockImplementation(() => {});
      const a4o = new OpenAIAdapter(mockConfig, gpt4oModel);
      await a4o.generateResponse({
        model: gpt4oModel.apiName,
        messages: basicMessages,
        verbosity: "low",
      });
      expect(mockResponsesCreate.mock.calls[0][0].text).toBeUndefined();
      consoleWarnSpy.mockRestore();
    });

    it("forwards outputFormat as text.format", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a = new OpenAIAdapter(mockConfig, gpt5Model);
      const schema = { type: "json_schema", name: "person", schema: {} };
      await a.generateResponse({
        model: gpt5Model.apiName,
        messages: basicMessages,
        outputFormat: schema,
      });
      expect(mockResponsesCreate.mock.calls[0][0].text).toEqual({
        format: schema,
      });
    });

    it("forwards temperature/top_p on non-reasoning models but omits them (with a warning) on reasoning models", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a4o = new OpenAIAdapter(mockConfig, gpt4oModel);
      await a4o.generateResponse({
        model: gpt4oModel.apiName,
        messages: basicMessages,
        temperature: 0.7,
        topP: 0.9,
      });
      const callArgs = mockResponsesCreate.mock.calls[0][0];
      expect(callArgs.temperature).toBe(0.7);
      expect(callArgs.top_p).toBe(0.9);

      mockResponsesCreate.mockClear();
      const consoleWarnSpy = jest
        .spyOn(console, "warn")
        .mockImplementation(() => {});
      const a56 = new OpenAIAdapter(mockConfig, gpt56Model);
      await a56.generateResponse({
        model: gpt56Model.apiName,
        messages: basicMessages,
        temperature: 0.7,
      });
      const callArgs2 = mockResponsesCreate.mock.calls[0][0];
      expect(callArgs2.temperature).toBeUndefined();
      expect(callArgs2.top_p).toBeUndefined();
      consoleWarnSpy.mockRestore();
    });

    it("does not add a reasoning param (and warns) when budgetTokens is set on a non-reasoning model", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const consoleWarnSpy = jest
        .spyOn(console, "warn")
        .mockImplementation(() => {});
      const a4o = new OpenAIAdapter(mockConfig, gpt4oModel);
      await a4o.generateResponse({
        model: gpt4oModel.apiName,
        messages: basicMessages,
        budgetTokens: 5000,
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning).toBeUndefined();
      consoleWarnSpy.mockRestore();
    });

    it("treats o-series models (e.g. o3) as reasoningLegacy: effort is forwarded", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();
      const a = new OpenAIAdapter(mockConfig, o3Model);
      await a.generateResponse({
        model: o3Model.apiName,
        messages: basicMessages,
        effort: "high",
      });
      expect(mockResponsesCreate.mock.calls[0][0].reasoning).toEqual({
        effort: "high",
        summary: "auto",
      });
    });
  });

  describe("sendChat field forwarding", () => {
    it("forwards effort, temperature, topP, outputFormat, verbosity, reasoningMode, budgetTokens, and previousResponseId to generateResponse", async () => {
      const a = new OpenAIAdapter(mockConfig, mockVisionModel);
      const genSpy = jest
        .spyOn(a, "generateResponse")
        .mockResolvedValue({
          role: "assistant",
          content: [{ type: "text", text: "hi" }],
        } as AIResponse);

      const chat: Chat = {
        responseHistory: [],
        visionUrl: null,
        model: mockVisionModel.apiName,
        prompt: "Hello",
        imageURL: null,
        maxTokens: 100,
        budgetTokens: 10000,
        systemPrompt: "Be nice",
        previousResponseId: "resp-abc",
        effort: "xhigh",
        temperature: 0.5,
        topP: 0.8,
        outputFormat: { type: "json_schema" },
        verbosity: "high",
        reasoningMode: "pro",
      };

      await a.sendChat(chat);

      expect(genSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: mockVisionModel.apiName,
          previousResponseId: "resp-abc",
          budgetTokens: 10000,
          effort: "xhigh",
          temperature: 0.5,
          topP: 0.8,
          outputFormat: { type: "json_schema" },
          verbosity: "high",
          reasoningMode: "pro",
        })
      );
    });
  });

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

    it("should yield ThinkingBlocks for 'reasoning_summary_text.delta' and then TextBlocks", async () => {
      const { mockResponsesCreate } = getMockOpenAIClient();

      async function* mockRealStream() {
        // First, yield the reasoning deltas as seen in the logs
        yield {
          type: "response.reasoning_summary_text.delta",
          delta: "**Respond",
        };
        yield { type: "response.reasoning_summary_text.delta", delta: "ing" };
        yield { type: "response.reasoning_summary_text.delta", delta: " to**" };

        // Then, yield the regular text output deltas
        yield { type: "response.output_text.delta", delta: "Hello!" };
        yield { type: "response.output_text.delta", delta: " I am here." };

        // Finally, yield the completed event
        yield {
          type: "response.completed",
          response: {
            id: "resp-123",
            usage: { input_tokens: 1, output_tokens: 1 },
          },
        };
      }

      mockResponsesCreate.mockResolvedValue(mockRealStream());

      const receivedBlocks: ContentBlock[] = [];
      const stream = adapter.streamResponse(basicOptions);
      for await (const block of stream) {
        receivedBlocks.push(block);
      }

      // We expect 6 blocks: 3 thinking, 2 text, 1 meta
      expect(receivedBlocks.length).toBe(6);

      // Verify the content of the blocks
      const thinkingText = receivedBlocks
        .filter((b) => b.type === "thinking")
        .map((b: any) => b.thinking)
        .join("");
      const regularText = receivedBlocks
        .filter((b) => b.type === "text")
        .map((b: any) => b.text)
        .join("");

      expect(thinkingText).toBe("**Responding to**");
      expect(regularText).toBe("Hello! I am here.");

      // CRITICAL: Verify the order of the blocks
      const blockOrder = receivedBlocks.map((b) => b.type);
      expect(blockOrder).toEqual([
        "thinking",
        "thinking",
        "thinking",
        "text",
        "text",
        "meta",
      ]);
    });
  });
});
