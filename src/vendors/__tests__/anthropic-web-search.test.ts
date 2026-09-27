import { AnthropicAdapter } from "../anthropic";
import {
  AIRequestOptions,
  ContentBlock,
  Message,
  ModelConfig,
  VendorConfig,
} from "../../types";
import { computeResponseCost } from "../../utils";

const mockMessagesCreate = jest.fn();
jest.mock("@anthropic-ai/sdk", () =>
  jest.fn().mockImplementation(() => ({
    messages: { create: mockMessagesCreate },
  }))
);

describe("AnthropicAdapter web search and fetch", () => {
  const config: VendorConfig = { apiKey: "test-anthropic-key" };
  const opus55: ModelConfig = {
    apiName: "claude-opus-5-5",
    isVision: false,
    isImageGeneration: false,
    isThinking: true,
    inputTokenCost: 4, // Per million tokens
    outputTokenCost: 20,
    webSearchCost: 0.01,
  };
  const messages: Message[] = [
    { role: "user", content: [{ type: "text", text: "Research this" }] },
  ];

  const collect = async (
    adapter: AnthropicAdapter,
    options: AIRequestOptions
  ) => {
    const blocks: ContentBlock[] = [];
    for await (const block of adapter.streamResponse(options)) {
      blocks.push(block);
    }
    return blocks;
  };

  const searchResult = {
    type: "web_search_tool_result",
    tool_use_id: "srvtoolu_1",
    caller: { type: "direct" },
    content: [
      {
        type: "web_search_result",
        url: "https://a.example/post",
        title: "Post A",
        page_age: "2 days ago",
        encrypted_content: "enc-a",
      },
    ],
  };
  const fetchResult = {
    type: "web_fetch_tool_result",
    tool_use_id: "srvtoolu_2",
    content: {
      type: "web_fetch_result",
      url: "https://b.example/doc",
      retrieved_at: "2026-09-27T00:00:00Z",
      content: {
        type: "document",
        title: "Doc B",
        source: { type: "text", media_type: "text/plain", data: "body" },
        citations: { enabled: true },
      },
    },
  };

  // A stream of one round, with the given content events and stop reason
  function* round(
    id: string,
    events: any[],
    stopReason: string,
    usage: { input: number; output: number; searches?: number; fetches?: number }
  ) {
    yield {
      type: "message_start",
      message: { id, usage: { input_tokens: usage.input } },
    };
    yield* events;
    yield {
      type: "message_delta",
      delta: { stop_reason: stopReason },
      usage: {
        output_tokens: usage.output,
        server_tool_use: {
          web_search_requests: usage.searches ?? 0,
          web_fetch_requests: usage.fetches ?? 0,
        },
      },
    };
    yield { type: "message_stop" };
  }
  const asStream = (events: Iterable<any>) =>
    (async function* () {
      yield* events;
    })();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("tool versions", () => {
    it.each([
      ["claude-opus-5-5", "web_search_20260209", "web_fetch_20260209"],
      ["claude-sonnet-4-6", "web_search_20260209", "web_fetch_20260209"],
      ["claude-3-5-sonnet-20241022", "web_search_20250305", "web_fetch_20250910"],
      ["claude-haiku-4-5", "web_search_20250305", "web_fetch_20250910"],
    ])("%s uses %s and %s", async (model, searchType, fetchType) => {
      const adapter = new AnthropicAdapter(config, { ...opus55, apiName: model });
      mockMessagesCreate.mockResolvedValue(asStream([]));

      await collect(adapter, {
        model,
        messages,
        webSearch: {
          fetch: true,
          maxUses: 5,
          allowedDomains: ["example.com"],
          blockedDomains: ["ignored.com"],
          userLocation: { country: "US", timezone: "America/Denver" },
        },
      });

      expect(mockMessagesCreate.mock.calls[0][0].tools).toEqual([
        {
          type: searchType,
          name: "web_search",
          max_uses: 5,
          allowed_domains: ["example.com"],
          user_location: {
            type: "approximate",
            country: "US",
            timezone: "America/Denver",
          },
        },
        {
          type: fetchType,
          name: "web_fetch",
          max_uses: 5,
          allowed_domains: ["example.com"],
          citations: { enabled: true },
        },
      ]);
    });

    it("adds only web_search without fetch, and keeps caller tools", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      mockMessagesCreate.mockResolvedValue(asStream([]));
      const custom = { name: "lookup", description: "d", input_schema: {} };

      await collect(adapter, {
        model: opus55.apiName,
        messages,
        tools: [custom],
        webSearch: { blockedDomains: ["spam.example"] },
      });

      expect(mockMessagesCreate.mock.calls[0][0].tools).toEqual([
        custom,
        {
          type: "web_search_20260209",
          name: "web_search",
          blocked_domains: ["spam.example"],
        },
      ]);
    });
  });

  describe("streamResponse", () => {
    it("yields research steps, results, citations, and per-search cost", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      const citation = {
        type: "web_search_result_location",
        url: "https://a.example/post",
        title: "Post A",
        cited_text: "A says so",
        encrypted_index: "idx",
      };
      const docCitation = {
        type: "char_location",
        document_index: 1,
        document_title: "Doc B",
        cited_text: "B says so",
        start_char_index: 0,
        end_char_index: 9,
      };
      mockMessagesCreate.mockResolvedValue(
        asStream(
          round(
            "msg_1",
            [
              {
                type: "content_block_start",
                index: 0,
                content_block: {
                  type: "server_tool_use",
                  id: "srvtoolu_1",
                  name: "web_search",
                  input: {},
                },
              },
              {
                type: "content_block_delta",
                index: 0,
                delta: { type: "input_json_delta", partial_json: '{"query":' },
              },
              {
                type: "content_block_delta",
                index: 0,
                delta: { type: "input_json_delta", partial_json: '"news"}' },
              },
              { type: "content_block_stop", index: 0 },
              { type: "content_block_start", index: 1, content_block: searchResult },
              { type: "content_block_stop", index: 1 },
              { type: "content_block_start", index: 2, content_block: fetchResult },
              { type: "content_block_stop", index: 2 },
              {
                type: "content_block_start",
                index: 3,
                content_block: { type: "text", text: "", citations: null },
              },
              {
                type: "content_block_delta",
                index: 3,
                delta: { type: "citations_delta", citation },
              },
              {
                type: "content_block_delta",
                index: 3,
                delta: { type: "citations_delta", citation: docCitation },
              },
              {
                type: "content_block_delta",
                index: 3,
                delta: { type: "text_delta", text: "Grounded." },
              },
              { type: "content_block_stop", index: 3 },
            ],
            "end_turn",
            { input: 1000, output: 100, searches: 2, fetches: 1 }
          )
        )
      );

      const blocks = await collect(adapter, {
        model: opus55.apiName,
        messages,
        webSearch: { fetch: true },
      });

      expect(blocks).toEqual([
        {
          type: "server_tool_use",
          id: "srvtoolu_1",
          name: "web_search",
          input: '{"query":"news"}',
          status: "completed",
          vendor: "anthropic",
        },
        {
          type: "web_search_tool_result",
          toolUseId: "srvtoolu_1",
          results: [
            {
              url: "https://a.example/post",
              title: "Post A",
              pageAge: "2 days ago",
            },
          ],
          vendor: "anthropic",
          raw: searchResult,
        },
        {
          type: "web_fetch_tool_result",
          toolUseId: "srvtoolu_2",
          url: "https://b.example/doc",
          title: "Doc B",
          retrievedAt: "2026-09-27T00:00:00Z",
          vendor: "anthropic",
          raw: fetchResult,
        },
        {
          type: "text",
          text: "",
          citations: [
            {
              type: "url_citation",
              url: "https://a.example/post",
              title: "Post A",
              citedText: "A says so",
              vendor: "anthropic",
              raw: citation,
            },
          ],
        },
        {
          type: "text",
          text: "",
          citations: [
            {
              type: "url_citation",
              url: "https://b.example/doc",
              title: "Doc B",
              citedText: "B says so",
              vendor: "anthropic",
              raw: docCitation,
            },
          ],
        },
        { type: "text", text: "Grounded." },
        {
          type: "meta",
          responseId: "msg_1",
          usage: {
            inputCost: computeResponseCost(1000, 4),
            outputCost: computeResponseCost(100, 20),
            webSearchCost: 0.02,
            didWebSearch: true,
            webSearchCount: 2,
            webFetchCount: 1,
            totalCost:
              computeResponseCost(1000, 4) + computeResponseCost(100, 20) + 0.02,
          },
        },
      ]);
    });

    it("maps an error result (content object instead of list)", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      const errorResult = {
        type: "web_search_tool_result",
        tool_use_id: "srvtoolu_9",
        content: {
          type: "web_search_tool_result_error",
          error_code: "max_uses_exceeded",
        },
      };
      mockMessagesCreate.mockResolvedValue(
        asStream([
          { type: "content_block_start", index: 0, content_block: errorResult },
        ])
      );

      const blocks = await collect(adapter, {
        model: opus55.apiName,
        messages,
        webSearch: {},
      });

      expect(blocks).toEqual([
        {
          type: "web_search_tool_result",
          toolUseId: "srvtoolu_9",
          results: [],
          errorCode: "max_uses_exceeded",
          vendor: "anthropic",
          raw: errorResult,
        },
      ]);
    });

    it("continues a pause_turn by sending the paused content back and sums usage", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      mockMessagesCreate
        .mockResolvedValueOnce(
          asStream(
            round(
              "msg_1",
              [
                {
                  type: "content_block_start",
                  index: 0,
                  content_block: {
                    type: "thinking",
                    thinking: "",
                    signature: "",
                  },
                },
                {
                  type: "content_block_delta",
                  index: 0,
                  delta: { type: "thinking_delta", thinking: "Plan" },
                },
                {
                  type: "content_block_delta",
                  index: 0,
                  delta: { type: "signature_delta", signature: "sig" },
                },
                { type: "content_block_stop", index: 0 },
                {
                  type: "content_block_start",
                  index: 1,
                  content_block: {
                    type: "server_tool_use",
                    id: "srvtoolu_1",
                    name: "web_search",
                    input: {},
                  },
                },
                {
                  type: "content_block_delta",
                  index: 1,
                  delta: {
                    type: "input_json_delta",
                    partial_json: '{"query":"x"}',
                  },
                },
                { type: "content_block_stop", index: 1 },
                {
                  type: "content_block_start",
                  index: 2,
                  content_block: searchResult,
                },
                { type: "content_block_stop", index: 2 },
              ],
              "pause_turn",
              { input: 100, output: 10, searches: 1 }
            )
          )
        )
        .mockResolvedValueOnce(
          asStream(
            round(
              "msg_2",
              [
                {
                  type: "content_block_start",
                  index: 0,
                  content_block: { type: "text", text: "" },
                },
                {
                  type: "content_block_delta",
                  index: 0,
                  delta: { type: "text_delta", text: "Done" },
                },
                { type: "content_block_stop", index: 0 },
              ],
              "end_turn",
              { input: 200, output: 20, searches: 2 }
            )
          )
        );

      const blocks = await collect(adapter, {
        model: opus55.apiName,
        messages,
        webSearch: {},
      });

      expect(mockMessagesCreate).toHaveBeenCalledTimes(2);
      const second = mockMessagesCreate.mock.calls[1][0];
      expect(second.messages).toEqual([
        { role: "user", content: [{ type: "text", text: "Research this" }] },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "Plan", signature: "sig" },
            {
              type: "server_tool_use",
              id: "srvtoolu_1",
              name: "web_search",
              input: { query: "x" },
            },
            searchResult,
          ],
        },
      ]);
      expect(second.tools).toEqual(mockMessagesCreate.mock.calls[0][0].tools);

      expect(blocks.some((b) => b.type === "error")).toBe(false);
      const meta = blocks[blocks.length - 1];
      expect(meta).toEqual({
        type: "meta",
        responseId: "msg_2",
        usage: {
          inputCost: computeResponseCost(300, 4),
          outputCost: computeResponseCost(30, 20),
          webSearchCost: 0.03,
          didWebSearch: true,
          webSearchCount: 3,
          totalCost:
            computeResponseCost(300, 4) + computeResponseCost(30, 20) + 0.03,
        },
      });
    });

    it("stops after the continuation limit and reports it", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      mockMessagesCreate.mockImplementation(async () =>
        asStream(round("msg_p", [], "pause_turn", { input: 1, output: 1 }))
      );

      const blocks = await collect(adapter, {
        model: opus55.apiName,
        messages,
        webSearch: {},
      });

      // The first request plus 3 continuations
      expect(mockMessagesCreate).toHaveBeenCalledTimes(4);
      expect(blocks.find((b) => b.type === "error")).toMatchObject({
        type: "error",
        publicMessage:
          "The research step limit was reached before the answer finished.",
      });
    });
  });

  describe("generateResponse", () => {
    it("maps research blocks and citations, continuing through pause_turn", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      const pausedContent = [
        {
          type: "server_tool_use",
          id: "srvtoolu_1",
          name: "web_search",
          input: { query: "x" },
        },
        searchResult,
      ];
      mockMessagesCreate
        .mockResolvedValueOnce({
          id: "msg_1",
          content: pausedContent,
          stop_reason: "pause_turn",
          usage: {
            input_tokens: 100,
            output_tokens: 10,
            server_tool_use: { web_search_requests: 1, web_fetch_requests: 0 },
          },
        })
        .mockResolvedValueOnce({
          id: "msg_2",
          content: [
            {
              type: "text",
              text: "Answer",
              citations: [
                {
                  type: "web_search_result_location",
                  url: "https://a.example/post",
                  title: "Post A",
                  cited_text: "quote",
                  encrypted_index: "i",
                },
              ],
            },
          ],
          stop_reason: "end_turn",
          usage: { input_tokens: 200, output_tokens: 20 },
        });

      const response = await adapter.generateResponse({
        model: opus55.apiName,
        messages,
        webSearch: {},
      });

      expect(mockMessagesCreate.mock.calls[1][0].messages[1]).toEqual({
        role: "assistant",
        content: pausedContent,
      });
      expect(response.content.map((b) => b.type)).toEqual([
        "server_tool_use",
        "web_search_tool_result",
        "text",
      ]);
      expect(response.content[0]).toMatchObject({
        input: '{"query":"x"}',
        vendor: "anthropic",
      });
      expect(response.content[2]).toMatchObject({
        text: "Answer",
        citations: [
          { url: "https://a.example/post", citedText: "quote" },
        ],
      });
      expect(response.usage).toMatchObject({
        webSearchCount: 1,
        webSearchCost: 0.01,
        didWebSearch: true,
      });
      expect(response.usage?.inputCost).toBeCloseTo(computeResponseCost(300, 4));
    });
  });

  describe("history", () => {
    it("echoes Claude research pairs verbatim, drops OpenAI and unpaired ones, and strips citations", async () => {
      const adapter = new AnthropicAdapter(config, opus55);
      mockMessagesCreate.mockResolvedValue(asStream([]));

      await collect(adapter, {
        model: opus55.apiName,
        messages: [
          ...messages,
          {
            role: "assistant",
            content: [
              {
                type: "server_tool_use",
                id: "srvtoolu_1",
                name: "web_search",
                input: '{"query":"news"}',
                vendor: "anthropic",
              },
              {
                type: "web_search_tool_result",
                toolUseId: "srvtoolu_1",
                results: [],
                vendor: "anthropic",
                raw: searchResult,
              },
              {
                // Unpaired: no result block
                type: "server_tool_use",
                id: "srvtoolu_orphan",
                name: "web_fetch",
                input: '{"url":"https://x.example"}',
                vendor: "anthropic",
              },
              {
                type: "server_tool_use",
                id: "ws_1",
                name: "web_search",
                input: '{"query":"news"}',
                vendor: "openai",
              },
              {
                type: "web_search_tool_result",
                toolUseId: "ws_1",
                results: [{ url: "https://o.example" }],
                vendor: "openai",
              },
              {
                type: "text",
                text: "Answer",
                citations: [
                  {
                    type: "url_citation",
                    url: "https://a.example/post",
                    vendor: "anthropic",
                  },
                ],
              },
            ],
          },
          { role: "user", content: [{ type: "text", text: "More?" }] },
        ],
      });

      expect(mockMessagesCreate.mock.calls[0][0].messages[1]).toEqual({
        role: "assistant",
        content: [
          {
            type: "server_tool_use",
            id: "srvtoolu_1",
            name: "web_search",
            input: { query: "news" },
          },
          searchResult,
          { type: "text", text: "Answer" },
        ],
      });
    });
  });
});
