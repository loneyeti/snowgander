import { OpenAIAdapter } from "../openai";
import {
  AIRequestOptions,
  ContentBlock,
  Message,
  ModelConfig,
  VendorConfig,
} from "../../types";

const mockResponsesCreate = jest.fn();
jest.mock("openai", () =>
  jest.fn().mockImplementation(() => ({
    responses: { create: mockResponsesCreate },
  }))
);

describe("OpenAIAdapter web search", () => {
  const config: VendorConfig = { apiKey: "test-openai-key" };
  const modelConfig: ModelConfig = {
    apiName: "gpt-6",
    isVision: false,
    isImageGeneration: false,
    isThinking: false,
    inputTokenCost: 1, // Per million tokens
    outputTokenCost: 2,
    webSearchCost: 0.01,
  };
  const messages: Message[] = [
    { role: "user", content: [{ type: "text", text: "What's new?" }] },
  ];
  let adapter: OpenAIAdapter;

  beforeEach(() => {
    jest.clearAllMocks();
    adapter = new OpenAIAdapter(config, modelConfig);
  });

  const collect = async (options: AIRequestOptions) => {
    const blocks: ContentBlock[] = [];
    for await (const block of adapter.streamResponse(options)) {
      blocks.push(block);
    }
    return blocks;
  };

  const searchCall = (id: string, query: string, sources?: string[]) => ({
    type: "web_search_call",
    id,
    status: "completed",
    action: {
      type: "search",
      query,
      ...(sources && {
        sources: sources.map((url) => ({ type: "url", url })),
      }),
    },
  });

  describe("request", () => {
    it("builds the web_search tool from webSearch and requests sources", async () => {
      async function* empty() {}
      mockResponsesCreate.mockResolvedValue(empty());

      await collect({
        model: "gpt-6",
        messages,
        webSearch: {
          allowedDomains: ["example.com"],
          userLocation: { country: "US", city: "Denver" },
          fetch: true, // Claude only; ignored here
        },
      });

      const args = mockResponsesCreate.mock.calls[0][0];
      expect(args.tools).toEqual([
        {
          type: "web_search",
          filters: { allowed_domains: ["example.com"] },
          user_location: { type: "approximate", city: "Denver", country: "US" },
        },
      ]);
      expect(args.include).toEqual(["web_search_call.action.sources"]);
    });

    it("upgrades a legacy web_search_preview tool to web_search", async () => {
      mockResponsesCreate.mockResolvedValue({
        id: "resp_1",
        output_text: "hi",
        output: [],
      });

      await adapter.generateResponse({
        model: "gpt-6",
        messages,
        tools: [{ type: "web_search_preview", search_context_size: "low" }],
      });

      const args = mockResponsesCreate.mock.calls[0][0];
      expect(args.tools).toEqual([
        { type: "web_search", search_context_size: "low" },
      ]);
      expect(args.include).toEqual(["web_search_call.action.sources"]);
    });

    it("does not add include without web search", async () => {
      async function* empty() {}
      mockResponsesCreate.mockResolvedValue(empty());

      await collect({ model: "gpt-6", messages });

      const args = mockResponsesCreate.mock.calls[0][0];
      expect(args.tools).toEqual([]);
      expect(args).not.toHaveProperty("include");
    });

    it("drops research blocks and citations from history", async () => {
      async function* empty() {}
      mockResponsesCreate.mockResolvedValue(empty());

      await collect({
        model: "gpt-6",
        messages: [
          ...messages,
          {
            role: "assistant",
            content: [
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
                results: [{ url: "https://a.example" }],
                vendor: "openai",
              },
              {
                type: "text",
                text: "Answer",
                citations: [
                  {
                    type: "url_citation",
                    url: "https://a.example",
                    vendor: "openai",
                  },
                ],
              },
            ],
          },
          { role: "user", content: [{ type: "text", text: "More?" }] },
        ],
      });

      const args = mockResponsesCreate.mock.calls[0][0];
      expect(args.input[1]).toEqual({
        role: "assistant",
        content: [{ type: "output_text", text: "Answer" }],
      });
    });
  });

  describe("streamResponse", () => {
    it("yields research steps, sources, and citations instead of fake thinking", async () => {
      async function* stream() {
        yield {
          type: "response.output_item.added",
          item: { type: "web_search_call", id: "ws_1", status: "in_progress" },
        };
        yield { type: "response.web_search_call.searching", item_id: "ws_1" };
        yield {
          type: "response.output_item.done",
          item: searchCall("ws_1", "latest news", [
            "https://a.example/1",
            "https://b.example/2",
          ]),
        };
        yield { type: "response.output_text.delta", delta: "News here." };
        yield {
          type: "response.output_text.annotation.added",
          annotation: {
            type: "url_citation",
            url: "https://a.example/1",
            title: "A",
            start_index: 0,
            end_index: 10,
          },
        };
        yield {
          type: "response.completed",
          response: {
            id: "resp_1",
            usage: { input_tokens: 100, output_tokens: 50 },
            output: [
              searchCall("ws_1", "latest news"),
              {
                type: "web_search_call",
                id: "ws_2",
                status: "completed",
                action: { type: "open_page", url: "https://a.example/1" },
              },
            ],
          },
        };
      }
      mockResponsesCreate.mockResolvedValue(stream());

      const blocks = await collect({ model: "gpt-6", messages, webSearch: {} });

      expect(blocks.filter((b) => b.type === "thinking")).toEqual([]);
      expect(blocks.slice(0, 5)).toEqual([
        {
          type: "server_tool_use",
          id: "ws_1",
          name: "web_search",
          input: "{}",
          status: "in_progress",
          vendor: "openai",
        },
        {
          type: "server_tool_use",
          id: "ws_1",
          name: "web_search",
          input: '{"query":"latest news"}',
          status: "completed",
          vendor: "openai",
        },
        {
          type: "web_search_tool_result",
          toolUseId: "ws_1",
          results: [
            { url: "https://a.example/1" },
            { url: "https://b.example/2" },
          ],
          vendor: "openai",
        },
        { type: "text", text: "News here." },
        {
          type: "text",
          text: "",
          citations: [
            {
              type: "url_citation",
              url: "https://a.example/1",
              title: "A",
              startIndex: 0,
              endIndex: 10,
              vendor: "openai",
            },
          ],
        },
      ]);

      // One search plus one page open: only the search is billed
      const meta = blocks[blocks.length - 1];
      expect(meta.type).toBe("meta");
      if (meta.type !== "meta") return;
      expect(meta.usage?.webSearchCount).toBe(1);
      expect(meta.usage?.webSearchCost).toBeCloseTo(0.01);
      expect(meta.usage?.didWebSearch).toBe(true);
      expect(meta.usage?.totalCost).toBeCloseTo(0.0001 + 0.0001 + 0.01);
    });

    it("maps open_page actions to a url input", async () => {
      async function* stream() {
        yield {
          type: "response.output_item.done",
          item: {
            type: "web_search_call",
            id: "ws_3",
            status: "completed",
            action: { type: "open_page", url: "https://c.example" },
          },
        };
      }
      mockResponsesCreate.mockResolvedValue(stream());

      const blocks = await collect({ model: "gpt-6", messages, webSearch: {} });

      expect(blocks).toEqual([
        {
          type: "server_tool_use",
          id: "ws_3",
          name: "web_search",
          input: '{"url":"https://c.example"}',
          status: "completed",
          vendor: "openai",
        },
      ]);
    });
  });

  describe("generateResponse", () => {
    it("returns research steps and a cited text block with offsets into the joined text", async () => {
      mockResponsesCreate.mockResolvedValue({
        id: "resp_2",
        output_text: "First. Second.",
        usage: { input_tokens: 10, output_tokens: 10 },
        output: [
          searchCall("ws_1", "q1", ["https://a.example"]),
          searchCall("ws_2", "q2"),
          {
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: "First. ", annotations: [] },
              {
                type: "output_text",
                text: "Second.",
                annotations: [
                  {
                    type: "url_citation",
                    url: "https://a.example",
                    title: "A",
                    start_index: 0,
                    end_index: 7,
                  },
                  { type: "file_citation", file_id: "f" },
                ],
              },
            ],
          },
        ],
      });

      const response = await adapter.generateResponse({
        model: "gpt-6",
        messages,
        webSearch: {},
      });

      expect(response.content.map((b) => b.type)).toEqual([
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "text",
      ]);
      expect(response.content[3]).toEqual({
        type: "text",
        text: "First. Second.",
        citations: [
          {
            type: "url_citation",
            url: "https://a.example",
            title: "A",
            startIndex: 7,
            endIndex: 14,
            vendor: "openai",
          },
        ],
      });
      expect(response.usage?.webSearchCount).toBe(2);
      expect(response.usage?.webSearchCost).toBeCloseTo(0.02);
    });
  });
});
