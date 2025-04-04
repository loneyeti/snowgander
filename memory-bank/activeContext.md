# Active Context: Memory Bank Initialization (2025-04-03)

## Current Focus

- **Task:** Initialize and populate the Memory Bank for the AI Vendor Abstraction Layer project (`snowgander`).
- **Action:** Reviewing all files in the `src/` directory (`factory.ts`, `index.ts`, `types.ts`, `utils.ts`, and all vendor adapters in `src/vendors/`) to understand the current state, architecture, patterns, and technical details.
- **Goal:** Create the core Memory Bank documents (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`) to establish a baseline understanding for future work.

## Key Observations During Review

- **Content Blocks:** The system heavily relies on the `ContentBlock[]` structure defined in `types.ts` for representing message content consistently across different vendors. This includes text, images (URL and base64 data), and thinking steps. Adapters are responsible for mapping to/from this structure.
- **Cost Calculation:** A utility `computeResponseCost` exists and is used by adapters to calculate costs based on token usage reported by vendor APIs and configured per-token costs (`inputTokenCost`, `outputTokenCost` in `ModelConfig`). Results are returned in the `UsageResponse` object.
- **MCP Tool Handling:** A recurring pattern is that the `sendMCPChat` method in _all_ adapters currently throws an error. The comments explicitly state that the responsibility for managing the MCP tool lifecycle (defining tools for the API, handling tool use requests, executing tools, sending back results) lies with the _consuming application_, not the adapters themselves. This is a significant design decision and limitation of the current adapters.
- **Unimplemented/TODOs:**
  - OpenAI adapter needs mapping for non-text output types (like tool calls) (`TODO` comment).
  - OpenRouter adapter needs proper mapping for complex content blocks (vision/other types) (`TODO` comment).
  - OpenRouter adapter needs investigation regarding tool/function calling pass-through (`TODO` comment).
  - Google adapter needs pre-processing for URL-based images (`image` block type) into base64 `inlineData` before passing to the adapter (logged warning).
- **Strict ContentBlocks:** The user emphasized the strict use of `ContentBlocks` for all responses, which seems aligned with the current implementation in `types.ts` and the adapters' attempts to map to/from this structure.

## Next Steps

1.  Complete the creation of the `progress.md` file, summarizing the implemented features and known limitations/TODOs.
2.  Signal completion of the Memory Bank update task.
