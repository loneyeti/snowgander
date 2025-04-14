# Active Context: Memory Bank Update (2025-04-05)

## Current Focus

- **Task:** Implement OpenRouter reasoning token support.
- **Action:** Modified `src/vendors/openrouter.ts` and `src/vendors/__tests__/openrouter.test.ts`.
- **Goal:** Allow users to leverage OpenRouter's reasoning capabilities using the existing `budgetTokens` parameter and `isThinking` flag, maintaining vendor neutrality in the core interfaces.

## Key Observations During Review

- **Type System:** The `types.ts` file defines a comprehensive type system with:

  - `ContentBlock` union type (including `TextBlock`, `ImageBlock`, `ImageDataBlock`, `ThinkingBlock`, etc.) for standardized message content representation
  - Clear interfaces for adapters (`AIVendorAdapter`), configurations (`VendorConfig`, `ModelConfig`), and messaging (`Message`, `Chat`, `ChatResponse`)
  - Robust definition of tool-related types (`ToolUseBlock`, `ToolResultBlock`, `MCPAvailableTool`)
  - Explicit capability flags in the `AIVendorAdapter` interface

- **Factory Implementation:** The `factory.ts` file implements a clean factory pattern:

  - Maintains a static `vendorConfigs` Map to store configurations
  - Uses the `getAdapter(vendorName, modelConfig)` method to instantiate vendor-specific adapters
  - Ensures proper configuration injection into adapters via constructor parameters
  - Supports multiple vendors (OpenAI, Anthropic, Google, OpenRouter)

- **Adapter Evolution:**

  - **Anthropic:** Integrates tool handling directly in `sendChat`/`generateResponse`. Uses `thinkingMode` and `budgetTokens` for its specific thinking implementation.
  - **OpenRouter:** Now supports reasoning tokens. It uses the `isThinking` capability flag and the `budgetTokens` parameter (passed as `reasoning: { max_tokens: ... }` to the API). Reasoning output is parsed into a `ThinkingBlock`. This aligns OpenRouter's reasoning with Anthropic's thinking mechanism using shared interface fields (`isThinking`, `budgetTokens`, `ThinkingBlock`).

  - Tools are handled directly in `sendChat` and `generateResponse` methods
  - Detailed mapping of content blocks between our system format and vendor API formats continues to be refined.

- **Index Exports:** The `index.ts` file cleanly exports:
  - The factory as the primary interface for consuming applications
  - All necessary types consumers might need
  - An option (commented out) to export individual adapters if needed

## Next Steps

1.  Update `progress.md` to reflect the addition of OpenRouter reasoning support and the unified approach using `isThinking`/`budgetTokens`.
2.  Update `systemPatterns.md` to clarify the unified handling of thinking/reasoning.
3.  Update `.clinerules` with the pattern of using `isThinking`/`budgetTokens` for both Anthropic and OpenRouter.
4.  Run tests (`npm test`) to confirm changes.
5.  Attempt completion.
