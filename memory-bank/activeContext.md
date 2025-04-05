# Active Context: Memory Bank Update (2025-04-05)

## Current Focus

- **Task:** Update the Memory Bank for the AI Vendor Abstraction Layer project (`snowgander`) based on review of `src/types.ts` and `src/factory.ts`.
- **Action:** Reviewing key files in the `src/` directory with particular attention to `types.ts` (defining the core interfaces and data structures) and `factory.ts` (implementing the adapter factory pattern).
- **Goal:** Ensure Memory Bank accurately reflects the current implementation, with special focus on the type system and factory pattern implementation.

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

- **Adapter Evolution:** Examining the Anthropic adapter shows:

  - Tools are handled directly in `sendChat` and `generateResponse` methods
  - The `sendMCPChat` method no longer exists in the Anthropic adapter, suggesting these functions are now integrated
  - Detailed mapping of content blocks between our system format and Anthropic's API format

- **Index Exports:** The `index.ts` file cleanly exports:
  - The factory as the primary interface for consuming applications
  - All necessary types consumers might need
  - An option (commented out) to export individual adapters if needed

## Next Steps

1.  Update `progress.md` to reflect the current state of implementation, noting any changes to tool handling and content block processing.
2.  Consider reviewing the implementation of other adapters (OpenAI, Google, OpenRouter) to verify consistent patterns.
3.  Document any evolving patterns in how the library handles MCP tools, especially since the direct `sendMCPChat` method approach appears to have been replaced.
