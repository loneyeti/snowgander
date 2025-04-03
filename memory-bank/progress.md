# Progress: AI Vendor Abstraction Layer (2025-04-03)

## What Works

*   **Core Abstraction:** Factory (`AIVendorFactory`) and Adapter (`AIVendorAdapter`) patterns are implemented.
*   **Basic Chat:** Standard chat completion (`generateResponse`, `sendChat`) is functional for all implemented vendors (OpenAI, Anthropic, Google, OpenRouter), primarily handling text-based interactions.
*   **Configuration:** Adapters are instantiated with `VendorConfig` (API keys, etc.) and `ModelConfig` (capabilities, costs, API name).
*   **Content Blocks (Basic):** The `ContentBlock[]` system is defined in `types.ts` and used for structuring messages and responses. Adapters perform basic mapping for `TextBlock`.
*   **Capability Flags:** Adapters correctly report `isVisionCapable`, `isImageGenerationCapable`, `isThinkingCapable` based on `ModelConfig`.
*   **Cost Calculation:** Basic cost estimation based on reported token counts and configured costs is implemented using `computeResponseCost` and returned in `UsageResponse`.
*   **Vendor Support:** Adapters exist for OpenAI, Anthropic, Google AI, and OpenRouter.
*   **Image Generation (OpenAI):** `generateImage` is implemented for the OpenAI adapter (DALL-E 3).
*   **Vision Input (Basic):**
    *   Google adapter handles `ImageDataBlock` (base64).
    *   OpenAI adapter handles `imageData` (base64) by converting it to `image_url` format.
*   **Thinking Blocks (Anthropic):** Anthropic adapter handles sending and receiving `ThinkingBlock` content.

## What's Left to Build / TODOs

*   **MCP Tool Support:**
    *   **Major Gap:** `sendMCPChat` is **not implemented** in any adapter. All adapters currently throw errors. The design delegates full MCP lifecycle management (tool definition, execution, result handling) to the consuming application. This needs to be addressed either by implementing it in the adapters or confirming the delegation strategy is acceptable.
    *   OpenAI adapter needs mapping for tool call outputs (`TODO` comment).
    *   OpenRouter adapter requires investigation into whether tool/function calling parameters are passed through (`TODO` comment).
*   **Enhanced Content Block Mapping:**
    *   OpenRouter adapter needs proper mapping for complex content blocks (vision, etc.) beyond simple text extraction (`TODO` comment).
    *   Google adapter requires the *calling application* to pre-process URL-based images (`ImageBlock`) into base64 (`ImageDataBlock`) before sending them to the adapter (logged warning). The adapter itself doesn't fetch URLs.
*   **Refinement & Testing:**
    *   Add comprehensive unit and integration tests for all adapters and functionalities.

## Current Status

*   **Foundation Built:** The core abstraction layer, type definitions, factory, and basic adapter implementations are in place.
*   **Memory Bank Initialized:** Core documentation files have been created based on the current codebase (as of 2025-04-03).
*   **Ready for Feature Enhancement:** The library provides basic chat and some modality support, but requires significant work to fully support advanced features like MCP tools and robust multimodal handling across all vendors.

## Known Issues / Limitations

*   **No MCP Tool Handling:** As mentioned, this is the most significant functional gap. The library currently cannot manage tool interactions with vendor APIs.
*   **Inconsistent Multimodal Support:** Handling of image inputs varies: Google requires pre-processed base64, OpenAI handles base64, Anthropic doesn't support vision, OpenRouter's support is basic/untested. URL-based images (`ImageBlock`) are not automatically handled by adapters.
*   **Limited Image Generation:** Only OpenAI adapter supports image generation.
*   **Potential OpenAI Endpoint Issue:** Use of `client.responses.create` in `OpenAIAdapter` might be suboptimal.
