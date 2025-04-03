# Tech Context: AI Vendor Abstraction Layer

## Core Technology

*   **Language:** TypeScript
*   **Environment:** Node.js (implied, as it's a TS library likely consumed by a Node backend)
*   **Module System:** ES Modules (`import`/`export` syntax observed)

## Key Dependencies

*   **Vendor SDKs:**
    *   `@anthropic-ai/sdk`: For Anthropic adapter.
    *   `@google/generative-ai`: For Google AI (Gemini) adapter.
    *   `openai`: For OpenAI and OpenRouter adapters. (Note: OpenRouter uses the OpenAI SDK structure).
*   **Build Tools:**
    *   `typescript` (tsc): For compiling TypeScript to JavaScript (inferred from `tsconfig.json` presence).

## Development Setup

*   **Configuration:** Requires `tsconfig.json` for TypeScript compilation settings.
*   **Package Management:** Uses `npm` (inferred from `package.json` and `package-lock.json`).
*   **Structure:** Source code is organized under `src/`, with vendor-specific adapters in `src/vendors/`. Core types, the factory, and utilities are at the root of `src/`.

## Technical Constraints & Considerations

*   **SDK Compatibility:** Relies on the specific APIs and data structures exposed by each vendor's SDK. Changes in vendor SDKs may require adapter updates.
*   **Content Block Mapping:** The accuracy and completeness of mapping the internal `ContentBlock[]` structure to each vendor's required format is crucial, especially for multimodal inputs (images) and specialized outputs (thinking steps, tool calls).
*   **Environment Variables:** While the library itself promotes configuration injection, the *consuming application* will be responsible for managing API keys (likely via environment variables) and passing them into the `VendorConfig`.
*   **OpenAI Endpoint:** The `OpenAIAdapter` currently uses `client.responses.create`. This might differ in capabilities or stability compared to the more standard `client.chat.completions.create` endpoint used by the `OpenRouterAdapter`.
