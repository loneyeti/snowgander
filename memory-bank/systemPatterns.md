# System Patterns: AI Vendor Abstraction Layer

## Core Architecture: Adapter Pattern + Factory Pattern

1.  **Adapter Pattern:**

    - **Interface:** `AIVendorAdapter` defines the standard contract for all interactions (e.g., `generateResponse`, `sendChat`, `generateImage`, `sendMCPChat`).
    - **Adapters:** Concrete classes (`OpenAIAdapter`, `AnthropicAdapter`, `GoogleAIAdapter`, `OpenRouterAdapter`) implement the `AIVendorAdapter` interface. Each adapter encapsulates the logic specific to a particular vendor's API and SDK, translating between the library's standard format (`AIRequestOptions`, `AIResponse`, `ContentBlock[]`) and the vendor's specific format.
    - **Client:** The consuming application interacts _only_ with the `AIVendorAdapter` interface, remaining unaware of the specific vendor implementation details.

2.  **Factory Pattern:**
    - **Factory:** `AIVendorFactory` is responsible for creating instances of the concrete adapter classes.
    - **Configuration:** The factory uses externally provided configuration (`VendorConfig` for API keys/URLs, `ModelConfig` for model-specific details like API name, capabilities, costs) stored in a static map (`vendorConfigs`).
    - **Instantiation:** The `getAdapter(vendorName, modelConfig)` method looks up the configuration and returns the appropriate, initialized adapter instance based on the requested `vendorName` and `modelConfig`. This decouples the client from the instantiation logic.

## Key Technical Decisions

- **TypeScript:** Used for strong typing, improved maintainability, and better developer experience.
- **Standardized Content Blocks (`ContentBlock[]`):** A union type (`ContentBlock`) encompassing various structures (`TextBlock`, `ImageDataBlock`, `ThinkingBlock`, `ImageBlock`, `RedactedThinkingBlock`) is used _consistently_ for representing content in both requests (`Message.content`) and responses (`AIResponse.content`, `ChatResponse.content`). This forces adapters to handle diverse content types and provides a uniform structure for the consuming application.
- **Configuration Injection:** Vendor and model configurations (`VendorConfig`, `ModelConfig`) are passed _into_ the factory and adapters, rather than being hardcoded or read directly from environment variables within the library. This promotes flexibility and testability.
- **Capability Flags:** Adapters explicitly declare their capabilities (`isVisionCapable`, `isImageGenerationCapable`, `isThinkingCapable`) based on the `ModelConfig`, allowing the consuming application to make informed decisions. The `isThinkingCapable` flag is used for both Anthropic's thinking and OpenRouter's reasoning features.
- **Cost Calculation Utility:** A separate `computeResponseCost` utility function standardizes cost calculation based on token counts and per-token costs. Adapters use this utility to populate the `UsageResponse` in their results.
- **Vendor SDKs:** Each adapter leverages the official SDK for its respective vendor (e.g., `@anthropic-ai/sdk`, `@google/generative-ai`, `openai`).
- **Error Handling:** Adapters are expected to throw errors for unsupported operations (e.g., `generateImage` in Anthropic) or unimplemented features (e.g., `sendMCPChat` in most current adapters).
- **Unified Thinking/Reasoning:** Both Anthropic and OpenRouter adapters utilize the `isThinkingCapable` flag and the `budgetTokens` parameter (from the `Chat` interface) to control their respective thinking/reasoning features. The output is consistently represented using the `ThinkingBlock` content type.

## Component Relationships

```mermaid
graph TD
    subgraph ConsumingApplication ["Consuming Application"]
        AppCode["Application Code"]
    end

    subgraph AbstractionLayer ["AI Vendor Abstraction Library (src/)"]
        Factory[("AIVendorFactory")] -- Creates --> AdaptersImplement
        Index["index.ts (Exports)"] -- Exports --> Factory
        Index -- Exports --> Types
        Index -- Exports --> AdapterInterface

        subgraph TypesAndUtils ["Types & Utilities"]
            Types["types.ts <br> (AIVendorAdapter, AIRequestOptions, AIResponse, ContentBlock[], VendorConfig, ModelConfig, etc.)"]
            Utils["utils.ts <br> (computeResponseCost)"]
        end

        subgraph Adapters ["Vendor Adapters (src/vendors/)"]
            direction LR
            AdapterInterface[/"AIVendorAdapter"/]
            AdaptersImplement(Adapters <br> OpenAIAdapter <br> AnthropicAdapter <br> GoogleAIAdapter <br> OpenRouterAdapter) -- Implements --> AdapterInterface
            AdaptersImplement -- Uses --> VendorSDKs[("Vendor SDKs <br> @openai/api <br> @anthropic-ai/sdk <br> @google/generative-ai")]
            AdaptersImplement -- Uses --> Types
            AdaptersImplement -- Uses --> Utils
        end
    end

    AppCode -- Uses --> Factory
    AppCode -- Uses --> AdapterInterface
    AppCode -- Uses --> Types

    Factory -- Uses --> Types
    Factory -- Creates --> AdaptersImplement

    %% Styling
    classDef subgraph fill:#f9f,stroke:#333,stroke-width:2px;
    class AbstractionLayer,ConsumingApplication subgraph;
```

## Design Considerations

- **MCP Tool Handling (Evolving):** The approach to MCP tool handling is evolving. While initially designed for the consuming application to manage the full lifecycle, adapters are beginning to integrate tool support directly within their core methods (`sendChat`, `generateResponse`). For example, the `AnthropicAdapter` now formats MCP tools for the Anthropic API. The goal is to achieve a consistent tool handling pattern across all adapters, simplifying usage for the consuming application. The original `sendMCPChat` method may become deprecated or removed if this integrated approach proves successful across all vendors.
- **Complex Content Mapping:** Adapters need to carefully map the internal `ContentBlock[]` structure to the specific formats required by each vendor API, especially for multimodal inputs (images). Some adapters (e.g., OpenRouter, Google) currently have basic text extraction or warnings for unhandled types, indicating areas for potential enhancement.
- **OpenAI Endpoint Choice:** The `OpenAIAdapter` uses `client.responses.create`, which appears less common than `client.chat.completions.create` used by `OpenRouterAdapter`. This might have implications for feature support (like tool calling) and requires careful handling of input/output types.
