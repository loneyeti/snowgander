# Product Context: AI Vendor Abstraction Layer

## Problem Solved

Integrating various third-party AI vendor APIs (like OpenAI, Anthropic, Google) directly into an application leads to:
*   **Tight Coupling:** The application becomes heavily dependent on specific vendor SDKs and API structures.
*   **Difficult Switching:** Changing AI vendors or even models within a vendor requires significant code refactoring.
*   **Inconsistent Handling:** Different vendors have different ways of representing requests, responses, capabilities (like vision or thinking steps), and content types (text, images). Handling this inconsistency directly in the application logic is complex and error-prone.
*   **Complex Configuration:** Managing API keys and model-specific details for multiple vendors can become cumbersome.
*   **Lack of Standardized Cost Tracking:** Calculating and comparing costs across different vendor pricing models is difficult.

## How It Works (Solution)

This library provides a unified abstraction layer to address these problems:

1.  **Standard Interface (`AIVendorAdapter`):** Defines a consistent contract for interacting with any supported AI vendor. All vendor-specific logic is encapsulated within adapters implementing this interface.
2.  **Factory (`AIVendorFactory`):** Simplifies the creation of vendor adapters. The application requests an adapter by name, and the factory, using injected configuration (`VendorConfig`, `ModelConfig`), returns the appropriate, configured instance.
3.  **Unified Content (`ContentBlock[]`):** All message content (input and output) is represented using a standardized array of `ContentBlock` objects (`TextBlock`, `ImageDataBlock`, `ThinkingBlock`, etc.). This allows the application to handle diverse content types consistently, regardless of the underlying vendor. Adapters are responsible for translating between this internal format and the vendor-specific format.
4.  **Abstracted Configuration (`VendorConfig`, `ModelConfig`):** Vendor API keys and model details (capabilities, costs, API names) are managed through simple configuration objects, decoupling them from the application code.
5.  **Capability Flags:** Adapters expose boolean flags (`isVisionCapable`, `isImageGenerationCapable`, `isThinkingCapable`) allowing the application to query model capabilities before attempting specific operations.
6.  **Cost Calculation:** Provides a utility (`computeResponseCost`) and integrates cost calculation into adapter responses (`UsageResponse`) based on token counts reported by the vendor and configured costs per token, offering a standardized way to estimate interaction costs.

## User Experience Goals (for the Developer using the Library)

*   **Simplicity:** Easily integrate and use different AI models with minimal boilerplate code.
*   **Flexibility:** Switch between AI vendors or models with only configuration changes, not code changes.
*   **Consistency:** Interact with all vendors through the same methods and data structures.
*   **Transparency:** Understand model capabilities and estimated costs through the standardized interface.
*   **Maintainability:** Isolate vendor-specific logic, making the consuming application easier to maintain and update.
