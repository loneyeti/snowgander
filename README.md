# Snowgander AI Vendor Abstraction Layer (`snowgander`)

[![npm version](https://badge.fury.io/js/snowgander.svg)](https://badge.fury.io/js/snowgander)

This TypeScript library provides a robust, maintainable, and extensible abstraction layer over various AI vendor APIs (OpenAI, Anthropic, Google AI, OpenRouter). It allows consuming applications to interact with different AI models through a standardized interface, simplifying integration and vendor switching.

## Features

- **Standardized Interface:** `AIVendorAdapter` defines a common contract for chat completion, image generation (where supported), and capability checks.
- **Vendor Abstraction:** Hides vendor-specific SDK details behind a consistent API.
- **Factory Pattern:** `AIVendorFactory` simplifies adapter creation based on vendor name and model configuration.
- **Configuration Injection:** Allows API keys and other vendor settings to be injected via `AIVendorFactory.setVendorConfig`.
- **Standardized Content:** Uses `ContentBlock[]` (`TextBlock`, `ImageBlock`, `ImageDataBlock`, `ThinkingBlock`, etc.) for consistent handling of diverse message content across vendors.
- **Capability Reporting:** Adapters report model capabilities (`isVisionCapable`, `isImageGenerationCapable`, `isThinkingCapable`) based on configuration.
- **Cost Calculation:** Provides utilities and standardized response fields (`UsageResponse`) for estimating interaction costs based on token usage.
- **Type Safety:** Written in TypeScript with shared type definitions.

## Current Limitations

- **No MCP Tool Handling:** The `sendMCPChat` method is **not implemented** in any adapter. Responsibility for defining tools for vendor APIs, handling tool use requests, executing tools, and sending results back currently lies entirely with the consuming application.
- **Inconsistent Multimodal Support:** Handling of image inputs varies between adapters. URL-based images (`ImageBlock`) are generally not automatically processed by adapters; base64-encoded images (`ImageDataBlock`) have better support but may require pre-processing by the consuming application for some vendors (e.g., Google). Check specific adapter capabilities.
- **Limited Image Generation:** Currently, only the OpenAI adapter supports image generation (`generateImage`).

## Installation

```bash
npm install snowgander
```

## Configuration

Before using the factory, configure the necessary API keys for each vendor you intend to use. This should typically happen once during application startup.

```typescript
import { AIVendorFactory } from "snowgander";

// Example configuration (likely done in a central setup file)
if (process.env.OPENAI_API_KEY) {
  AIVendorFactory.setVendorConfig("openai", {
    apiKey: process.env.OPENAI_API_KEY,
    organizationId: process.env.OPENAI_ORG_ID, // Optional
  });
}

if (process.env.ANTHROPIC_API_KEY) {
  AIVendorFactory.setVendorConfig("anthropic", {
    apiKey: process.env.ANTHROPIC_API_KEY,
  });
}

if (process.env.GOOGLE_API_KEY) {
  AIVendorFactory.setVendorConfig("google", {
    apiKey: process.env.GOOGLE_API_KEY,
  });
}

if (process.env.OPENROUTER_API_KEY) {
  // OpenRouter uses OpenAI's SDK structure but needs its own config
  AIVendorFactory.setVendorConfig("openrouter", {
    apiKey: process.env.OPENROUTER_API_KEY,
    // OpenRouter might require specific headers like 'HTTP-Referer' or 'X-Title'
    // Pass them via optionalHeaders if needed by your application context
    // optionalHeaders: { 'HTTP-Referer': 'YourAppUrl', 'X-Title': 'YourAppTitle' }
  });
}
```

## Usage

The primary interaction points for a consuming application are:

1.  **`AIVendorFactory`:** Used to configure vendor API keys (`setVendorConfig`) and retrieve adapter instances (`getAdapter`).
2.  **`ModelConfig`:** An object defining the specific model's capabilities, API name, and costs. This is typically fetched dynamically and passed to the factory.
3.  **`sendChat`:** The main method on the retrieved adapter for conducting chat conversations.

**Note:** While methods like `generateResponse`, `generateImage`, and `sendMCPChat` exist, the long-term goal is to consolidate most common interactions (including image generation and tool use) into the `sendChat` method for a simpler, unified interface.

Here's a typical workflow:

1.  **Import:** Import the factory and necessary types.

    ```typescript
    import {
      AIVendorFactory,
      ModelConfig,
      Chat, // Represents the conversation history
      ChatResponse, // The result from sendChat
      AIVendorAdapter,
      // Other types like ContentBlock, TextBlock, ImageDataBlock as needed
    } from "snowgander";
    ```

2.  **Prepare `ModelConfig`:** Create a `ModelConfig` object (e.g., fetched from a database).

    ```typescript
    // Example: Fetch model details first
    // const modelData = await getModelDetailsFromDB('gpt-4-turbo');
    const modelConfig: ModelConfig = {
      apiName: modelData.apiName, // e.g., "gpt-4-turbo"
      isVisionCapable: modelData.isVision,
      isImageGenerationCapable: modelData.isImageGeneration,
      isThinkingCapable: modelData.isThinking,
      inputTokenCost: modelData.inputTokenCost ?? undefined, // Cost per 1m tokens
      outputTokenCost: modelData.outputTokenCost ?? undefined, // Cost per 1m tokens
    };
    ```

3.  **Get Adapter:** Use the factory to get the appropriate adapter instance.

    ```typescript
    // Example: Fetch vendor name first
    // const vendorName = modelData.vendorName; // e.g., "openai"
    const adapter: AIVendorAdapter = AIVendorFactory.getAdapter(
      vendorName,
      modelConfig
    );
    ```

4.  **Use `sendChat`:** Manage conversation history and interact with the model.

    ```typescript
    // Example Chat object (manage this in your application state)
    let chat: Chat = {
      messages: [
        {
          role: "system",
          content: [{ type: "text", text: "You are a helpful assistant." }],
        },
        {
          role: "user",
          content: [
            { type: "text", text: "Hello! What can you do?" },
            // Add image content if model supports vision and adapter handles it:
            // { type: 'imageData', mimeType: 'image/png', data: 'base64encoded...' }
          ],
        },
      ],
      // Optional parameters can be added here or passed directly to sendChat
      // temperature: 0.7,
      // maxTokens: 150,
    };

    try {
      // Send the entire chat history to the adapter
      const chatResponse: ChatResponse = await adapter.sendChat(chat, {
        // Override or add parameters for this specific call if needed
        // temperature: 0.8,
      });

      // Update your chat history with the assistant's response
      chat.messages.push(chatResponse.message);

      console.log(
        "Assistant Response:",
        JSON.stringify(chatResponse.message.content, null, 2)
      );
      if (chatResponse.usage) {
        console.log("Estimated Cost:", chatResponse.usage.estimatedCost);
        console.log("Input Tokens:", chatResponse.usage.inputTokens);
        console.log("Output Tokens:", chatResponse.usage.outputTokens);
      }
    } catch (error) {
      console.error("Error sending chat:", error);
    }

    // --- Example: Image Generation (Current Method) ---
    // Note: This functionality is planned to be integrated into sendChat in the future.
    if (adapter.isImageGenerationCapable()) {
      const imagePrompt = "A cute cat wearing sunglasses";
      try {
        // The OpenAI adapter currently takes a simple prompt string.
        // Other adapters might require different parameters or not support it.
        const imageUrl: string = await adapter.generateImage(imagePrompt); // Returns data URI or URL
        console.log("Generated Image URL/Data:", imageUrl);
        // You might add this image URL/data as a message in your chat history
      } catch (error) {
        console.error("Error generating image:", error);
      }
    } else {
      console.log(
        `Model ${modelConfig.apiName} does not support image generation.`
      );
    }
    ```

## Development

- **Build:** Run `npm run build` to compile TypeScript to JavaScript in the `dist/` folder.
- **Testing:** Run `npm test` to execute tests using Jest.

## Exports

- `AIVendorFactory`: Class for creating adapter instances.
- Types: `AIVendorAdapter`, `AIRequestOptions`, `AIResponse`, `VendorConfig`, `ModelConfig`, `ContentBlock`, `ThinkingBlock`, `RedactedThinkingBlock`, `TextBlock`, `ImageBlock`, `ImageDataBlock`, `MCPTool`, `Message`.
