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

1.  **Import:** Import the factory and necessary types.

    ```typescript
    import {
      AIVendorFactory,
      ModelConfig,
      Chat, // Assuming Chat type is defined in your application or another shared package
      AIRequestOptions,
      AIResponse,
      ChatResponse, // Assuming ChatResponse is defined elsewhere or use AIResponse
      AIVendorAdapter,
    } from "snowgander";
    ```

    _Note: `Chat` and `ChatResponse` were mentioned in the previous README but are not explicitly exported by this library according to `src/index.ts`. You might need to define these or use the exported `Message[]` and `AIResponse` types._

2.  **Prepare `ModelConfig`:** Create a `ModelConfig` object containing the necessary details about the specific AI model being used (e.g., fetched from a database or configuration file).
    ```typescript
    // Example: Fetch model details first
    // const modelData = await getModelDetailsFromDB('gpt-4-turbo');
    const modelConfig: ModelConfig = {
      apiName: modelData.apiName, // e.g., "gpt-4-turbo"
      isVisionCapable: modelData.isVision,
      isImageGenerationCapable: modelData.isImageGeneration,
      isThinkingCapable: modelData.isThinking,
      inputTokenCost: modelData.inputTokenCost ?? undefined, // Cost per 1k tokens
      outputTokenCost: modelData.outputTokenCost ?? undefined, // Cost per 1k tokens
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
4.  **Use Adapter Methods:** Call methods like `generateResponse` or `generateImage`.

    ```typescript
    // Using generateResponse (granular control)
    const requestOptions: AIRequestOptions = {
      model: modelConfig.apiName,
      messages: [
        {
          role: "system",
          content: [{ type: "text", text: "You are a helpful assistant." }],
        },
        { role: "user", content: [{ type: "text", text: "Hello, world!" }] },
        // Add image content if model supports vision and adapter handles it:
        // { role: 'user', content: [{ type: 'imageData', mimeType: 'image/png', data: 'base64encoded...' }] }
      ],
      // Optional parameters:
      // temperature: 0.7,
      // maxTokens: 100,
      // stream: false, // Streaming is not explicitly supported by all adapters yet
    };
    try {
      const aiResponse: AIResponse = await adapter.generateResponse(
        requestOptions
      );
      console.log("AI Response:", JSON.stringify(aiResponse.content, null, 2));
      if (aiResponse.usage) {
        console.log("Estimated Cost:", aiResponse.usage.estimatedCost);
        console.log("Input Tokens:", aiResponse.usage.inputTokens);
        console.log("Output Tokens:", aiResponse.usage.outputTokens);
      }
    } catch (error) {
      console.error("Error generating response:", error);
    }

    // Using generateImage (check capability first!)
    if (adapter.isImageGenerationCapable()) {
      const imagePrompt = "A futuristic cityscape at sunset";
      try {
        // Note: generateImage API might evolve; current adapters might expect different inputs.
        // The OpenAI adapter currently takes a simple prompt string.
        const imageUrl: string = await adapter.generateImage(imagePrompt); // Returns data URI or URL
        console.log("Generated Image URL/Data:", imageUrl);
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
