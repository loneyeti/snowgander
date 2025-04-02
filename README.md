# Snowgoose AI Vendors Package (`@snowgoose/ai-vendors`)

This package provides a set of reusable adapters for interacting with various AI vendor APIs (OpenAI, Anthropic, Google AI, OpenRouter) within the Snowgoose project ecosystem. It offers a standardized interface (`AIVendorAdapter`) and a factory (`AIVendorFactory`) for easy instantiation and use.

## Features

- **Standardized Interface:** `AIVendorAdapter` defines a common contract for chat completion, image generation (where supported), and capability checks.
- **Vendor Abstraction:** Hides vendor-specific SDK details behind a consistent API.
- **Factory Pattern:** `AIVendorFactory` simplifies adapter creation based on vendor name and model configuration.
- **Configuration:** Allows API keys and other vendor settings to be injected via `AIVendorFactory.setVendorConfig`.
- **Type Safety:** Written in TypeScript with shared type definitions (`ModelConfig`, `Chat`, `ChatResponse`, `ContentBlock`, etc.).

## Installation (within Snowgoose Monorepo)

This package is intended for use within the Snowgoose monorepo using npm workspaces.

1.  Ensure the root `package.json` includes `"workspaces": ["packages/*"]`.
2.  Ensure the root `package.json` includes `"@snowgoose/ai-vendors": "1.0.0"` (or the current version) in its `dependencies`.
3.  Run `npm install` in the root directory of the monorepo.

## Configuration

Before using the factory, configure the necessary API keys for each vendor you intend to use. This should typically happen once during application startup.

```typescript
import { AIVendorFactory } from "@snowgoose/ai-vendors";

// Example configuration (likely done in a central setup file like chat.repository.ts)
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

// Add configurations for Google, OpenRouter similarly...
```

## Usage

1.  **Import:** Import the factory and necessary types.
    ```typescript
    import {
      AIVendorFactory,
      ModelConfig,
      Chat,
      AIRequestOptions,
    } from "@snowgoose/ai-vendors";
    ```
2.  **Prepare `ModelConfig`:** Create a `ModelConfig` object containing the necessary details about the specific AI model being used (usually derived from database data).
    ```typescript
    // Example: Fetch Prisma model data first
    // const prismaModel = await prisma.model.findUnique(...);
    const modelConfig: ModelConfig = {
      apiName: prismaModel.apiName,
      isVision: prismaModel.isVision,
      isImageGeneration: prismaModel.isImageGeneration,
      isThinking: prismaModel.isThinking,
      inputTokenCost: prismaModel.inputTokenCost ?? undefined,
      outputTokenCost: prismaModel.outputTokenCost ?? undefined,
    };
    ```
3.  **Get Adapter:** Use the factory to get the appropriate adapter instance.
    ```typescript
    // Example: Fetch vendor name first
    // const vendorName = apiVendor.name;
    const adapter = AIVendorFactory.getAdapter(vendorName, modelConfig);
    ```
4.  **Use Adapter Methods:** Call methods like `sendChat`, `generateImage`, or `generateResponse`.

    ```typescript
    // Using sendChat (simplified)
    const chatData: Chat = {
      /* ... populate chat data ... */
    };
    const response: ChatResponse = await adapter.sendChat(chatData);

    // Using generateResponse (more granular control)
    const requestOptions: AIRequestOptions = {
      model: modelConfig.apiName,
      messages: [
        /* ... message history ... */
      ],
      systemPrompt: "Your system prompt here",
      // ... other options
    };
    const aiResponse: AIResponse =
      await adapter.generateResponse(requestOptions);

    // Using generateImage (for capable models)
    const imageUrl: string = await adapter.generateImage(chatData); // Returns data URI or URL
    ```

## Development

- **Build:** Run `npm run build` within this package directory (`packages/ai-vendors`) to compile TypeScript to JavaScript in the `dist/` folder.
- **Testing:** (Add details if tests are implemented)

## Exports

- `AIVendorFactory`: Class for creating adapter instances.
- Types: `AIVendorAdapter`, `AIRequestOptions`, `AIResponse`, `VendorConfig`, `ModelConfig`, `Chat`, `ChatResponse`, `ContentBlock`, `ThinkingBlock`, `RedactedThinkingBlock`, `TextBlock`, `ImageBlock`, `ImageDataBlock`, `MCPTool`, `Message`.
