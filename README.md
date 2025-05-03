````markdown
# Snowgander: AI Vendor Abstraction Layer

[![npm version](https://badge.fury.io/js/snowgander.svg)](https://badge.fury.io/js/snowgander)

---

**Check out Snowgoose!**

Snowgander is the core abstraction engine powering [**Snowgoose**](https://snowgoose.app), an intelligent agent framework.

If you're building AI-powered applications, agents, or workflows, **Snowgoose** provides the tools and structure you need on top of Snowgander.

**Explore the Snowgoose App:** [**snowgoose.app**](https://snowgoose.app)
**Check out the Snowgoose Repo:** [**github.com/loneyeti/snowgoose**](https://github.com/loneyeti/snowgoose)

---

## What is Snowgander?

Snowgander is a TypeScript library that makes it easy to talk to different AI models (like those from OpenAI, Anthropic, Google, OpenRouter) using one simple, consistent interface.

Stop writing vendor-specific code! Use Snowgander to:

- Easily switch between AI models and vendors.
- Use a single method (`sendChat`) for most interactions.
- Handle different types of content (text, images, tool use, thinking steps) uniformly.

## How to Use It: The Basics

Using Snowgander involves these simple steps:

1.  **Configure API Keys:** Tell the factory your API keys.
2.  **Get an Adapter:** Ask the factory for an adapter for the specific model you want.
3.  **Send a Chat:** Use the adapter's `sendChat` method, passing the conversation state.
4.  **Get the Response:** Receive a standardized response containing the AI's message and usage info.

### 1. Installation

```bash
npm install snowgander
```
````

### 2. Configuration (Set API Keys)

Do this once when your application starts.

```typescript
import { AIVendorFactory } from "snowgander";

// Example: Load keys from environment variables
if (process.env.OPENAI_API_KEY) {
  AIVendorFactory.setVendorConfig("openai", {
    apiKey: process.env.OPENAI_API_KEY,
  });
  // Also configure 'openai-image' if using DALL-E via OpenAIImageAdapter
  AIVendorFactory.setVendorConfig("openai-image", {
    apiKey: process.env.OPENAI_API_KEY,
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
  AIVendorFactory.setVendorConfig("openrouter", {
    apiKey: process.env.OPENROUTER_API_KEY,
  });
}

// Add other vendors as needed...
```

### 3. How to Initiate an Adapter

To get an adapter, you need:

- The **vendor name** (e.g., `"openai"`, `"anthropic"`).
- A **`ModelConfig` object** describing the model. This usually comes from your application's settings or database.

```typescript
import {
  AIVendorFactory,
  ModelConfig, // Describes the specific model
  AIVendorAdapter, // The type for the adapter instance
} from "snowgander";

// --- Prepare ModelConfig (Example) ---
// In a real app, you'd likely fetch this dynamically
const gpt4oConfig: ModelConfig = {
  apiName: "gpt-4o", // The name the vendor API expects
  isVision: true,
  isImageGeneration: false,
  isThinking: false, // Does it support structured thinking output?
  inputTokenCost: 10, // Cost per million input tokens (optional)
  outputTokenCost: 30, // Cost per million output tokens (optional)
};

// --- Get the Adapter ---
try {
  const vendorName = "openai"; // Or "anthropic", "google", etc.
  const adapter: AIVendorAdapter = AIVendorFactory.getAdapter(
    vendorName,
    gpt4oConfig // Pass the config for the specific model
  );

  console.log(
    `Successfully got adapter for ${vendorName} / ${gpt4TurboConfig.apiName}`
  );
  // Now you can use the 'adapter' instance!
} catch (error) {
  console.error("Failed to get adapter:", error);
}
```

### 4. How to Send a Chat

The main way to interact is using the adapter's `sendChat` method. You pass it a `Chat` object which holds the entire conversation state.

```typescript
import { Chat, ChatResponse, TextBlock } from "snowgander";

// Assume 'adapter' is the AIVendorAdapter instance obtained above

// --- Prepare the Chat Object (Manage this state in your app) ---
let currentChat: Chat = {
  model: gpt4oConfig.apiName, // The model being used
  responseHistory: [
    // Load previous messages here...
    // Example:
    // { role: 'user', content: [{ type: 'text', text: 'Hi there!' }] },
    // { role: 'assistant', content: [{ type: 'text', text: 'Hello! How can I help?' }] }
  ],
  prompt: "Tell me a short joke about computers.", // The user's latest input

  // Optional fields:
  systemPrompt: "You are a witty comedian.", // Sets the AI's persona
  maxTokens: 100, // Limit response length for this turn
  visionUrl: null, // Set to an image URL for vision models
  budgetTokens: null, // Set > 0 to enable 'thinking' mode if supported
  // Add openaiImageGenerationOptions or openaiImageEditOptions here for image models
};

// --- Send the Chat ---
async function makeApiCall() {
  try {
    console.log("Sending chat to AI...");
    const response: ChatResponse = await adapter.sendChat(currentChat);

    console.log("AI Response Received!");

    // --- Process the Response (See Step 5 below) ---
    // IMPORTANT: Update your chat state for the next turn
    currentChat.responseHistory.push(response); // Add the AI's response to history
    currentChat.prompt = ""; // Clear the prompt, ready for next user input

    // Display or use the response content
    const assistantMessage = response.content
      .filter((block): block is TextBlock => block.type === "text") // Get only text blocks
      .map((block) => block.text)
      .join("\n"); // Join if there are multiple text blocks

    console.log("Assistant:", assistantMessage);

    if (response.usage) {
      console.log(`Cost: $${response.usage.totalCost.toFixed(6)}`);
    }
  } catch (error) {
    console.error("Error sending chat:", error);
    // Handle errors appropriately (e.g., show message to user)
  }
}

makeApiCall();
```

### 5. What to Expect in Return (`ChatResponse`)

The `sendChat` method returns a `ChatResponse` object with this structure:

```typescript
interface ChatResponse {
  role: string; // Usually "assistant" (or "error" if something went wrong)
  content: ContentBlock[]; // An array containing the AI's response parts
  usage?: {
    // Optional cost information
    inputCost: number;
    outputCost: number;
    totalCost: number;
  };
}
```

- **`role`**: Identifies the sender (usually `"assistant"`).
- **`content`**: This is an array (`ContentBlock[]`). Each item in the array represents a part of the response. Common types include:
  - `TextBlock`: Contains plain text (`{ type: 'text', text: '...' }`).
  - `ImageDataBlock`: Contains image data (`{ type: 'image_data', mimeType: '...', base64Data: '...' }`).
  - `ThinkingBlock`: Contains structured thinking steps if requested (`{ type: 'thinking', thinking: '...', signature: '...' }`).
  - `ToolUseBlock`: Indicates the AI wants to use a tool (`{ type: 'tool_use', name: '...', input: '...' }`). Your application needs to handle this.
  - `ErrorBlock`: If an error occurred during processing (`{ type: 'error', publicMessage: '...', privateMessage: '...' }`).
- **`usage`**: If available, provides the estimated cost for the interaction based on token counts.

**Your application needs to:**

1.  Check the `role` (e.g., handle `"error"`).
2.  Iterate through the `content` array to process and display the different blocks appropriately (e.g., render text, display images, trigger tool execution).
3.  **Crucially, add the entire `ChatResponse` object to your `responseHistory`** so the AI has context for the next turn.

## Development

- **Build:** `npm run build` (Compiles TS to `dist/`)
- **Test:** `npm test` (Runs Jest tests)

---

**Ready to build more complex AI agents?**

Snowgander provides the vendor abstraction. [**Snowgoose**](https://snowgoose.app) provides the framework.

Give your AI projects structure and power with Snowgoose!

**Explore the Snowgoose App:** [**snowgoose.app**](https://snowgoose.app)
**Check out the Snowgoose Repo:** [**github.com/loneyeti/snowgoose**](https://github.com/loneyeti/snowgoose)

---

```

```
