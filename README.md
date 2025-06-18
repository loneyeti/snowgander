Of course! Adding streaming support is a great feature, and it's important to document it clearly. I've updated the `README.md` to include the new streaming functionality, restructured the "how-to" section to be more logical for new users, and made some other minor improvements for clarity and correctness.

Here is the updated `README.md` file:

---

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

Snowgander is a TypeScript library that makes it easy to talk to different AI models (like those from OpenAI, Anthropic, Google, and OpenRouter) using one simple, consistent interface.

Stop writing vendor-specific code! Use Snowgander to:

- Easily switch between AI models and vendors.
- Use a single method (`sendChat`) for most conversational interactions.
- **Stream responses in real-time** for interactive UIs.
- Handle different types of content (text, images, tool use, thinking steps) uniformly.

## How to Use It: The Basics

### 1. Installation

```bash
npm install snowgander
```

### 2. Configuration (Set API Keys)

Do this once when your application starts. The factory holds the configuration for each vendor.

```typescript
import { AIVendorFactory } from "snowgander";

// Example: Load keys from environment variables
if (process.env.OPENAI_API_KEY) {
  const config = { apiKey: process.env.OPENAI_API_KEY };
  AIVendorFactory.setVendorConfig("openai", config);
  // Also configure 'openai-image' if using DALL-E via OpenAIImageAdapter
  AIVendorFactory.setVendorConfig("openai-image", config);
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
    // OpenRouter's baseURL is the default, but you can override it
    baseURL: "https://openrouter.ai/api/v1",
  });
}
```

### 3. How to Initiate an Adapter

To get an adapter, you need the **vendor name** (e.g., `"openai"`) and a **`ModelConfig` object** describing the specific model you want to use.

```typescript
import { AIVendorFactory, ModelConfig, AIVendorAdapter } from "snowgander";

// --- Prepare ModelConfig (Example) ---
// In a real app, you'd likely fetch this from a database or config file.
const gpt4oConfig: ModelConfig = {
  apiName: "gpt-4o", // The name the vendor API expects
  isVision: true, // Can it process images?
  isImageGeneration: true, // Can it generate images?
  isThinking: false, // Does it support structured thinking output?
  inputTokenCost: 5, // Cost per million input tokens (optional)
  outputTokenCost: 15, // Cost per million output tokens (optional)
  imageOutputTokenCost: 20, // Special cost for image generation output
  webSearchCost: 0.05, // Flat fee when a web search tool is used
};

// --- Get the Adapter ---
let adapter: AIVendorAdapter;
try {
  const vendorName = "openai"; // Or "anthropic", "google", etc.
  adapter = AIVendorFactory.getAdapter(
    vendorName,
    gpt4oConfig // Pass the config for the specific model
  );

  console.log(
    `Successfully got adapter for ${vendorName} / ${gpt4oConfig.apiName}`
  );
  // Now you can use the 'adapter' instance!
} catch (error) {
  console.error("Failed to get adapter:", error);
}
```

### 4. Making API Calls

Snowgander provides two primary ways to interact with AI models: a standard request-response model and a real-time streaming model.

#### A. Standard Request/Response (`generateResponse`)

The `generateResponse` method is the fundamental way to get a complete response from an AI. You provide all the messages at once, and it returns the full response after the model has finished processing.

```typescript
import { AIRequestOptions, AIResponse, TextBlock } from "snowgander";

// Assume 'adapter' is an AIVendorAdapter instance obtained above

async function getStandardResponse() {
  const options: AIRequestOptions = {
    model: gpt4oConfig.apiName,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Tell me a short joke about computers." },
        ],
      },
    ],
    systemPrompt: "You are a witty comedian.",
    maxTokens: 100,
  };

  try {
    console.log("Sending request to AI...");
    const response: AIResponse = await adapter.generateResponse(options);
    console.log("AI Response Received!");

    // Process the response content (see section 5 for details)
    const assistantMessage = response.content
      .filter((block): block is TextBlock => block.type === "text")
      .map((block) => block.text)
      .join("");

    console.log("Assistant:", assistantMessage);

    if (response.usage) {
      console.log(`Cost: $${response.usage.totalCost.toFixed(6)}`);
    }
  } catch (error) {
    console.error("Error generating response:", error);
  }
}

getStandardResponse();
```

#### B. Streaming Responses (`streamResponse`)

For interactive applications where you want to display the response as it's being generated (like a "typewriter" effect), use the `streamResponse` method. It returns an `AsyncGenerator` that yields response chunks (`ContentBlock`s) in real-time.

```typescript
import { AIRequestOptions, ContentBlock, TextBlock } from "snowgander";

// Assume 'adapter' is an AIVendorAdapter instance obtained above

async function getStreamingResponse() {
  const options: AIRequestOptions = {
    model: gpt4oConfig.apiName,
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "Write a short poem about the sea." }],
      },
    ],
    systemPrompt: "You are a thoughtful poet.",
    maxTokens: 150,
  };

  try {
    console.log("Streaming response from AI...");
    const stream = adapter.streamResponse(options);
    if (!stream) {
      console.log("This adapter does not support streaming.");
      return;
    }

    process.stdout.write("Assistant: ");
    // Use a for-await-of loop to process chunks as they arrive
    for await (const chunk of stream) {
      if (chunk.type === "text") {
        // In a real app, you would append this to your UI
        process.stdout.write(chunk.text);
      } else if (chunk.type === "error") {
        console.error(`\nStream Error: ${chunk.privateMessage}`);
        break;
      }
      // You can also handle other chunk types like 'thinking' or 'image_data'
    }
    console.log("\n--- End of Stream ---");
    // Note: Final usage/cost information is not available when streaming.
  } catch (error) {
    console.error("\nError streaming response:", error);
  }
}

getStreamingResponse();
```

#### C. Stateful Conversations (`sendChat`)

For building chatbots, the `sendChat` method is a convenient helper. It wraps `generateResponse` and simplifies state management. You pass it a `Chat` object containing the full conversation history and the new prompt, and it internally constructs the request for you.

```typescript
import { Chat, ChatResponse, TextBlock } from "snowgander";

// Assume 'adapter' is an AIVendorAdapter instance obtained above

// --- Manage this Chat object state in your application ---
let currentChat: Chat = {
  model: gpt4oConfig.apiName,
  responseHistory: [], // Start with an empty history for a new conversation
  prompt: "Tell me a short joke about computers.", // The user's first message
  systemPrompt: "You are a witty comedian.",
  maxTokens: 100,
  visionUrl: null,
  budgetTokens: null,
};

async function haveAConversation() {
  try {
    console.log(`User: ${currentChat.prompt}`);
    const response: ChatResponse = await adapter.sendChat(currentChat);
    console.log("AI Response Received!");

    const assistantMessage = response.content
      .filter((block): block is TextBlock => block.type === "text")
      .map((block) => block.text)
      .join("");
    console.log("Assistant:", assistantMessage);

    // --- IMPORTANT: Update your chat state for the next turn ---
    // 1. Add the user's prompt to the history
    currentChat.responseHistory.push({
      role: "user",
      content: [{ type: "text", text: currentChat.prompt }],
    });
    // 2. Add the AI's full response to the history
    currentChat.responseHistory.push(response);
    // 3. Clear the prompt, ready for the next user input
    currentChat.prompt = "";
  } catch (error) {
    console.error("Error sending chat:", error);
  }
}

haveAConversation();
```

### 5. Understanding the Response

Both `generateResponse` and `sendChat` return a response object (`AIResponse` or `ChatResponse`) with this structure:

```typescript
interface ChatResponse {
  role: string; // Usually "assistant" (or "error" if something went wrong)
  content: ContentBlock[]; // An array containing the AI's response parts
  usage?: {
    // Optional cost information
    inputCost: number;
    outputCost: number;
    totalCost: number;
    // ... other optional cost fields
  };
}
```

- **`role`**: Identifies the sender (usually `"assistant"`).
- **`content`**: This is an array (`ContentBlock[]`). Each item in the array represents a part of the response. Common types include:
  - `TextBlock`: Contains plain text (`{ type: 'text', text: '...' }`).
  - `ImageDataBlock`: Contains generated image data (`{ type: 'image_data', ... }`).
  - `ThinkingBlock`: Contains structured thinking steps from the model (`{ type: 'thinking', ... }`).
  - `ToolUseBlock`: Indicates the AI wants to use a tool that your application must handle.
  - `ErrorBlock`: If an error occurred (`{ type: 'error', ... }`).
- **`usage`**: If available, provides the estimated cost for the interaction based on token counts.

**Your application needs to:**

1.  Check the `role` (e.g., handle an `"error"` role).
2.  Iterate through the `content` array to process and display the different blocks appropriately (e.g., render text, display images, trigger tool execution).
3.  **For conversations, add the entire response object to your `responseHistory`** so the AI has context for the next turn.

## Development

- **Build:** `npm run build` (Compiles TS to `dist/`)
- **Test:** `npm test` (Runs Jest tests)

---

**Ready to build more complex AI agents?**

Snowgander provides the vendor abstraction. [**Snowgoose**](https://snowgoose.app) provides the framework.

Give your AI projects structure and power with Snowgoose!

**Explore the Snowgoose App:** [**snowgoose.app**](https://snowgoose.app)
**Check out the Snowgoose Repo:** [**github.com/loneyeti/snowgoose**](https://github.com/loneyeti/snowgoose)
