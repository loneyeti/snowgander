// Export the factory
export { AIVendorFactory } from "./factory";

// Export core types needed by the consuming application
export type {
  AIVendorAdapter,
  AIRequestOptions,
  AIResponse,
  VendorConfig,
  ModelConfig,
  Chat,
  ChatResponse,
  ContentBlock,
  ThinkingBlock,
  RedactedThinkingBlock,
  TextBlock,
  ImageBlock,
  ImageDataBlock,
  ToolResultBlock,
  ToolUseBlock,
  MCPAvailableTool,
  MCPTool,
  Message,
  OpenAIImageGenerationOptions,
  OpenAIImageEditOptions,
  ImageGenerationResponse,
  ImageEditResponse,
} from "./types";

// Optionally export individual adapters if direct use is needed,
// but typically the factory is the intended interface.
// export { OpenAIAdapter } from './vendors/openai';
// export { AnthropicAdapter } from './vendors/anthropic';
// export { GoogleAIAdapter } from './vendors/google';
// export { OpenRouterAdapter } from './vendors/openrouter';
