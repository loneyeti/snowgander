// Configuration for a specific AI model, abstracted from Prisma
export interface ModelConfig {
  apiName: string; // e.g., 'gpt-4-turbo'
  isVision: boolean;
  isImageGeneration: boolean;
  isThinking: boolean;
  inputTokenCost?: number;
  outputTokenCost?: number;
  // Add any other fields from the original Model used by adapters if needed
}

// --- Types moved from app/_lib/model.ts ---

// Interface for Anthropic thinking blocks
export interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  signature: string;
}

export interface RedactedThinkingBlock {
  type: "redacted_thinking";
  data: string;
}

export interface TextBlock {
  type: "text";
  text: string;
}

// Represents URL-based images
export interface ImageBlock {
  type: "image";
  url: string;
}

// Define a specific block type for raw image data (e.g., from Google)
export interface ImageDataBlock {
  type: "image_data";
  mimeType: string;
  base64Data: string;
}

export interface ToolUseBlock {
  type: "tool_use";
  name: string;
  input: string;
}

export interface ToolResultBlock {
  type: "tool_result";
  toolUseId: number;
  content: ContentBlock[];
}

// Define the ContentBlock union type *once*, including all variants
export type ContentBlock =
  | ThinkingBlock
  | RedactedThinkingBlock
  | TextBlock
  | ImageBlock // Represents URL-based images
  | ImageDataBlock // Represents raw image data
  | ToolUseBlock
  | ToolResultBlock;

export interface MCPAvailableTool {
  name: string;
  description: string;
  input_schema: string;
}

// Content of a chat response can either be plain text or ContentBlock array
export interface ChatResponse {
  role: string;
  content: ContentBlock[];
  usage?: UsageResponse;
}

// Represents the state/data needed for a chat interaction
export interface Chat {
  responseHistory: ChatResponse[]; // History of messages in the chat
  visionUrl: string | null; // URL for vision input (or null)
  model: string; // Identifier for the AI model being used (maps to ModelConfig.apiName)
  prompt: string; // The current user prompt/input text
  imageURL: string | null; // URL of an image (alternative to imageData, maybe for display?)
  maxTokens: number | null; // Max tokens for the response
  budgetTokens: number | null; // Token budget for thinking mode
  systemPrompt?: string; // The actual system prompt text from the persona
  mcpAvailableTools?: MCPAvailableTool[];
}

// Represents an MCP Tool configuration
export interface MCPTool {
  id: number;
  name: string;
  path: string;
  env_vars?: Record<string, string>;
}

// --- Original types.ts content (now integrated/using ContentBlock) ---

// Represents a single message in a chat history or request
export interface Message {
  role: string; // e.g., 'user', 'assistant', 'system'
  content: ContentBlock[]; // Content can be simple text or structured blocks
}

export interface UsageResponse {
  inputCost: number;
  outputCost: number;
  totalCost: number;
}

// Represents the structured response from an AI vendor adapter
export interface AIResponse {
  role: string; // Typically 'assistant'
  content: ContentBlock[]; // The generated content
  // Optionally include usage stats if adapters provide them
  usage?: UsageResponse;
}

// Options for making a request to an AI vendor adapter
export interface AIRequestOptions {
  model: string; // Model identifier (e.g., ModelConfig.apiName)
  messages: Message[]; // Array of messages for context/prompt
  maxTokens?: number; // Max tokens for the response
  temperature?: number; // Sampling temperature
  systemPrompt?: string; // System-level instructions
  visionUrl?: string; // Optional base64 image data for vision
  modelId?: number; // Optional original model ID (if needed by adapter logic)
  thinkingMode?: boolean; // Flag to enable thinking mode (if supported)
  budgetTokens?: number; // Token budget for thinking mode
  prompt?: string; // The primary user prompt (often redundant if included in messages)
  // Add other vendor-specific options if necessary (e.g., tools for Anthropic/Google)
  tools?: any[]; // Formatted tools for API call (e.g., Anthropic)
}

// Interface defining the contract for all AI vendor adapters
export interface AIVendorAdapter {
  // Generates a response based on the provided options
  generateResponse(options: AIRequestOptions): Promise<AIResponse>;
  // Generates an image based on the chat context (usually the prompt)
  // Returns a data URI string (data:mime/type;base64,...) or potentially a URL if uploaded by adapter
  generateImage(chat: Chat): Promise<string>;
  // Simplified method to send a full chat context (history, prompt, config)
  sendChat(chat: Chat): Promise<ChatResponse>;

  // Capability flags
  isVisionCapable?: boolean;
  isImageGenerationCapable?: boolean;
  isThinkingCapable?: boolean; // For models supporting explicit thinking steps

  // Optional cost information (per million tokens)
  inputTokenCost?: number;
  outputTokenCost?: number;
}

// Configuration needed for initializing a vendor adapter
export interface VendorConfig {
  apiKey: string; // The API key for the vendor
  organizationId?: string; // Optional organization ID (e.g., for OpenAI)
  baseURL?: string; // Optional base URL override (e.g., for proxies or self-hosted models)
}
