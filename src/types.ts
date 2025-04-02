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

// Define the ContentBlock union type *once*, including all variants
export type ContentBlock =
  | ThinkingBlock
  | RedactedThinkingBlock
  | TextBlock
  | ImageBlock // Represents URL-based images
  | ImageDataBlock; // Represents raw image data

// Content of a chat response can either be plain text or ContentBlock array
export interface ChatResponse {
  role: string;
  content: string | ContentBlock[];
}

// Represents the state/data needed for a chat interaction
export interface Chat {
  responseHistory: ChatResponse[]; // History of messages in the chat
  personaId: number; // ID of the persona being used
  outputFormatId: number; // ID of the desired output format
  mcpToolId?: number; // Optional ID of an MCP tool being used
  renderTypeName: string; // Name of the rendering type for output
  imageData: string | null; // Base64 image data for vision input (or null)
  model: string; // Identifier for the AI model being used (maps to ModelConfig.apiName)
  modelId: number; // Original ID from the database (may or may not be needed by adapter)
  prompt: string; // The current user prompt/input text
  imageURL: string | null; // URL of an image (alternative to imageData, maybe for display?)
  maxTokens: number | null; // Max tokens for the response
  budgetTokens: number | null; // Token budget for thinking mode
  personaPrompt?: string; // The actual system prompt text from the persona
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
  content: string | ContentBlock[]; // Content can be simple text or structured blocks
}

// Represents the structured response from an AI vendor adapter
export interface AIResponse {
  role: string; // Typically 'assistant'
  content: string | ContentBlock[]; // The generated content
  // Optionally include usage stats if adapters provide them
  // usage?: { inputTokens: number; outputTokens: number; };
}

// Options for making a request to an AI vendor adapter
export interface AIRequestOptions {
  model: string; // Model identifier (e.g., ModelConfig.apiName)
  messages: Message[]; // Array of messages for context/prompt
  maxTokens?: number; // Max tokens for the response
  temperature?: number; // Sampling temperature
  systemPrompt?: string; // System-level instructions
  imageData?: string; // Optional base64 image data for vision
  modelId?: number; // Optional original model ID (if needed by adapter logic)
  thinkingMode?: boolean; // Flag to enable thinking mode (if supported)
  budgetTokens?: number; // Token budget for thinking mode
  prompt?: string; // The primary user prompt (often redundant if included in messages)
  // Add other vendor-specific options if necessary (e.g., tools for Anthropic/Google)
  // tools?: any[];
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
  // Method specifically for handling interactions involving MCP tools
  sendMCPChat(chat: Chat, mcpToolData: MCPTool): Promise<ChatResponse>;

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
