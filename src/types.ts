// Configuration for a specific AI model, abstracted from Prisma
export interface ModelConfig {
  apiName: string; // e.g., 'gpt-4-turbo'
  isVision: boolean;
  isImageGeneration: boolean;
  isThinking: boolean;
  inputTokenCost?: number;
  outputTokenCost?: number;
  imageOutputTokenCost?: number; // New: Cost for generated image output tokens
  webSearchCost?: number; // New: Flat fee for using the web search tool
  // Add any other fields from the original Model used by adapters if needed
}

// --- Custom Error Type ---
export class NotImplementedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotImplementedError";
  }
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
  generationId?: string;
}

// Define a specific block type for raw image data (e.g., from Google)
export interface ImageDataBlock {
  type: "image_data";
  id?: string | null; // The ID of the image generation call, e.g., "ig_123"
  mimeType: string;
  base64Data: string;
}

export interface ToolUseBlock {
  type: "tool_use";
  id?: string; // Add optional ID field from vendor API (e.g., Anthropic)
  name: string;
  input: string; // Keep as string representation of input JSON
}

export interface ToolResultBlock {
  type: "tool_result";
  toolUseId: string; // Changed from number to string to match vendor API IDs
  content: ContentBlock[];
}

// Define the ContentBlock union type *once*, including all variants
export interface ImageGenerationCallBlock {
  type: "image_generation_call";
  id: string; // The ID of the image generation call, e.g., "ig_123"
}

export interface MetaBlock {
  type: "meta";
  responseId: string;
  usage?: UsageResponse;
}

export type ContentBlock =
  | ThinkingBlock
  | RedactedThinkingBlock
  | TextBlock
  | ImageBlock // Represents URL-based images
  | ImageDataBlock // Represents raw image data
  | ToolUseBlock
  | ToolResultBlock
  | ErrorBlock // Added ErrorBlock
  | ImageGenerationCallBlock
  | MetaBlock; // Add MetaBlock here

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
  responseId?: string; // Add this line
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
  previousResponseId?: string; // Add this line
  mcpAvailableTools?: MCPAvailableTool[];
  // Options specific to OpenAI Image Adapter, passed via Chat
  openaiImageGenerationOptions?: OpenAIImageGenerationOptions;
  openaiImageEditOptions?: OpenAIImageEditOptions;
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
  content: ContentBlock[]; // Content must be an array of ContentBlocks
}

export interface UsageResponse {
  inputCost: number;
  outputCost: number;
  webSearchCost?: number; // New: The flat fee for web search, if applied
  didGenerateImage?: boolean;
  didWebSearch?: boolean;
  totalCost: number;
}

// Represents the structured response from an AI vendor adapter
export interface AIResponse {
  role: string; // Typically 'assistant'
  content: ContentBlock[]; // The generated content
  responseId?: string; // Add this line
  // Optionally include usage stats if adapters provide them
  usage?: UsageResponse;
}

// --- Error Block Type ---
export interface ErrorBlock {
  type: "error";
  code?: string | undefined;
  publicMessage: string; // Safe to show to the end-user
  privateMessage: string; // Detailed error for logging/debugging
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
  previousResponseId?: string; // Add this line
  // Generic tools array for API calls
  tools?: any[];
  // Data retention control
  store?: boolean; // Whether to store the response (default true)
  // Optional: Specific options for OpenAI Image Generation API
  openaiImageGenerationOptions?: OpenAIImageGenerationOptions;
  // Optional: Specific options for OpenAI Image Editing API
  openaiImageEditOptions?: OpenAIImageEditOptions;
  // New: Specific flag to control image generation per request
  useImageGeneration?: boolean;
}

// --- OpenAI Image API Specific Options ---
// To be nested within AIRequestOptions

export interface OpenAIImageGenerationOptions {
  n?: number; // Kept for compatibility with images.generate API
  quality?: "low" | "medium" | "high" | "auto";
  size?: "1024x1024" | "1536x1024" | "1024x1536" | "auto";
  background?: "transparent" | "opaque" | "auto";
  // The 'compression' option is for JPEG/WebP which we are not supporting yet.
  // The 'format' option is for specifying output format which is also out of scope for now.
  user?: string; // Kept for tracking/safety purposes
}

export interface OpenAIImageEditOptions {
  // prompt is usually taken from AIRequestOptions.prompt or messages
  image?: (ImageDataBlock | ImageBlock)[]; // Input image(s) - require adapter to handle URL/base64 conversion
  mask?: ImageDataBlock | ImageBlock; // Optional mask image - require adapter to handle URL/base64 conversion
  n?: number; // Number of images to generate (default 1)
  // response_format is always b64_json for this adapter
  size?: "1024x1024" | "1536x1024" | "1024x1536" | "auto"; // Image dimensions (default auto)
  quality?: "low" | "medium" | "high" | "auto"; // Quality setting (default auto)
  user?: string; // User identifier
  moderation?: "auto" | "low"; // Moderation strictness (default auto)
}

// --- Image Generation/Editing Response Types ---

export interface ImageGenerationResponse {
  // Array of generated images, represented as ImageDataBlocks
  images: ImageDataBlock[];
  // Standard usage stats
  usage: UsageResponse;
}

export interface ImageEditResponse {
  // Array of edited images, represented as ImageDataBlocks
  images: ImageDataBlock[];
  // Standard usage stats
  usage: UsageResponse;
}

// Interface defining the contract for all AI vendor adapters
export interface AIVendorAdapter {
  // Generates a response based on the provided options
  generateResponse(options: AIRequestOptions): Promise<AIResponse>;
  // Simplified method to send a full chat context (history, prompt, config)
  sendChat(chat: Chat): Promise<ChatResponse>;
  // Optional method for MCP-specific chat interactions (if needed)
  sendMCPChat?(
    chat: Chat,
    tools: MCPAvailableTool[],
    options?: AIRequestOptions
  ): Promise<ChatResponse>;
  // Optional method for streaming responses
  streamResponse?(
    options: AIRequestOptions
  ): AsyncGenerator<ContentBlock, void, unknown>;

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
