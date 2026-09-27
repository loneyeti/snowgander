// Configuration for a specific AI model, abstracted from Prisma
export interface ModelConfig {
  apiName: string; // e.g., 'gpt-4-turbo'
  isVision: boolean;
  isImageGeneration: boolean;
  isThinking: boolean;
  inputTokenCost?: number;
  outputTokenCost?: number;
  imageOutputTokenCost?: number; // New: Cost for generated image output tokens
  webSearchCost?: number; // Fee per web search call
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
  // Web sources this text is grounded in (OpenAI url_citation annotations,
  // Claude web_search_result_location / web_fetch document citations).
  // Streaming adapters may yield empty-text chunks that carry only citations.
  citations?: Citation[];
}

// --- Grounded web research ---

export interface Citation {
  type: "url_citation";
  url: string;
  title?: string;
  citedText?: string; // Claude: the quoted source text
  startIndex?: number; // OpenAI: character offsets into the output text
  endIndex?: number;
  vendor: "openai" | "anthropic";
  raw?: unknown; // Original vendor citation object
}

// One research step: a web search or page fetch run by the vendor's server.
export interface ServerToolUseBlock {
  type: "server_tool_use";
  id: string;
  name: string; // "web_search" | "web_fetch"
  input: string; // JSON string, e.g. {"query": "..."} or {"url": "..."}
  status?: "in_progress" | "completed" | "failed";
  vendor: "openai" | "anthropic";
}

export interface WebSearchResultItem {
  url: string;
  title?: string;
  pageAge?: string | null;
}

export interface WebSearchToolResultBlock {
  type: "web_search_tool_result";
  toolUseId: string;
  results: WebSearchResultItem[];
  errorCode?: string;
  vendor: "openai" | "anthropic";
  // Claude: the verbatim API block (with encrypted_content). Keep it in history
  // so the adapter can send it back on later turns.
  raw?: unknown;
}

export interface WebFetchToolResultBlock {
  type: "web_fetch_tool_result";
  toolUseId: string;
  url?: string;
  title?: string;
  retrievedAt?: string | null;
  errorCode?: string;
  vendor: "anthropic";
  raw?: unknown; // Verbatim API block, echoed back in history
}

// Vendor-neutral web search request options. Each adapter builds its own tool.
export interface WebSearchOptions {
  fetch?: boolean; // Claude only: also enable web_fetch
  maxUses?: number; // Claude only: max searches (and fetches) per request
  allowedDomains?: string[];
  blockedDomains?: string[]; // Claude only
  userLocation?: {
    city?: string;
    region?: string;
    country?: string; // ISO 3166-1 alpha-2
    timezone?: string; // IANA timezone
  };
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
  // Streaming image generation: partial previews share the final image's id.
  // The last block for an id has isPartial false (or undefined) and is the one to persist.
  isPartial?: boolean;
  partialImageIndex?: number;
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
  | ServerToolUseBlock
  | WebSearchToolResultBlock
  | WebFetchToolResultBlock
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
  // Claude 4.6+ / GPT-5+ specific options (optional — omit to preserve prior behavior)
  effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"; // Effort level for adaptive thinking / reasoning
  temperature?: number; // Sampling temperature (rejected outright on some newer models)
  topP?: number; // Top-p sampling (alternative to temperature, only for legacy-tier models)
  outputFormat?: any; // Structured output format for output_config / text.format
  thinkingDisplay?: "summarized" | "omitted"; // Claude 4.7+: thinking.display (adapter defaults to "summarized" where the API default is "omitted")
  verbosity?: "low" | "medium" | "high"; // GPT-5+: controls text.verbosity
  reasoningMode?: "standard" | "pro"; // GPT-5.6+: controls reasoning.mode
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
  webSearchCost?: number; // ModelConfig.webSearchCost x webSearchCount
  didGenerateImage?: boolean;
  didWebSearch?: boolean;
  webSearchCount?: number; // Web searches run by the vendor for this response
  webFetchCount?: number; // Claude web fetches (billed as tokens only)
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
  // Grounded web research: the adapter adds its vendor's web search (and, for
  // Claude, web fetch) tool. Ignored by adapters without web search support.
  webSearch?: WebSearchOptions;
  // Data retention control
  store?: boolean; // Whether to store the response (default true)
  // Optional: Specific options for OpenAI Image Generation API
  openaiImageGenerationOptions?: OpenAIImageGenerationOptions;
  // Optional: Specific options for OpenAI Image Editing API
  openaiImageEditOptions?: OpenAIImageEditOptions;
  // New: Specific flag to control image generation per request
  useImageGeneration?: boolean;
  // Claude 4.6+ / GPT-5+ specific options
  effort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"; // Effort level for adaptive thinking / reasoning (Claude 4.6+, GPT-5+)
  outputFormat?: any; // Structured output format for output_config / text.format (Claude 4.6+, GPT-5+)
  topP?: number; // Top-p sampling (alternative to temperature, only for legacy-tier models)
  thinkingDisplay?: "summarized" | "omitted"; // Claude 4.7+: thinking.display (adapter defaults to "summarized" where the API default is "omitted")
  verbosity?: "low" | "medium" | "high"; // GPT-5+: controls text.verbosity
  reasoningMode?: "standard" | "pro"; // GPT-5.6+: controls reasoning.mode
}

// --- OpenAI Image API Specific Options ---
// To be nested within AIRequestOptions

// Grok Imagine aspect ratios (grok-imagine-image models only)
export type GrokImageAspectRatio =
  | "1:1"
  | "3:4"
  | "4:3"
  | "9:16"
  | "16:9"
  | "2:3"
  | "3:2"
  | "9:19.5"
  | "19.5:9"
  | "9:20"
  | "20:9"
  | "1:2"
  | "2:1"
  | "21:9"
  | "5:2"
  | "auto";

export interface OpenAIImageGenerationOptions {
  n?: number; // Kept for compatibility with images.generate API (ignored by the Responses image_generation tool). Grok: 1-10
  model?: string; // Image model for the Responses image_generation tool (e.g. "gpt-image-2.5-sunburst")
  quality?: "low" | "medium" | "high" | "xhigh" | "max" | "auto"; // Grok: low/medium/auto on grok-imagine-image-2.0 only (high+ maps to medium)
  aspectRatio?: GrokImageAspectRatio; // Grok only: output aspect ratio (default auto; derived from size when omitted)
  resolution?: "1k" | "1.5k" | "2k"; // Grok only: output resolution (default 1k)
  size?: "1024x1024" | "1536x1024" | "1024x1536" | "auto" | `${number}x${number}`; // Custom sizes: multiples of 16, 1:3-3:1, max 3840px edges
  background?: "transparent" | "opaque" | "auto";
  outputFormat?: "png" | "jpeg" | "webp"; // Default png
  outputCompression?: number; // 0-100, jpeg/webp only
  moderation?: "auto" | "low";
  action?: "auto" | "generate" | "edit"; // Responses tool; Grok: auto edits the prior/attached image when present
  partialImages?: 0 | 1 | 2 | 3; // Streaming partial previews (adapter default 2)
  inputFidelity?: "low" | "high";
  inputImageMask?: { fileId?: string; imageUrl?: string }; // Responses tool only
  user?: string; // Kept for tracking/safety purposes
}

export interface OpenAIImageEditOptions {
  // prompt is usually taken from AIRequestOptions.prompt or messages
  image?: (ImageDataBlock | ImageBlock)[]; // Input image(s) - require adapter to handle URL/base64 conversion. Grok: up to 5, overrides history-derived inputs
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
