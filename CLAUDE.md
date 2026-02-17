# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Snowgander is a TypeScript library that provides a unified abstraction layer for multiple AI vendors (OpenAI, Anthropic, Google, OpenRouter, Grok). It powers the Snowgoose agent framework. The library uses the Adapter pattern to normalize vendor-specific APIs into a consistent interface.

## Development Commands

**Build:**
```bash
npm run build
```
Compiles TypeScript to the `dist/` directory.

**Test:**
```bash
npm test
```
Runs all Jest tests. Test files are located in `src/__tests__/` and `src/vendors/__tests__/`.

**Publish:**
```bash
npm run prepublishOnly
```
Automatically runs build before publishing to npm.

## Architecture

### Core Design Patterns

The codebase uses two primary patterns:

1. **Adapter Pattern**: Each vendor (OpenAI, Anthropic, Google, etc.) has its own adapter class implementing the `AIVendorAdapter` interface
2. **Factory Pattern**: `AIVendorFactory` instantiates the appropriate adapter based on vendor name and model configuration

### Key Components

**Types (`src/types.ts`):**
- Defines all shared interfaces and types across the library
- `AIVendorAdapter`: The core interface all vendor adapters must implement
- `ContentBlock`: Union type for all message content (text, images, thinking, tools, etc.)
- `ModelConfig`: Configuration for specific AI models (apiName, capabilities, costs)
- `VendorConfig`: Vendor-specific configuration (apiKey, baseURL, etc.)
- `Chat`: Represents stateful conversation context with history

**Factory (`src/factory.ts`):**
- `AIVendorFactory`: Central factory for creating vendor adapters
- `setVendorConfig()`: Configure API keys and settings for each vendor
- `getAdapter()`: Returns the appropriate adapter instance for a vendor/model pair
- Currently supports: `openai`, `anthropic`, `google`, `openrouter`, `openai-image`, `grok`

**Adapters (`src/vendors/*.ts`):**
Each adapter class:
- Implements `AIVendorAdapter` interface
- Takes `VendorConfig` and `ModelConfig` in constructor
- Translates between Snowgander's unified types and vendor-specific SDK formats
- Handles vendor-specific quirks (e.g., Anthropic's thinking blocks, OpenAI's response format)

**Utilities (`src/utils.ts`):**
- `computeResponseCost()`: Calculates cost based on tokens and per-million-token pricing
- `getImageDataFromUrl()`: Fetches images and converts to base64 with MIME type detection

### Content Block System

All message content flows through the `ContentBlock[]` array type. This ensures uniformity across vendors:

- `TextBlock`: Plain text content
- `ImageBlock`: URL-based images
- `ImageDataBlock`: Raw base64 image data
- `ThinkingBlock`: Structured thinking/reasoning steps (Anthropic/OpenRouter)
- `ToolUseBlock` / `ToolResultBlock`: Tool/function calling
- `ErrorBlock`: Error information
- `MetaBlock`: Response metadata and usage stats

Each adapter is responsible for converting vendor-specific response formats into these normalized blocks.

### Capability Flags

Adapters expose capability flags set from `ModelConfig`:
- `isVisionCapable`: Can process image inputs
- `isImageGenerationCapable`: Can generate images
- `isThinkingCapable`: Supports explicit thinking/reasoning output

These flags allow consuming applications to conditionally enable features based on model capabilities.

### Cost Tracking

Cost calculation is standardized via:
1. `ModelConfig` specifies per-million-token costs (`inputTokenCost`, `outputTokenCost`, etc.)
2. Adapters use `computeResponseCost()` to calculate costs
3. `UsageResponse` returns structured cost breakdown (`inputCost`, `outputCost`, `totalCost`)

## Adding a New Vendor Adapter

To add support for a new AI vendor:

1. **Create adapter file**: `src/vendors/newvendor.ts`
2. **Implement `AIVendorAdapter`**:
   - Required: `generateResponse()`, `sendChat()`
   - Optional: `streamResponse()`, `sendMCPChat()`
3. **Convert message formats**:
   - Input: Transform Snowgander `Message[]` to vendor SDK format
   - Output: Transform vendor response to `ContentBlock[]`
4. **Handle vendor-specific features**: Use capability flags and `ModelConfig` to enable/disable features
5. **Add to factory**: Import and register in `AIVendorFactory.getAdapter()` switch statement
6. **Write tests**: Create `src/vendors/__tests__/newvendor.test.ts`

## Important Patterns

**Unified Thinking/Reasoning:**
Use `isThinkingCapable` flag, `budgetTokens` parameter, and `ThinkingBlock` type to handle both Anthropic's extended thinking and OpenRouter's reasoning features consistently.

**Claude 4.6+ Support:**
The Anthropic adapter automatically detects model versions and uses the appropriate API format:

- **Claude 4.6+ Models** (`claude-opus-4-6`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`):
  - Uses adaptive thinking: `thinking: {type: "adaptive"}`
  - Maps `budgetTokens` to `effort` parameter (low: 0-1, medium: 1-8192, high: 8192+)
  - Supports explicit `effort` parameter in `AIRequestOptions` (overrides budgetTokens mapping)
  - Supports `outputFormat` for structured outputs via `output_config.format`
  - Cannot use both `temperature` and `topP` (will throw error)
  - Handles new stop reasons: `refusal`, `model_context_window_exceeded`

- **Legacy Models** (Claude 3.x, 4.0-4.5):
  - Uses `thinking: {type: "enabled", budget_tokens: N}`
  - Claude 3.x models support both temperature and topP simultaneously
  - Claude 4.0-4.5 models only support temperature OR topP (not both)

Example:
```typescript
// Claude 4.6 with adaptive thinking
const response = await adapter.generateResponse({
  model: 'claude-opus-4-6',
  messages: [...],
  thinkingMode: true,
  effort: 'high',  // or use budgetTokens for automatic mapping
  maxTokens: 4096,
});

// Claude 3.x with legacy thinking
const response = await adapter.generateResponse({
  model: 'claude-3-opus-20240229',
  messages: [...],
  thinkingMode: true,
  budgetTokens: 5000,
  temperature: 0.7,
  topP: 0.9,  // Supported on 3.x models
});
```

**Configuration Injection:**
Never hardcode API keys or vendor settings. Always use `VendorConfig` and `ModelConfig` injected via factory.

**Tool/MCP Handling:**
The approach is evolving. The Anthropic adapter now handles tools directly in `sendChat()` and `generateResponse()`. When adding new adapters, follow the pattern in `AnthropicAdapter` for tool integration.

**OpenAI API Pattern:**
The OpenAI adapter uses `client.responses.create()` for the latest OpenAI API format. New OpenAI-compatible adapters should follow this pattern.
