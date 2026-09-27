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
- Currently supports: `openai`, `anthropic`, `google`, `openrouter`, `openai-image`, `grok` (chat + Grok Imagine image generation/editing)

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

**Claude Model Support (3.x through Opus 4.8 / Sonnet 5 / Fable 5 / Mythos 5):**
The Anthropic adapter (`src/vendors/anthropic.ts`) classifies each request's model into two
*independent* tiers, rather than a single generation string, because Anthropic's thinking API
and its sampling-parameter constraints don't change in lockstep across releases:

- `getThinkingTier(model)` → `"legacyBudget" | "adaptiveTransitional" | "adaptiveOnly"` — controls
  whether thinking uses `budget_tokens` or `{type: "adaptive"}`, and which `effort` levels are valid.
- `getSamplingPolicy(model)` → `"both" | "single" | "none"` — controls how many of
  `temperature`/`topP` may be sent. This does **not** track the thinking tier 1:1: e.g. Sonnet 4.5
  and Haiku 4.5 use legacy `budget_tokens` thinking, but like every Claude 4+ model they still
  reject sending both sampling params together.

| Thinking tier | Models | Thinking request shape | Effort levels |
|---|---|---|---|
| `legacyBudget` (default/fallback) | Claude 3.x, Opus 4.0/4.1/4.5, Sonnet 4.0/4.5, Haiku 3.x/4.5, any unrecognized model | `thinking: {type: "enabled", budget_tokens: N}` | none — no `output_config` |
| `adaptiveTransitional` | `claude-opus-4-6`, `claude-sonnet-4-6` | `thinking: {type: "adaptive"}` | low / medium / high / max |
| `adaptiveOnly` | `claude-opus-4-7`, `claude-opus-4-8`, `claude-opus-5`, `claude-opus-5-5`, `claude-sonnet-5`, `claude-fable-5`, `claude-mythos-5` | `thinking: {type: "adaptive"}` only — `budget_tokens` is never sent | low / medium / high / xhigh / max |

| Sampling policy | Models | Behavior |
|---|---|---|
| `both` | Claude 2.x / 3.x only | `temperature` and `topP` may both be sent |
| `single` | Everything 4.x that isn't `adaptiveOnly` (4.0-4.5, the 4.6 family, Sonnet 4.5, Haiku 4.5, etc.) | At most one of `temperature`/`topP` — throws if both are set |
| `none` | `adaptiveOnly` tier (Opus 4.7/4.8, Opus 5/5.5, Sonnet 5, Fable 5, Mythos 5) | Throws if *either* `temperature` or `topP` is set at all |

Other tier-driven behavior:
- `effort` is honored on both adaptive tiers and is sent whenever provided, even with
  `thinkingMode` off (on the Claude 5 family it's the only thinking control; the API default is
  `medium` on Opus 5.5, `high` on earlier models). `normalizeEffort` maps the OpenAI-only values:
  `minimal`→`low`; `none`→`low` on `adaptiveOnly` (thinking can't be disabled) or thinking omitted
  on `adaptiveTransitional`; `xhigh`→`high` on `adaptiveTransitional` (no xhigh there). When
  `effort` is omitted and thinking is requested, `budgetTokens` is mapped via `mapBudgetToEffort`
  (low/medium/high only).
- `thinking.display`: `adaptiveOnly` models default to `"omitted"` (empty thinking text, and on
  Opus 5.5 the progress notes between tool calls arrive as thinking blocks), so the adapter sends
  `display: "summarized"` unless `thinkingDisplay` overrides it. Models that think by default
  (`thinksByDefault`: the Claude 5 family, not Opus 4.7/4.8) get `thinking: {type: "adaptive",
  display}` even when `thinkingMode` is off; Opus 4.7/4.8 only get it when thinking is requested.
  The 4.6 family defaults to summarized, so display is sent there only when `thinkingDisplay` is set.
- `redacted_thinking` blocks are mapped through responses, streams, and request history, since
  thinking blocks must be echoed back unmodified in tool loops.
- Adaptive-tier models reject assistant prefill; `assertNoPrefill` throws before the API call when
  the last message is an assistant turn.
- `outputFormat` (structured outputs via `output_config.format`) is supported on both adaptive
  tiers, not just `adaptiveOnly`.
- Default `max_tokens` (when the caller doesn't set `maxTokens`, e.g. Snowgoose): `max_tokens` caps
  thinking + text combined, so the adapter defaults high: **64000 for `streamResponse`**, **16000 for
  `generateResponse`/`sendChat`** (the SDK throws for non-streaming requests above ~21.3k, and above
  8192 for Opus 4.0/4.1). Both are clamped to the model's max output (`getMaxOutputTokens`: 4096 for
  Claude 3 Opus/Sonnet/Haiku, 8192 for 3.5, 32000 for Opus 4.0/4.1, 64000 otherwise). Legacy-tier
  `budget_tokens` is clamped so at least 1024 tokens remain for the answer.
- Stop reasons `refusal`, `max_tokens`, and `model_context_window_exceeded` each produce an
  `ErrorBlock` via `buildStopReasonError` (appended in `generateResponse`, yielded before the meta
  block in `streamResponse`) so they're never silent. Refusals include `stop_details.category`
  (e.g. `"cyber"`, `"bio"`, `"reasoning_extraction"`) in `ErrorBlock.privateMessage`.
- `sendChat()` forwards `effort`, `temperature`, `topP`, `outputFormat`, and `thinkingDisplay` from `Chat` to
  `generateResponse()` (all optional fields on `Chat` — omitting them preserves prior behavior for
  existing callers). `thinkingMode` is derived as `(chat.budgetTokens ?? 0) > 0 || !!chat.effort`.

Example:
```typescript
// Opus 4.8 (adaptiveOnly) — adaptive thinking, no sampling params allowed
const response = await adapter.generateResponse({
  model: 'claude-opus-4-8',
  messages: [...],
  thinkingMode: true,
  effort: 'xhigh',  // or omit and use budgetTokens for automatic low/medium/high mapping
  maxTokens: 4096,
});

// Claude 3.x (legacyBudget) — classic thinking, both sampling params allowed
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

**Web Search / Grounded Research (OpenAI + Claude):**
Callers set `AIRequestOptions.webSearch` (`WebSearchOptions`), and each adapter builds its own tool:
- OpenAI: `buildTools` adds `{type: "web_search"}` (filters, user_location) and `include: ["web_search_call.action.sources"]`, and rewrites a legacy `web_search_preview` in `tools` to `web_search`.
- Anthropic: `buildWebTools` picks `web_search_20260209` / `web_fetch_20260209` for adaptive tiers, else `web_search_20250305` / `web_fetch_20250910`. `fetch: true` adds web_fetch with citations enabled. Caller tools with the same `name` win.

Research is normalized into `ServerToolUseBlock` (a step, with `input` as a JSON string), `WebSearchToolResultBlock` / `WebFetchToolResultBlock` (with `errorCode` for the HTTP-200 error form), and `TextBlock.citations` (`Citation`). Every block has `vendor`.
- OpenAI streams a `server_tool_use` on `output_item.added` (in_progress) and again on `output_item.done` (completed, same id), plus a result block when `action.sources` exist.
- Streaming citations (OpenAI `output_text.annotation.added`, Claude `citations_delta`) are yielded as `{type: "text", text: "", citations}`.
- Claude document citations without a URL are resolved through fetched document titles.
- Claude result blocks keep the verbatim API block in `raw`. `formatMessages` echoes Claude `server_tool_use` + result pairs (both halves required), drops OpenAI-origin or unpaired blocks, and sends text back without citations. OpenAI and the other adapters drop research blocks from history.
- `pause_turn` is continued internally (`MAX_PAUSE_CONTINUATIONS` = 3) by appending the paused raw content as an assistant turn. This bypasses `assertNoPrefill`. Usage is summed across rounds. If the limit is hit, the result is an `ErrorBlock`.
- `webSearchCost` is per search call: OpenAI counts `web_search_call` items except `open_page` / `find`, and Claude uses `usage.server_tool_use.web_search_requests`. `UsageResponse` adds `webSearchCount` / `webFetchCount`.

**OpenAI Image Generation Streaming:**
Snowgoose streams images through `OpenAIAdapter.streamResponse()` using the Responses API
`image_generation` tool (built by `buildImageGenerationTool` from `OpenAIImageGenerationOptions`;
`partial_images` defaults to 2). Each image call yields `ImageDataBlock`s that share the `ig_...`
item id: partial previews (`isPartial: true`, `partialImageIndex`) from
`response.image_generation_call.partial_image`, then the final full-resolution image
(`isPartial: false`) from `response.output_item.done` (with `response.completed` output as a
fallback). Consumers should replace the displayed image by id and persist only the final block.
For multi-turn edits, include an `ImageGenerationCallBlock` (`{type: "image_generation_call", id}`)
in history (or use `previousResponseId`); `mapMessagesToApiInput` sends it as a reference and drops
other image data. Moderation failures surface as an `ErrorBlock` with `code: "moderation_blocked"`.

**Grok Imagine (image generation / editing):**
`GrokAdapter` routes to the Imagine endpoints when the model name contains `-image` (e.g.
`grok-imagine-image-2.0`), or `useImageGeneration`/`openaiImageGenerationOptions` is set, and the
model is `isImageGeneration`. Both `/images/generations` and `/images/edits` go through the OpenAI
SDK's generic `client.post()`. xAI's edit endpoint is JSON-only, and `images.edit()` sends multipart.
- Options reuse `OpenAIImageGenerationOptions`: `n`, `aspectRatio`, `resolution` (`1k`/`1.5k`/`2k`),
  `quality`, `action`, `user`. `size: "WxH"` maps to the nearest supported aspect ratio when
  `aspectRatio` is omitted. `quality` is sent only to `grok-imagine-image-2.0`
  (`-quality`/`-pro` redirect there as of Nov 2 2026). There, `high`/`xhigh`/`max` map to `medium`.
- Multi-turn editing is stateless: `action: "auto"` (default) calls `/images/edits` whenever
  `collectEditImages` finds a source. Sources are either explicit `openaiImageEditOptions.image`, or
  the most recent non-partial assistant image followed by the latest user message's images and
  `visionUrl`. Order matters: the prior image is `<IMAGE_0>` and sets the aspect ratio. Max 5 sources.
  One source is sent as `image` without `aspect_ratio`; several are sent as `images`.
  `action: "generate"` ignores history. `action: "edit"` with no source returns an `ErrorBlock`.
- Always requests `b64_json` (xAI URLs expire, and history must be re-sendable). Returns
  `ImageDataBlock`s with synthetic `grok_img_*` ids. Cost comes from `usage.cost_in_usd_ticks`
  (1e10 ticks = $1). `streamResponse` yields the images, then a `meta` block.

**OpenAI API Pattern:**
The OpenAI adapter uses `client.responses.create()` for the latest OpenAI API format. New OpenAI-compatible adapters should follow this pattern.
