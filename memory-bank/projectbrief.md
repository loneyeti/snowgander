# Project Brief: AI Vendor Abstraction Layer (Snowgander)

## Core Goal

To create a robust, maintainable, and extensible TypeScript library that acts as an abstraction layer over various AI vendor APIs (OpenAI, Anthropic, Google, OpenRouter). This library allows a consuming application to interact with different AI models through a standardized interface, simplifying integration and vendor switching.

## Key Requirements

1.  **Standardized Interface:** Define a common `AIVendorAdapter` interface that all vendor-specific adapters must implement.
2.  **Factory Pattern:** Implement an `AIVendorFactory` to easily instantiate the correct adapter based on configuration.
3.  **Abstracted Configuration:** Use `VendorConfig` and `ModelConfig` interfaces to decouple vendor/model details from the core application logic. Configuration should be injectable.
4.  **Standardized Content Representation:** Utilize a `ContentBlock` system (`TextBlock`, `ImageBlock`, `ImageDataBlock`, `ThinkingBlock`, etc.) for all message content and AI responses to handle diverse media types consistently.
5.  **Core Functionality:** Support standard chat completion (`generateResponse`, `sendChat`), image generation (`generateImage`), and potentially specialized interactions like MCP tool usage (`sendMCPChat`).
6.  **Capability Reporting:** Adapters should indicate their capabilities (vision, image generation, thinking).
7.  **Cost Calculation:** Incorporate mechanisms to track and calculate the estimated cost of AI interactions based on token usage and model pricing.
8.  **Extensibility:** Design the library to easily accommodate new AI vendors or models in the future.
9.  **Type Safety:** Leverage TypeScript for strong typing throughout the library.

## Target Audience

Developers integrating AI capabilities into applications who need flexibility in choosing AI vendors and models without tightly coupling their codebase to specific APIs.
