import { AIVendorFactory } from "../factory";
import { VendorConfig, ModelConfig, AIVendorAdapter } from "../types";
import { OpenAIAdapter } from "../vendors/openai";
import { AnthropicAdapter } from "../vendors/anthropic";
import { GoogleAIAdapter } from "../vendors/google";
import { OpenRouterAdapter } from "../vendors/openrouter";

// Mock the actual adapter constructors to prevent real API calls/SDK initialization
jest.mock("../vendors/openai");
jest.mock("../vendors/anthropic");
jest.mock("../vendors/google");
jest.mock("../vendors/openrouter");

describe("AIVendorFactory", () => {
  const mockOpenAIConfig: VendorConfig = { apiKey: "openai-key" };
  const mockAnthropicConfig: VendorConfig = { apiKey: "anthropic-key" };
  const mockGoogleConfig: VendorConfig = { apiKey: "google-key" };
  const mockOpenRouterConfig: VendorConfig = {
    apiKey: "openrouter-key",
    baseURL: "https://openrouter.ai/api/v1",
  }; // Example baseURL

  const mockGPT4Model: ModelConfig = {
    apiName: "gpt-4-turbo",
    isVision: true,
    isImageGeneration: false, // GPT-4 itself doesn't generate images, DALL-E does
    isThinking: false,
    inputTokenCost: 10 / 1_000_000,
    outputTokenCost: 30 / 1_000_000,
  };

  const mockClaude3Model: ModelConfig = {
    apiName: "claude-3-opus-20240229",
    isVision: true,
    isImageGeneration: false,
    isThinking: true, // Anthropic supports thinking
    inputTokenCost: 15 / 1_000_000,
    outputTokenCost: 75 / 1_000_000,
  };

  const mockGeminiModel: ModelConfig = {
    apiName: "gemini-1.5-pro-latest",
    isVision: true,
    isImageGeneration: false, // Gemini API doesn't directly generate images via chat
    isThinking: false,
    inputTokenCost: 7 / 1_000_000,
    outputTokenCost: 21 / 1_000_000,
  };

  beforeEach(() => {
    // Clear mock constructor calls before each test
    // We cannot directly clear the private static vendorConfigs map,
    // but setVendorConfig will overwrite entries for each test.
    (OpenAIAdapter as jest.Mock).mockClear();
    (AnthropicAdapter as jest.Mock).mockClear();
    (GoogleAIAdapter as jest.Mock).mockClear();
    (OpenRouterAdapter as jest.Mock).mockClear();

    // Mock the instances returned by constructors to have basic properties
    const mockAdapterInstance = {
      isVisionCapable: false,
      isImageGenerationCapable: false,
      isThinkingCapable: false,
      inputTokenCost: 0,
      outputTokenCost: 0,
      // Add mock methods if needed for deeper tests later
      generateResponse: jest.fn(),
      generateImage: jest.fn(),
      sendChat: jest.fn(),
      sendMCPChat: jest.fn(),
    };

    (OpenAIAdapter as jest.Mock).mockImplementation(
      (config: VendorConfig, model: ModelConfig) => ({
        ...mockAdapterInstance,
        constructorArgs: { config, model }, // Store args for inspection
        isVisionCapable: model.isVision,
        isImageGenerationCapable: model.isImageGeneration, // Should be false for GPT-4 itself
        inputTokenCost: model.inputTokenCost,
        outputTokenCost: model.outputTokenCost,
      })
    );
    (AnthropicAdapter as jest.Mock).mockImplementation(
      (config: VendorConfig, model: ModelConfig) => ({
        ...mockAdapterInstance,
        constructorArgs: { config, model },
        isVisionCapable: model.isVision,
        isThinkingCapable: model.isThinking,
        inputTokenCost: model.inputTokenCost,
        outputTokenCost: model.outputTokenCost,
      })
    );
    (GoogleAIAdapter as jest.Mock).mockImplementation(
      (config: VendorConfig, model: ModelConfig) => ({
        ...mockAdapterInstance,
        constructorArgs: { config, model },
        isVisionCapable: model.isVision,
        inputTokenCost: model.inputTokenCost,
        outputTokenCost: model.outputTokenCost,
      })
    );
    (OpenRouterAdapter as jest.Mock).mockImplementation(
      (config: VendorConfig, model: ModelConfig) => ({
        ...mockAdapterInstance,
        constructorArgs: { config, model },
        isVisionCapable: model.isVision, // Assuming OpenRouter passes vision through
        inputTokenCost: model.inputTokenCost, // OpenRouter costs might differ, but use model's for test
        outputTokenCost: model.outputTokenCost,
      })
    );
  });

  // Test setting config indirectly by checking if getAdapter works later
  it("should allow setting vendor configuration", () => {
    // No direct assertion possible as vendorConfigs is private.
    // We'll verify this implicitly in the tests below.
    AIVendorFactory.setVendorConfig("openai", mockOpenAIConfig);
    // If setVendorConfig didn't work, subsequent getAdapter calls would fail.
  });

  it("should throw an error if getting an adapter for an unconfigured vendor", () => {
    expect(() => {
      AIVendorFactory.getAdapter("unregistered", mockGPT4Model);
    }).toThrow("No configuration found for vendor: unregistered"); // Corrected error message
  });

  it("should return an OpenAIAdapter instance for 'openai' vendor", () => {
    AIVendorFactory.setVendorConfig("openai", mockOpenAIConfig); // Use setVendorConfig
    const adapter = AIVendorFactory.getAdapter("openai", mockGPT4Model);
    // expect(adapter).toBeInstanceOf(OpenAIAdapter); // Fails due to mock returning plain object
    const mockInstance = adapter as any; // Cast to access mock properties
    expect(OpenAIAdapter).toHaveBeenCalledWith(mockOpenAIConfig, mockGPT4Model); // Verify constructor call
    expect(mockInstance.constructorArgs.config).toEqual(mockOpenAIConfig);
    expect(mockInstance.constructorArgs.model).toEqual(mockGPT4Model);
    expect(mockInstance.isVisionCapable).toBe(true);
    expect(mockInstance.isImageGenerationCapable).toBe(false); // Based on mockGPT4Model
    expect(mockInstance.isThinkingCapable).toBe(false);
    expect(mockInstance.inputTokenCost).toBe(mockGPT4Model.inputTokenCost);
    expect(mockInstance.outputTokenCost).toBe(mockGPT4Model.outputTokenCost);
  });

  it("should return an AnthropicAdapter instance for 'anthropic' vendor", () => {
    AIVendorFactory.setVendorConfig("anthropic", mockAnthropicConfig); // Use setVendorConfig
    const adapter = AIVendorFactory.getAdapter("anthropic", mockClaude3Model);
    // expect(adapter).toBeInstanceOf(AnthropicAdapter); // Fails due to mock
    const mockInstance = adapter as any;
    expect(AnthropicAdapter).toHaveBeenCalledWith(
      // Verify constructor call
      mockAnthropicConfig,
      mockClaude3Model
    );
    expect(mockInstance.constructorArgs.config).toEqual(mockAnthropicConfig);
    expect(mockInstance.constructorArgs.model).toEqual(mockClaude3Model);
    expect(mockInstance.isVisionCapable).toBe(true);
    expect(mockInstance.isImageGenerationCapable).toBe(false);
    expect(mockInstance.isThinkingCapable).toBe(true); // Based on mockClaude3Model
    expect(mockInstance.inputTokenCost).toBe(mockClaude3Model.inputTokenCost);
    expect(mockInstance.outputTokenCost).toBe(mockClaude3Model.outputTokenCost);
  });

  it("should return a GoogleAIAdapter instance for 'google' vendor", () => {
    AIVendorFactory.setVendorConfig("google", mockGoogleConfig); // Use setVendorConfig
    const adapter = AIVendorFactory.getAdapter("google", mockGeminiModel);
    // expect(adapter).toBeInstanceOf(GoogleAIAdapter); // Fails due to mock
    const mockInstance = adapter as any;
    expect(GoogleAIAdapter).toHaveBeenCalledWith(
      // Verify constructor call
      mockGoogleConfig,
      mockGeminiModel
    );
    expect(mockInstance.constructorArgs.config).toEqual(mockGoogleConfig);
    expect(mockInstance.constructorArgs.model).toEqual(mockGeminiModel);
    expect(mockInstance.isVisionCapable).toBe(true);
    expect(mockInstance.isImageGenerationCapable).toBe(false);
    expect(mockInstance.isThinkingCapable).toBe(false);
    expect(mockInstance.inputTokenCost).toBe(mockGeminiModel.inputTokenCost);
    expect(mockInstance.outputTokenCost).toBe(mockGeminiModel.outputTokenCost);
  });

  it("should return an OpenRouterAdapter instance for 'openrouter' vendor", () => {
    AIVendorFactory.setVendorConfig("openrouter", mockOpenRouterConfig); // Use setVendorConfig
    // OpenRouter can route to various models, let's test with a GPT-like model config
    const adapter = AIVendorFactory.getAdapter("openrouter", mockGPT4Model);
    // expect(adapter).toBeInstanceOf(OpenRouterAdapter); // Fails due to mock
    const mockInstance = adapter as any;
    expect(OpenRouterAdapter).toHaveBeenCalledWith(
      // Verify constructor call
      mockOpenRouterConfig,
      mockGPT4Model
    );
    expect(mockInstance.constructorArgs.config).toEqual(mockOpenRouterConfig);
    expect(mockInstance.constructorArgs.model).toEqual(mockGPT4Model);
    // Assuming OpenRouter adapter reflects underlying model capabilities from ModelConfig
    expect(mockInstance.isVisionCapable).toBe(true);
    expect(mockInstance.isImageGenerationCapable).toBe(false);
    expect(mockInstance.isThinkingCapable).toBe(false);
    expect(mockInstance.inputTokenCost).toBe(mockGPT4Model.inputTokenCost);
    expect(mockInstance.outputTokenCost).toBe(mockGPT4Model.outputTokenCost);
  });

  it("should throw an error for an unsupported vendor string", () => {
    // Set a dummy config first to bypass the initial config check
    AIVendorFactory.setVendorConfig("unsupportedvendor", {
      apiKey: "dummy-key",
    });
    expect(() => {
      AIVendorFactory.getAdapter("unsupportedvendor", mockGPT4Model);
    }).toThrow("Unsupported vendor: unsupportedvendor");
  });

  // Add a test to ensure case-insensitivity of vendor names
  it("should handle vendor names case-insensitively", () => {
    AIVendorFactory.setVendorConfig("openai", mockOpenAIConfig);
    const adapterUpper = AIVendorFactory.getAdapter("OpenAI", mockGPT4Model);
    const adapterMixed = AIVendorFactory.getAdapter("oPeNaI", mockGPT4Model);
    // expect(adapterUpper).toBeInstanceOf(OpenAIAdapter); // Fails due to mock
    // expect(adapterMixed).toBeInstanceOf(OpenAIAdapter); // Fails due to mock
    expect(OpenAIAdapter).toHaveBeenCalledTimes(2); // Check constructor calls
    expect(OpenAIAdapter).toHaveBeenCalledWith(mockOpenAIConfig, mockGPT4Model); // Check args on last call
  });
});
