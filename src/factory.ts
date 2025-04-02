// Removed Prisma Model import
// Removed application-specific imports (prisma, getApiVendor)
import { AIVendorAdapter, VendorConfig, ModelConfig } from "./types"; // Added ModelConfig import
import { OpenAIAdapter } from "./vendors/openai";
import { AnthropicAdapter } from "./vendors/anthropic";
import { GoogleAIAdapter } from "./vendors/google";
import { OpenRouterAdapter } from "./vendors/openrouter";

export class AIVendorFactory {
  private static vendorConfigs: Map<string, VendorConfig> = new Map();

  static setVendorConfig(vendorName: string, config: VendorConfig) {
    this.vendorConfigs.set(vendorName.toLowerCase(), config);
  }

  // Updated signature: accepts vendorName (string) and modelConfig (ModelConfig)
  // Removed async as it no longer fetches data
  static getAdapter(
    vendorName: string,
    modelConfig: ModelConfig
  ): AIVendorAdapter {
    const lowerVendorName = vendorName.toLowerCase();

    const config = this.vendorConfigs.get(lowerVendorName);
    if (!config) {
      throw new Error(`No configuration found for vendor: ${lowerVendorName}`);
    }

    let adapter: AIVendorAdapter;

    // Use lowerVendorName directly and pass modelConfig to constructors
    switch (lowerVendorName) {
      case "openai":
        adapter = new OpenAIAdapter(config, modelConfig);
        break;
      case "anthropic":
        adapter = new AnthropicAdapter(config, modelConfig);
        break;
      case "google":
        adapter = new GoogleAIAdapter(config, modelConfig);
        break;
      case "openrouter":
        adapter = new OpenRouterAdapter(config, modelConfig);
        break;
      default:
        throw new Error(`Unsupported vendor: ${vendorName}`);
    }

    return adapter;
  }
}
