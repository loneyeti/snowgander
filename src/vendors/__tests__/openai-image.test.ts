import OpenAI from 'openai';
import { OpenAIImageAdapter } from '../openai-image';
import {
  VendorConfig,
  ModelConfig,
  AIRequestOptions,
  NotImplementedError,
  ImageGenerationResponse,
  Chat, // Added missing import
} from '../../types';

// Mock the OpenAI SDK
jest.mock('openai', () => {
  const mockImages = {
    generate: jest.fn(),
    edit: jest.fn(), // Mock edit even if not implemented yet
  };
  return jest.fn().mockImplementation(() => ({
    images: mockImages,
  }));
});

describe('OpenAIImageAdapter', () => {
  let mockClient: OpenAI;
  let vendorConfig: VendorConfig;
  let modelConfig: ModelConfig;
  let adapter: OpenAIImageAdapter;

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    mockClient = new OpenAI({ apiKey: 'test-key' }); // Instantiated mock

    vendorConfig = { apiKey: 'test-key' };
    modelConfig = {
      apiName: 'gpt-image-1', // Example model name
      isVision: false,
      isImageGeneration: true,
      isThinking: false,
      inputTokenCost: 0.01 / 1000000, // Example costs
      outputTokenCost: 0.03 / 1000000,
    };
    adapter = new OpenAIImageAdapter(vendorConfig, modelConfig);
  });

  it('should throw error if API key is missing', () => {
    expect(() => new OpenAIImageAdapter({ apiKey: '' }, modelConfig)).toThrow(
      'openai-image API key is required.'
    );
  });

  describe('generateImage', () => {
    it('should call OpenAI images.generate with correct parameters and return mapped response', async () => {
      const mockApiResponse = {
        created: Date.now() / 1000,
        data: [
          { b64_json: 'base64encodedstring1', revised_prompt: 'revised prompt 1' },
          { b64_json: 'base64encodedstring2', revised_prompt: 'revised prompt 2' },
        ],
      };
      // Setup the mock implementation for generate
      (mockClient.images.generate as jest.Mock).mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: 'gpt-image-1', // Can be overridden by openaiImageGenerationOptions
        messages: [], // Not directly used, prompt extracted
        prompt: 'a cat wearing a hat',
        openaiImageGenerationOptions: {
          n: 2,
          size: '1024x1024',
          quality: 'hd',
          response_format: 'b64_json', // Adapter forces this
          style: 'vivid',
          user: 'test-user',
        },
      };

      const expectedResponse: ImageGenerationResponse = {
        images: [
          { type: 'image_data', mimeType: 'image/png', base64Data: 'base64encodedstring1' },
          { type: 'image_data', mimeType: 'image/png', base64Data: 'base64encodedstring2' },
        ],
        usage: { inputCost: 0, outputCost: 0, totalCost: 0 }, // Placeholder usage
      };

      const response = await adapter.generateImage(options);

      expect(mockClient.images.generate).toHaveBeenCalledTimes(1);
      expect(mockClient.images.generate).toHaveBeenCalledWith({
        model: 'gpt-image-1',
        prompt: 'a cat wearing a hat',
        n: 2,
        quality: 'hd',
        response_format: 'b64_json',
        size: '1024x1024',
        style: 'vivid',
        user: 'test-user',
      });
      expect(response).toEqual(expectedResponse);
    });

    it('should extract prompt from last user message if options.prompt is missing', async () => {
       const mockApiResponse = { data: [{ b64_json: 'test' }] };
      (mockClient.images.generate as jest.Mock).mockResolvedValue(mockApiResponse);

      const options: AIRequestOptions = {
        model: 'gpt-image-1',
        messages: [
            { role: 'user', content: [{ type: 'text', text: 'first message' }] },
            { role: 'assistant', content: [{ type: 'text', text: 'response' }] },
            { role: 'user', content: [{ type: 'text', text: 'last user prompt' }] },
        ],
        // No explicit prompt here
        openaiImageGenerationOptions: { size: '1024x1024' },
      };

      await adapter.generateImage(options);

      expect(mockClient.images.generate).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'last user prompt' })
      );
    });

    it('should throw error if no prompt is available', async () => {
      const options: AIRequestOptions = {
        model: 'gpt-image-1',
        messages: [], // No messages
        // No prompt
        openaiImageGenerationOptions: { size: '1024x1024' },
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        'A text prompt is required for image generation.'
      );
    });

     it('should throw error if openaiImageGenerationOptions are missing', async () => {
      const options: AIRequestOptions = {
        model: 'gpt-image-1',
        messages: [],
        prompt: 'test prompt',
        // Missing openaiImageGenerationOptions
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        'openaiImageGenerationOptions are required for OpenAI image generation.'
      );
    });

    it('should handle API errors gracefully', async () => {
      const apiError = new Error('OpenAI API Error');
      (mockClient.images.generate as jest.Mock).mockRejectedValue(apiError);

      const options: AIRequestOptions = {
        model: 'gpt-image-1',
        prompt: 'test',
        messages: [],
        openaiImageGenerationOptions: { size: '1024x1024' },
      };

      await expect(adapter.generateImage(options)).rejects.toThrow(
        `Failed to generate image: ${apiError.message}`
      );
    });
  });

  // Test unimplemented methods
  describe('Unimplemented Methods', () => {
    it('generateResponse should throw NotImplementedError', async () => {
      const options: AIRequestOptions = { model: 'test', messages: [] };
      await expect(adapter.generateResponse(options)).rejects.toThrow(NotImplementedError);
      await expect(adapter.generateResponse(options)).rejects.toThrow(
        'openai-image adapter does not support generateResponse. Use generateImage or editImage.'
      );
    });

    it('sendChat should throw NotImplementedError', async () => {
       const chat: Chat = { model: 'test', responseHistory: [], prompt: 'hi', visionUrl: null, imageURL: null, maxTokens: null, budgetTokens: null };
       await expect(adapter.sendChat(chat)).rejects.toThrow(NotImplementedError);
       await expect(adapter.sendChat(chat)).rejects.toThrow(
         'openai-image adapter does not support sendChat. Use generateImage or editImage.'
       );
    });

     it('editImage should throw NotImplementedError', async () => {
       const options: AIRequestOptions = { model: 'test', messages: [], openaiImageEditOptions: { image: [] } }; // Provide minimal required options
       await expect(adapter.editImage(options)).rejects.toThrow(NotImplementedError);
       await expect(adapter.editImage(options)).rejects.toThrow(
         'openai-image adapter editImage not fully implemented yet (image input handling needed).'
       );
     });

     // sendMCPChat is optional and not implemented, so no test needed unless added later
  });
});
