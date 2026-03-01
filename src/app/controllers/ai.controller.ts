import { Request, Response, NextFunction } from 'express';
import OpenAI from 'openai';
import { config } from '../../core/config';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';

const openai = new OpenAI({
  baseURL: config.ai.baseUrl,
  apiKey: config.ai.apiKey,
});

interface ChatResult {
  message: string;
  model: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * POST /ai/chat
 * Body: { message: string, model?: string, systemPrompt?: string }
 */
export const chat = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { message, model, systemPrompt } = req.body;

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }

    messages.push({ role: 'user', content: message });

    const completion = await openai.chat.completions.create({
      model: model || config.ai.defaultModel,
      messages,
    });

    const choice = completion.choices[0];
    if (!choice?.message?.content) {
      throw new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, 'No response from AI');
    }

    const response: ApiResponse<ChatResult> = {
      success: true,
      data: {
        message: choice.message.content,
        model: completion.model,
        usage: completion.usage
          ? {
              promptTokens: completion.usage.prompt_tokens,
              completionTokens: completion.usage.completion_tokens,
              totalTokens: completion.usage.total_tokens,
            }
          : undefined,
      },
      message: 'Chat completed successfully',
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};
