import { Request, Response, NextFunction } from 'express';
import OpenAI, { toFile } from 'openai';
import sharp from 'sharp';
import { config } from '../../core/config';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';

const openai = new OpenAI({
  baseURL: config.ai.baseUrl,
  apiKey: config.ai.apiKey,
});

// Separate client for image editing — requires a provider that implements
// POST /images/edits (inpainting). OpenAI dall-e-2 is the canonical choice;
// Venice.ai and most OpenAI-compatible proxies only expose /images/generations.
const openaiImage = new OpenAI({
  baseURL: config.ai.imageBaseUrl,
  apiKey: config.ai.imageApiKey,
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

// ── AI image clean utility ──────────────────────────────────────────────────

export interface FaceRegion {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Sends a cropped face image to an OpenAI-compatible image-edit endpoint,
 * instructing the model to remove non-facial overlays (security patterns,
 * holograms, watermarks, printed text) while protecting the face area via
 * an inpainting mask (opaque = keep, transparent = edit).
 */
export async function aiCleanFaceImage(
  imageBuffer: Buffer,
  faceRegion?: FaceRegion,
): Promise<Buffer> {
  const meta = await sharp(imageBuffer).metadata();
  const width = meta.width ?? 512;
  const height = meta.height ?? 512;

  // Ensure the image is a clean PNG before sending
  const pngBuffer = await sharp(imageBuffer).png().toBuffer();

  // Build inpainting mask: transparent everywhere (edit), opaque over face (keep)
  let maskBuffer: Buffer;
  if (faceRegion) {
    // Fill face bounding box with fully opaque pixels
    const facePixels = Buffer.alloc(faceRegion.width * faceRegion.height * 4, 255);
    maskBuffer = await sharp({
      create: { width, height, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
    })
      .composite([{
        input: facePixels,
        raw: { width: faceRegion.width, height: faceRegion.height, channels: 4 },
        left: faceRegion.left,
        top: faceRegion.top,
      }])
      .png()
      .toBuffer();
  } else {
    // No region known — fully transparent mask lets the model clean the whole image
    maskBuffer = await sharp({
      create: { width, height, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
    }).png().toBuffer();
  }

  const [imageFile, maskFile] = await Promise.all([
    toFile(pngBuffer, 'face.png', { type: 'image/png' }),
    toFile(maskBuffer, 'mask.png', { type: 'image/png' }),
  ]);

  let response;
  try {
    response = await openaiImage.images.edit({
      model: config.ai.imageModel,
      image: imageFile,
      mask: maskFile,
      prompt:
        'Remove all non-facial overlays from this face photo: ID card security patterns, ' +
        'guilloche lines, holographic overlays, watermarks, printed text, and background ' +
        'graphics. Preserve the person\'s face, skin tone, hair, and facial features ' +
        'exactly as they appear — do not alter or regenerate any facial detail.',
      n: 1,
      response_format: 'b64_json',
    });
  } catch (err: any) {
    // Many OpenAI-compatible providers (e.g. Venice.ai) do not implement
    // POST /images/edits. Fall back to the original image so the rest of
    // the face-preprocess pipeline can still run.
    const status = err?.status ?? err?.response?.status;
    const isUnsupported = status === 404 || status === 501;
    if (isUnsupported) {
      console.warn(
        `[aiCleanFaceImage] ${config.ai.imageBaseUrl} returned ${status} for images/edits — ` +
        'falling back to unmodified crop. Set AI_IMAGE_BASE_URL to a provider that supports inpainting.',
      );
      return imageBuffer;
    }
    throw err;
  }

  const b64 = response.data?.[0]?.b64_json;
  if (!b64) throw new Error('AI image clean returned no image data');

  return Buffer.from(b64, 'base64');
}
