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
  const origWidth  = meta.width  ?? 512;
  const origHeight = meta.height ?? 512;

  // DALL-E 2 requires square RGBA PNG at exactly 256, 512, or 1024.
  // Choose the smallest supported size that fits both dimensions.
  const targetSize: 256 | 512 | 1024 =
    origWidth <= 256 && origHeight <= 256 ? 256
    : origWidth <= 512 && origHeight <= 512 ? 512
    : 1024;

  // Compute the scale + letterbox offsets used by sharp's `contain` resize so
  // we can map faceRegion coordinates into the square canvas.
  const scale   = Math.min(targetSize / origWidth, targetSize / origHeight);
  const scaledW = Math.round(origWidth  * scale);
  const scaledH = Math.round(origHeight * scale);
  const offsetX = Math.round((targetSize - scaledW) / 2);
  const offsetY = Math.round((targetSize - scaledH) / 2);

  // Square RGBA PNG — both requirements for DALL-E 2 inpainting.
  const pngBuffer = await sharp(imageBuffer)
    .resize(targetSize, targetSize, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .ensureAlpha()
    .png()
    .toBuffer();

  // Build inpainting mask: transparent everywhere (edit), opaque over face (keep).
  // Map faceRegion from original-crop space into the square-canvas space.
  let maskBuffer: Buffer;
  if (faceRegion) {
    const mLeft   = Math.round(faceRegion.left   * scale) + offsetX;
    const mTop    = Math.round(faceRegion.top    * scale) + offsetY;
    const mWidth  = Math.max(1, Math.round(faceRegion.width  * scale));
    const mHeight = Math.max(1, Math.round(faceRegion.height * scale));

    const facePixels = Buffer.alloc(mWidth * mHeight * 4, 255);
    maskBuffer = await sharp({
      create: { width: targetSize, height: targetSize, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
    })
      .composite([{
        input: facePixels,
        raw: { width: mWidth, height: mHeight, channels: 4 },
        left: mLeft,
        top: mTop,
      }])
      .png()
      .toBuffer();
  } else {
    // No region known — fully transparent mask lets the model clean the whole image
    maskBuffer = await sharp({
      create: { width: targetSize, height: targetSize, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
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
      prompt: 'This is a cropped face photo taken from an ID card. '
      +'Clean and prepare it for facial-recognition analysis. '
      +'Remove all visual artifacts such as overlaid anti-counterfeiting lines, guilloche patterns, '
      +'microprinting, Ben Day dots, holograms, text, or background graphics. '
      +'Eliminate any unnatural lines or shapes that don’t match real human skin or facial features. '
      +'Keep the person’s face, lighting, and proportions exactly as in the original—no new geometry or '
      +'expressions. It’s acceptable to smooth or blur small areas where artifacts are removed instead '
      +'of inventing detail. Keep existing hair and clothing; don’t add hands, timestamps, or borders. '
      +'The final image should show the same unaltered face against a plain, unobtrusive, solid background, '
      +'ready for accurate facial embedding.',
      n: 1,
      size: `${targetSize}x${targetSize}` as '256x256' | '512x512' | '1024x1024',
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

  // OpenAI returns at targetSize×targetSize. Crop out the letterbox padding and
  // resize back to original dimensions so callers' face coordinates stay valid.
  return sharp(Buffer.from(b64, 'base64'))
    .extract({ left: offsetX, top: offsetY, width: scaledW, height: scaledH })
    .resize(origWidth, origHeight)
    .png()
    .toBuffer();
}
