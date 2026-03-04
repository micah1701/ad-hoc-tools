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

// ── OCR: Driver's Licence / ID card ────────────────────────────────────────

const EXTRACTION_PROMPT = `
You are analyzing a US or Canadian driver's license card image.
Extract the printed data fields. US and Canadian licenses have small AAMVA reference numbers printed before each field value (e.g., "1 SMITH", "2 JOHN ADAM", "3 01/15/1985").

CRITICAL: Only extract values you can directly read from the image. Do NOT guess, infer, or fabricate any field. If a field is blurry, obscured, cut off, or simply not visible, omit it from the output entirely — do not fill it with a placeholder, example value, or assumption. It is far better to return fewer fields than to return incorrect data.

Extract values for these AAMVA field numbers if visible:
- 1: last name
- 2: given names (first and middle)
- 3: date of birth
- 4a: issue date
- 4b: expiration date
- 4d: license/ID number
- 5: document ID
- 8: address (full address)
- 15: sex
- 16: height
- 18: eye color

Return a JSON object with exactly these keys. Omit any key where the value is not clearly legible in the image:
{
  "last_name": "string",
  "given_names": "string",
  "dob": "YYYY-MM-DD",
  "issue_date": "YYYY-MM-DD",
  "expiration_date": "YYYY-MM-DD",
  "license_number": "string",
  "doc_id": "string",
  "address": "string",
  "sex": "M or F",
  "height": "string",
  "eye_color": "string",
  "country": "US or CA",
  "extraction_confidence": "high if 5+ fields found, partial if fewer, low if image quality is poor"
}

US date format on card: MM/DD/YYYY. Canadian date format: YYYY/MM/DD.
Normalize ALL dates to YYYY-MM-DD in your output.
Return ONLY the JSON object, no explanation or markdown.
`.trim();

interface OcrIdResult {
  data: Record<string, string>;
  model: string;
}

/**
 * POST /ai/ocr-id
 * Body: { image: string (base64, with or without data URI prefix),
 *         mediaType?: string (default "image/jpeg"),
 *         model?: string }
 */
export const ocrId = async (req: Request, res: Response, next: NextFunction) => {
  try {
    // model needs text+vision. When possible, use private: qwen3-vl-235b-a22b
    // openai-gpt-4o-2024-11-20 works quick but is not private (and costs more)
    const { image, mediaType = 'image/jpeg', model = 'qwen3-vl-235b-a22b' } = req.body;

    // Strip optional data-URI prefix (e.g. "data:image/jpeg;base64,...")
    const base64Data = (image as string).replace(/^data:[^;]+;base64,/, '');
    const dataUri = `data:${mediaType};base64,${base64Data}`;

    const completion = await openai.chat.completions.create({
      model: model || config.ai.defaultModel,
      max_tokens: 512,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: { url: dataUri },
            },
            {
              type: 'text',
              text: EXTRACTION_PROMPT,
            },
          ],
        },
      ],
    });

    const rawText = completion.choices[0]?.message?.content ?? '';

    // Strip any accidental markdown fences the model may wrap around the JSON
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, 'No JSON found in AI response');
    }

    const parsed = JSON.parse(jsonMatch[0]);

    const response: ApiResponse<OcrIdResult> = {
      success: true,
      data: {
        data: parsed,
        model: completion.model,
      },
      message: 'ID OCR completed successfully',
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

  const isDallE2 = config.ai.imageModel === 'dall-e-2';

  // dall-e-2: square at exactly 256, 512, or 1024.
  // gpt-image models: pick the closest supported aspect ratio — portrait (2:3 = 1024×1536),
  // landscape (3:2 = 1536×1024), or square (1:1 = 1024×1024) — so the canvas wastes as
  // little vertical space as possible and crown-to-neck headroom is preserved.
  let targetW: number;
  let targetH: number;
  let sizeStr: string;

  if (isDallE2) {
    const sq: 256 | 512 | 1024 = origWidth <= 256 && origHeight <= 256 ? 256
      : origWidth <= 512 && origHeight <= 512 ? 512
      : 1024;
    targetW = targetH = sq;
    sizeStr = `${sq}x${sq}`;
  } else {
    const ratio = origWidth / origHeight;
    if (ratio < 0.8) {
      targetW = 1024; targetH = 1536; // portrait 2:3
    } else if (ratio > 1.25) {
      targetW = 1536; targetH = 1024; // landscape 3:2
    } else {
      targetW = 1024; targetH = 1024; // square 1:1
    }
    sizeStr = `${targetW}x${targetH}`;
  }

  // Scale + letterbox offsets so faceRegion coordinates map into the canvas.
  const scale   = Math.min(targetW / origWidth, targetH / origHeight);
  const scaledW = Math.round(origWidth  * scale);
  const scaledH = Math.round(origHeight * scale);
  const offsetX = Math.round((targetW - scaledW) / 2);
  const offsetY = Math.round((targetH - scaledH) / 2);

  // RGBA PNG at the target canvas dimensions.
  const pngBuffer = await sharp(imageBuffer)
    .resize(targetW, targetH, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
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
      create: { width: targetW, height: targetH, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
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
      create: { width: targetW, height: targetH, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } },
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
      prompt: `This image is a cropped facial photo taken from an ID card. 
    The transparent area of the mask marks the regions containing surface artifacts.

    Perform a strictly conservative restoration limited to the masked regions only. 
    Remove unwanted overlays such as security lines, guilloche or wavy patterns, microprinting, holograms, watermarks, or any unnatural marks that differ from real skin or hair tones. 
    Do not modify or generate any new facial features, lighting, or proportions. 
    Ensure all geometry—eyes, nose, mouth, jawline, cheeks, forehead—matches the unmasked source exactly. 
    If artifact removal leaves gaps, blend nearby true pixels smoothly without synthesizing new anatomy. 
    Outside the masked regions, leave every pixel completely unchanged. 
    Do not invent or beautify clothing, hair, or background. 
    Provide a clean, realistic version of the same person with overlays removed, suitable for precise facial‑recognition embedding.`,
      n: 1,
      size: sizeStr as '256x256' | '512x512' | '1024x1024' | '1024x1536' | '1536x1024',
      // response_format is only supported by dall-e-2; GPT image models return base64 by default.
      ...(isDallE2 && { response_format: 'b64_json' as const }),
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
