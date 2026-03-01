import { Request, Response, NextFunction } from 'express';
import sharp from 'sharp';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';
import { detectLargestFace } from '../utils/faceDetection';
import { aiCleanFaceImage } from './ai.controller';

interface ImageResult {
  image: string;
  format: string;
  width: number;
  height: number;
  faceDetected?: boolean;
}

/**
 * Strips the data URI prefix (e.g. "data:image/png;base64,") and returns a Buffer.
 */
function base64ToBuffer(input: string): Buffer {
  const base64Data = input.replace(/^data:[^;]+;base64,/, '');
  return Buffer.from(base64Data, 'base64');
}

/**
 * Horizontal median filter — suppresses vertical stripe patterns (e.g. anti-forgery wavy lines
 * on drivers licenses) while preserving vertical edges (face outline, nose, jawline).
 * Window is 1px tall × (2*windowHalf+1)px wide, applied per channel.
 */
async function suppressVerticalStripes(buf: Buffer, windowHalf = 7): Promise<Buffer> {
  const { data, info } = await sharp(buf).raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  const src = new Uint8Array(data);
  const dst = new Uint8Array(src.length);
  const winSize = windowHalf * 2 + 1;
  const mid = Math.floor(winSize / 2);
  const samples = new Uint8Array(winSize);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < winSize; i++) {
          const nx = Math.max(0, Math.min(width - 1, x - windowHalf + i));
          samples[i] = src[(y * width + nx) * channels + c];
        }
        // Partial selection sort to find the median without a full sort
        const tmp = samples.slice();
        for (let i = 0; i <= mid; i++) {
          let minIdx = i;
          for (let j = i + 1; j < winSize; j++) {
            if (tmp[j] < tmp[minIdx]) minIdx = j;
          }
          [tmp[i], tmp[minIdx]] = [tmp[minIdx], tmp[i]];
        }
        dst[(y * width + x) * channels + c] = tmp[mid];
      }
    }
  }

  return sharp(Buffer.from(dst), { raw: { width, height, channels } }).png().toBuffer();
}

/**
 * POST /image/resize
 * Body: { image: string (base64), width?: number, height?: number, maintainAspectRatio?: boolean }
 */
export const resizeImage = async (req: Request, res: Response, next: NextFunction) => {
  console.log("resizeImage function called in image.controller")
  try {
    const { image, width, height, maintainAspectRatio = true } = req.body;

    if (!width && !height) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'At least one of width or height is required');
    }

    const inputBuffer = base64ToBuffer(image);

    const resizeOptions: sharp.ResizeOptions = {
      width: width ? Number(width) : undefined,
      height: height ? Number(height) : undefined,
      fit: maintainAspectRatio ? 'inside' : 'fill',
      withoutEnlargement: false,
    };

    const outputBuffer = await sharp(inputBuffer).resize(resizeOptions).toBuffer();
    const metadata = await sharp(outputBuffer).metadata();

    const response: ApiResponse<ImageResult> = {
      success: true,
      data: {
        image: outputBuffer.toString('base64'),
        format: metadata.format ?? 'unknown',
        width: metadata.width ?? 0,
        height: metadata.height ?? 0,
      },
      message: 'Image resized successfully',
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * POST /image/face-preprocess
 * Denoises, enhances contrast, detects the largest face, crops to it with padding,
 * and optionally blurs the background — optimised for ID card / photo-ID inputs.
 * Body: { image: string (base64), padding?: number (0–1, default 0.2),
 *         denoise?: boolean (default true), contrastEnhance?: boolean (default true),
 *         blurBackground?: boolean (default true) }
 */
export const facePreprocess = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const {
      image,
      padding = 0.2,
      denoise = true,
      contrastEnhance = true,
      blurBackground = true,
      grayscale = true,
      useAi = false,
    } = req.body;

    const paddingFraction = Math.min(1, Math.max(0, Number(padding)));
    const inputBuffer = base64ToBuffer(image);

    // ── Step 1: Face detection on the RAW original image ────────────────────
    // Detection must run on the unmodified input so the model sees natural
    // photographic pixel distributions (what it was trained on). Preprocessing
    // distorts histograms enough to cause systematic missed detections.
    const faceBox = await detectLargestFace(inputBuffer);

    // Helper: full enhancement pipeline applied to a cropped buffer
    async function buildEnhancePipeline(src: Buffer): Promise<Buffer> {
      // 1. Greyscale first — eliminates the chromatic component of color-printed
      //    anti-forgery lines (rainbow gradients, hue-specific security printing)
      let buf = grayscale
        ? await sharp(src).greyscale().png().toBuffer()
        : src;

      // 2. Horizontal median — suppresses residual luminance variations from
      //    vertical wavy lines while preserving vertical face edges
      if (denoise) {
        buf = await suppressVerticalStripes(buf);
      }

      // 3. Point-noise removal + mild edge recovery
      let p = sharp(buf);
      if (denoise) {
        p = p
          .median(3)                                        // compression / hologram point noise
          .sharpen({ sigma: 0.8, m1: 0.5, m2: 1 });       // mild edge recovery (m2 reduced from 2→1)
      }

      // 4. Adaptive local contrast only — no global normalise() which amplifies
      //    stripe contrast before CLAHE can work against it
      if (contrastEnhance) {
        p = p.clahe({ width: 4, height: 4, maxSlope: 3 }); // larger tiles (3→4) = better local adaptation
      }

      // 5. 3-channel greyscale stack — face networks expecting 3-channel input
      //    receive identical R/G/B so they work without colour noise
      if (grayscale) {
        p = p.toColourspace('srgb');
      }

      return p.png().toBuffer();
    }

    if (!faceBox) {
      // Graceful fallback: apply preprocessing to the full image and return
      const preprocessed = await buildEnhancePipeline(inputBuffer);
      const meta = await sharp(preprocessed).metadata();
      const response: ApiResponse<ImageResult> = {
        success: true,
        data: {
          image: preprocessed.toString('base64'),
          format: 'png',
          width: meta.width ?? 0,
          height: meta.height ?? 0,
          faceDetected: false,
        },
        message: 'Image preprocessed successfully (no face detected)',
      };
      return res.status(HttpStatus.OK).json(response);
    }

    // ── Step 2: Crop from the RAW original using the detected bounding box ───
    const rawMeta = await sharp(inputBuffer).metadata();
    const imgW = rawMeta.width ?? 0;
    const imgH = rawMeta.height ?? 0;

    const padX = Math.round(faceBox.width * paddingFraction);
    const padY = Math.round(faceBox.height * paddingFraction);

    const cropLeft   = Math.max(0, Math.round(faceBox.x) - padX);
    const cropTop    = Math.max(0, Math.round(faceBox.y) - padY);
    const cropWidth  = Math.min(imgW - cropLeft, Math.round(faceBox.width)  + padX * 2);
    const cropHeight = Math.min(imgH - cropTop,  Math.round(faceBox.height) + padY * 2);

    const rawCrop = await sharp(inputBuffer)
      .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
      .toBuffer();

    // Face position within the crop (needed for AI clean and background blur)
    const faceInCropLeft   = Math.round(faceBox.x) - cropLeft;
    const faceInCropTop    = Math.round(faceBox.y) - cropTop;
    const faceInCropWidth  = Math.min(Math.round(faceBox.width),  cropWidth  - faceInCropLeft);
    const faceInCropHeight = Math.min(Math.round(faceBox.height), cropHeight - faceInCropTop);

    // ── Step 2.5: AI overlay removal (optional) ──────────────────────────────
    // Sends the raw crop to an image-edit model with an inpainting mask that
    // protects the face region, asking the model to erase non-facial overlays
    // (security patterns, holograms, watermarks) from the surrounding area.
    let processBuffer = rawCrop;
    if (useAi) {
      processBuffer = await aiCleanFaceImage(rawCrop, {
        left: faceInCropLeft,
        top: faceInCropTop,
        width: faceInCropWidth,
        height: faceInCropHeight,
      });
    }

    // ── Step 3: Apply denoise + contrast enhancement to the cropped region ───
    let cropBuffer = await buildEnhancePipeline(processBuffer);

    // ── Step 4: Blur background (optional) ──────────────────────────────────
    if (blurBackground) {
      const blurredBg  = await sharp(cropBuffer).blur(20).toBuffer();
      const sharpFace  = await sharp(cropBuffer)
        .extract({ left: faceInCropLeft, top: faceInCropTop, width: faceInCropWidth, height: faceInCropHeight })
        .toBuffer();

      cropBuffer = await sharp(blurredBg)
        .composite([{ input: sharpFace, left: faceInCropLeft, top: faceInCropTop }])
        .png()
        .toBuffer();
    }

    // ── Response ─────────────────────────────────────────────────────────────
    const outputMeta = await sharp(cropBuffer).metadata();
    const response: ApiResponse<ImageResult> = {
      success: true,
      data: {
        image: cropBuffer.toString('base64'),
        format: 'png',
        width: outputMeta.width ?? 0,
        height: outputMeta.height ?? 0,
        faceDetected: true,
      },
      message: 'Image preprocessed successfully',
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * POST /image/web-optimize
 * Body: { image: string (base64), quality?: number (1-100, default 80) }
 */
export const webOptimize = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { image, quality = 80 } = req.body;

    const parsedQuality = Number(quality);
    if (parsedQuality < 1 || parsedQuality > 100) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'quality must be between 1 and 100');
    }

    const inputBuffer = base64ToBuffer(image);

    const outputBuffer = await sharp(inputBuffer)
      .webp({ quality: parsedQuality })
      .toBuffer();

    const metadata = await sharp(outputBuffer).metadata();

    const response: ApiResponse<ImageResult> = {
      success: true,
      data: {
        image: outputBuffer.toString('base64'),
        format: 'webp',
        width: metadata.width ?? 0,
        height: metadata.height ?? 0,
      },
      message: 'Image optimized for web successfully',
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};
