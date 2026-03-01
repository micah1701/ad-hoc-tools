import { Request, Response, NextFunction } from 'express';
import sharp from 'sharp';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';
import { detectLargestFace } from '../utils/faceDetection';

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
    } = req.body;

    const paddingFraction = Math.min(1, Math.max(0, Number(padding)));
    const inputBuffer = base64ToBuffer(image);

    // ── Step 1: Face detection on the RAW original image ────────────────────
    // Detection must run on the unmodified input so the model sees natural
    // photographic pixel distributions (what it was trained on). Preprocessing
    // distorts histograms enough to cause systematic missed detections.
    const faceBox = await detectLargestFace(inputBuffer);

    // Helper: build the denoise+contrast Sharp pipeline for any source buffer
    function buildEnhancePipeline(src: Buffer): sharp.Sharp {
      let p = sharp(src);
      if (denoise) {
        p = p
          .median(3)                                        // suppress compression / hologram noise
          .sharpen({ sigma: 0.8, m1: 0.5, m2: 2 });       // recover edge detail lost to median
      }
      if (contrastEnhance) {
        p = p
          .normalise({ lower: 1, upper: 99 })              // stretch histogram, clip 1% outliers
          .clahe({ width: 3, height: 3, maxSlope: 3 });    // adaptive local contrast (uneven lighting)
      }
      return p;
    }

    if (!faceBox) {
      // Graceful fallback: apply preprocessing to the full image and return
      const preprocessed = await buildEnhancePipeline(inputBuffer).png().toBuffer();
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

    // ── Step 3: Apply denoise + contrast enhancement to the cropped region ───
    let cropBuffer = await buildEnhancePipeline(rawCrop).png().toBuffer();

    // ── Step 4: Blur background (optional) ──────────────────────────────────
    if (blurBackground) {
      // Face position within the crop (padding offsets are the face's origin)
      const faceInCropLeft   = Math.round(faceBox.x) - cropLeft;
      const faceInCropTop    = Math.round(faceBox.y) - cropTop;
      const faceInCropWidth  = Math.min(Math.round(faceBox.width),  cropWidth  - faceInCropLeft);
      const faceInCropHeight = Math.min(Math.round(faceBox.height), cropHeight - faceInCropTop);

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
