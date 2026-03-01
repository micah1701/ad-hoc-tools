import { Request, Response, NextFunction } from 'express';
import sharp from 'sharp';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';

interface ImageResult {
  image: string;
  format: string;
  width: number;
  height: number;
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
