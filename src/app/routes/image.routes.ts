import { Router } from 'express';
import { body } from 'express-validator';
import { authenticate } from '../../core/middlewares/auth.middleware';
import { validate } from '../../core/middlewares/validation.middleware';
import { resizeImage, webOptimize, facePreprocess } from '../controllers/image.controller';

const imageRoutes = Router();

// Validation rules
const imageRequired = body('image')
  .notEmpty().withMessage('image (base64) is required')
  .isString().withMessage('image must be a string');

const resizeValidation = [
  imageRequired,
  body('width').optional().isInt({ min: 1 }).withMessage('width must be a positive integer'),
  body('height').optional().isInt({ min: 1 }).withMessage('height must be a positive integer'),
  body('maintainAspectRatio').optional().isBoolean().withMessage('maintainAspectRatio must be a boolean'),
];

const webOptimizeValidation = [
  imageRequired,
  body('quality').optional().isInt({ min: 1, max: 100 }).withMessage('quality must be an integer between 1 and 100'),
];

const facePreprocessValidation = [
  imageRequired,
  body('padding').optional().isFloat({ min: 0, max: 1 }).withMessage('padding must be a number between 0 and 1'),
  body('denoise').optional().isBoolean().withMessage('denoise must be a boolean'),
  body('contrastEnhance').optional().isBoolean().withMessage('contrastEnhance must be a boolean'),
  body('blurBackground').optional().isBoolean().withMessage('blurBackground must be a boolean'),
  body('grayscale').optional().isBoolean().withMessage('grayscale must be a boolean'),
  body('useAi').optional().isBoolean().withMessage('useAi must be a boolean'),
];

// Routes (all require authentication)
imageRoutes.post('/resize', authenticate, validate(resizeValidation), resizeImage);
imageRoutes.post('/web-optimize', authenticate, validate(webOptimizeValidation), webOptimize);
imageRoutes.post('/face-preprocess', authenticate, validate(facePreprocessValidation), facePreprocess);

export default imageRoutes;
