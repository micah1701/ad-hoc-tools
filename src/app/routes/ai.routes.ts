import { Router } from 'express';
import { body } from 'express-validator';
import { authenticate } from '../../core/middlewares/auth.middleware';
import { validate } from '../../core/middlewares/validation.middleware';
import { chat, ocrId } from '../controllers/ai.controller';

const aiRoutes = Router();

const chatValidation = [
  body('message')
    .notEmpty().withMessage('message is required')
    .isString().withMessage('message must be a string'),
  body('model').optional().isString().withMessage('model must be a string'),
  body('systemPrompt').optional().isString().withMessage('systemPrompt must be a string'),
];

const ocrIdValidation = [
  body('image')
    .notEmpty().withMessage('image (base64) is required')
    .isString().withMessage('image must be a string'),
  body('mediaType').optional().isString().withMessage('mediaType must be a string'),
  body('model').optional().isString().withMessage('model must be a string'),
];

// Routes (all require authentication)
aiRoutes.post('/chat', authenticate, validate(chatValidation), chat);
aiRoutes.post('/ocr-id', authenticate, validate(ocrIdValidation), ocrId);

export default aiRoutes;
