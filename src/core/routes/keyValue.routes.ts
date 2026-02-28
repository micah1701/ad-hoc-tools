import { Router } from 'express';
import { body } from 'express-validator';
import * as keyValueController from '../controllers/keyValue.controller';
import { validate } from '../middlewares/validation.middleware';
import { authenticate } from '../middlewares/auth.middleware';

const router = Router();

// Validation rules
const storeValidation = [
  body('key').notEmpty().withMessage('Key is required'),
  body('value').notEmpty().withMessage('Value is required')
];

const updateValidation = [
  body('value').notEmpty().withMessage('Value is required')
];

// Protected routes
router.post(
  '/',
  authenticate,
  validate(storeValidation),
  keyValueController.storeKeyValue
);

router.put(
  '/:uuid',
  authenticate,
  validate(updateValidation),
  keyValueController.updateKeyValue
);

router.get(
  '/:uuid',
  authenticate,
  keyValueController.getKeyValue
);

export default router;