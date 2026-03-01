import { Router } from 'express';
import { param } from 'express-validator';
import * as apiLogsController from '../controllers/apiLogs.controller';
import { validate } from '../middlewares/validation.middleware';
import { authenticate } from '../middlewares/auth.middleware';

const router = Router();

// Validation rules
const getApiLogValidation = [
  param('uuid')
    .isUUID()
    .withMessage('UUID must be a valid UUID format')
];

// Get recent API logs for the authenticated user (protected)
router.get('/', authenticate, apiLogsController.getMyApiLogs);

// Get API log by UUID endpoint (protected)
router.get(
  '/:uuid',
  authenticate,
  validate(getApiLogValidation),
  apiLogsController.getApiLogByUuid
);

export default router;