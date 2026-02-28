import { Router } from 'express';
import { param } from 'express-validator';
import * as sshKeysController from '../controllers/sshKeys.controller';
import { validate } from '../middlewares/validation.middleware';
import { authenticate } from '../middlewares/auth.middleware';

const router = Router();

// Validation rules
const generateSSHKeysValidation = [
  param('keyType')
    .isIn(['RSA2048', 'RSA4096', 'Ed25519', 'X25519'])
    .withMessage('Key type must be one of: RSA2048, RSA4096, Ed25519, X25519')
];

// Generate SSH key pair endpoint
router.post(
  '/generate/:keyType',
  authenticate,
  validate(generateSSHKeysValidation),
  sshKeysController.generateSSHKeys
);

export default router;