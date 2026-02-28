import { Router } from 'express';
import { body } from 'express-validator';
import { authController } from '../controllers';
import { validate } from '../middlewares/validation.middleware';
import { authenticate } from '../middlewares/auth.middleware';

const router = Router();

// Register validation rules
const registerValidation = [
  body('email').isEmail().withMessage('Must be a valid email'),
  body('password')
    .isLength({ min: 6 })
    .withMessage('Password must be at least 6 characters long'),
  body('name').notEmpty().withMessage('Name is required')
];

// Login validation rules
const loginValidation = [
  body('email').isEmail().withMessage('Must be a valid email'),
  body('password').notEmpty().withMessage('Password is required')
];

// Routes
router.post(
  '/register',
  validate(registerValidation),
  authController.register
);

router.post(
  '/login',
  validate(loginValidation),
  authController.login
);

router.get(
  '/profile',
  authenticate,
  authController.getProfile
);

router.post(
  '/refresh-token',
  authenticate,
  authController.refreshToken
);

export default router;