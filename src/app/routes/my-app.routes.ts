import { Router } from 'express';
import { body, param, query } from 'express-validator';
import { validate } from '../../core/middlewares/validation.middleware';
import { helloWorld } from '../controllers/my-app.controller';

const myAppRoutes = Router();

// Validation rules
const helloWorldValidation = [
  body('firstName').notEmpty().withMessage('First name is required'),
  body('lastName').notEmpty().withMessage('Last name is required')
];

// Routes
myAppRoutes.get('/status', (req, res) => {
  res.json({ status: 'My App is running' });
});

myAppRoutes.post(
  '/hello-world',
  validate(helloWorldValidation),
  helloWorld
);




export default myAppRoutes;