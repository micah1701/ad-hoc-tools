import { Router } from 'express';
import myAppRoutes from './my-app.routes';
import imageRoutes from './image.routes';
import aiRoutes from './ai.routes';

const router = Router();

// Application-specific routes
router.use('/my-app', myAppRoutes);
router.use('/image', imageRoutes);
router.use('/ai', aiRoutes);

// Handle 404 for app-specific routes
router.use('*', (req, res, next) => {
  // Pass through to next middleware if no app routes match
  next();
});

export default router;