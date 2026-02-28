import { Router } from 'express';
import coreRoutes from '../core/routes/core.routes';
import appRoutes from '../app/routes';

const router = Router();

// Mount core framework routes first
router.use('/', coreRoutes);

// Mount application-specific routes
router.use('/', appRoutes);

// Handle 404 - API endpoint not found (catch-all)
router.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'API endpoint not found'
  });
});

export default router;