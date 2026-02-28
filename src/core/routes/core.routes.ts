import { Router } from 'express';
import authRoutes from './auth.routes';
import keyValueRoutes from './keyValue.routes';
import sshKeysRoutes from './sshKeys.routes';
import apiLogsRoutes from './apiLogs.routes';

const router = Router();

// Health check endpoint
router.get('/health', (req, res) => {
  const response = { 
    status: 'ok', 
    timestamp: new Date(),
    requestId: (req as any).requestUuid || 'unknown'
  };
  res.json(response);
});

// Core framework routes
router.use('/auth', authRoutes);
router.use('/key-values', keyValueRoutes);
router.use('/ssh-keys', sshKeysRoutes);
router.use('/api-logs', apiLogsRoutes);

export default router;