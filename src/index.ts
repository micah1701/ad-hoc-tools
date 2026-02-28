import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import path from 'path';
import { config } from './core/config';
import { errorHandler } from './core/middlewares/error.middleware';
import { requestLogger } from './core/middlewares/requestLogger.middleware';
import routes from './routes';
import { logger } from './core/utils/logger';
import { testDatabaseConnection, isDatabaseConnected } from './core/utils/db';

// Initialize express app
const app = express();

// Apply middlewares
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      ...helmet.contentSecurityPolicy.getDefaultDirectives(),
      "script-src": ["'self'", "'unsafe-eval'", "'unsafe-inline'"],
      "script-src-attr": ["'unsafe-inline'"],
      "style-src": ["'self'", "'unsafe-inline'"],
    },
  },
})); // Security headers with CSP configuration
app.use(cors()); // Enable CORS
app.use(express.json()); // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies
app.use(morgan('dev')); // Request logging

// Add request logging middleware for API requests
app.use(config.endpointPrefix, requestLogger);

// Serve static files from public directory
app.use(express.static(path.join(__dirname, '../public')));

// Dashboard route
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/dashboard.html'));
});

// Landing page route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Health check endpoint (should work without database)
app.get('/api/health', (req, res) => {
  const dbStatus = isDatabaseConnected();
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    database: dbStatus ? 'connected' : 'disconnected',
    environment: config.environment
  });
});

// Apply routes
app.use(config.endpointPrefix, routes);

// Error handling middleware
app.use(errorHandler);

// Start server
const server = app.listen(config.port, async () => {
  logger.info(`Server running on port ${config.port} in ${config.environment} mode`);
  
  // Test database connection after server starts (non-blocking)
  await testDatabaseConnection();
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM signal received: closing HTTP server');
  server.close(() => {
    logger.info('HTTP server closed');
  });
});

export default app;