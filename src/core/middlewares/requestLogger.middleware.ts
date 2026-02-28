import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { verifyToken } from '../utils/jwt.utils';
import { encryptWithMasterKey } from '../utils/encryption';
import { logger } from '../utils/logger';
import { db } from '../utils/db';
import { config } from '../config';

// Extend Express Request type to include logging properties
declare global {
  namespace Express {
    interface Request {
      requestUuid?: string;
      startTime?: number;
      loggedUserId?: string | null;
    }
  }
}

/**
 * Sensitive fields that should be redacted from logs
 */
const SENSITIVE_FIELDS = [
  'password',
  'api_password',
  'passphrase',
  'account_secret',
  'private_key',
  'privatekey',
  'apisecret',
  'api_secret',
  'secret',
  'token',
  'authorization',
  'cookie',
  'x-api-key'
];

/**
 * Redacts sensitive information from an object
 */
const redactSensitiveData = (obj: any): any => {
  if (!obj || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(redactSensitiveData);
  }

  const redacted: any = {};
  
  for (const [key, value] of Object.entries(obj)) {
    const lowerKey = key.toLowerCase();
    
    // Check if this field should be redacted
    const shouldRedact = SENSITIVE_FIELDS.some(sensitiveField => 
      lowerKey.includes(sensitiveField)
    );
    
    if (shouldRedact) {
      redacted[key] = '[REDACTED]';
    } else if (typeof value === 'object' && value !== null) {
      // Recursively redact nested objects and arrays
      redacted[key] = redactSensitiveData(value);
    } else {
      redacted[key] = value;
    }
  }
  
  return redacted;
};

/**
 * Extracts user ID from JWT token without throwing errors
 */
const extractUserIdFromToken = (authHeader: string | undefined): string | null => {
  try {
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return null;
    }
    
    const token = authHeader.split(' ')[1];
    if (!token) {
      return null;
    }
    
    const decoded = verifyToken(token);
    return decoded.id || null;
  } catch (error) {
    // Token is invalid or expired, but we don't want to throw an error here
    return null;
  }
};

/**
 * Redacts JWT token from headers while preserving user ID extraction
 */
const redactHeaders = (headers: any): any => {
  const redactedHeaders = { ...headers };
  
  // Redact authorization header but keep a placeholder
  if (redactedHeaders.authorization) {
    redactedHeaders.authorization = '[REDACTED_JWT_TOKEN]';
  }
  
  // Redact other sensitive headers
  Object.keys(redactedHeaders).forEach(key => {
    const lowerKey = key.toLowerCase();
    if (SENSITIVE_FIELDS.some(field => lowerKey.includes(field))) {
      redactedHeaders[key] = '[REDACTED]';
    }
  });
  
  return redactedHeaders;
};

/**
 * Stores API request log in database
 */
const storeRequestLog = async (logData: {
  requestUuid: string;
  userId: string | null;
  method: string;
  url: string;
  statusCode: number | null;
  headers: any;
  requestBody: any;
  responseBody: any;
  ipAddress: string;
  userAgent: string | null;
  responseTimeMs: number | null;
  errorMessage: string | null;
}) => {
  try {
    // Encrypt sensitive data
    const encryptedHeaders = encryptWithMasterKey(JSON.stringify(logData.headers));
    const encryptedRequestBody = logData.requestBody 
      ? encryptWithMasterKey(JSON.stringify(logData.requestBody))
      : null;
    const encryptedResponseBody = logData.responseBody 
      ? encryptWithMasterKey(JSON.stringify(logData.responseBody))
      : null;

    if (config.database.type === 'mysql') {
      const tableName = db.getTableName('api_request_logs');
      await db.execute(
        `INSERT INTO ${tableName} (
          request_uuid, user_id, method, url, status_code, 
          encrypted_headers, encrypted_request_body, encrypted_response_body,
          ip_address, user_agent, response_time_ms, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          logData.requestUuid,
          logData.userId,
          logData.method,
          logData.url,
          logData.statusCode,
          encryptedHeaders,
          encryptedRequestBody,
          encryptedResponseBody,
          logData.ipAddress,
          logData.userAgent,
          logData.responseTimeMs,
          logData.errorMessage
        ]
      );
    } else {
      // Supabase implementation
      await (db as any).insertApiRequestLog({
        request_uuid: logData.requestUuid,
        user_id: logData.userId,
        method: logData.method,
        url: logData.url,
        status_code: logData.statusCode,
        encrypted_headers: encryptedHeaders,
        encrypted_request_body: encryptedRequestBody,
        encrypted_response_body: encryptedResponseBody,
        ip_address: logData.ipAddress,
        user_agent: logData.userAgent,
        response_time_ms: logData.responseTimeMs,
        error_message: logData.errorMessage
      });
    }
  } catch (error) {
    // Log the error but don't fail the request
    logger.error('Failed to store API request log:', error);
  }
};

/**
 * Adds requestId to JSON response if it's an object
 */
const addRequestIdToResponse = (body: any, requestUuid: string): any => {
  // Only add requestId to object responses (not strings, numbers, etc.)
  if (body && typeof body === 'object' && !Array.isArray(body)) {
    return {
      ...body,
      requestId: requestUuid
    };
  }
  return body;
};

/**
 * Request logging middleware - captures request data
 */
export const requestLogger = (req: Request, res: Response, next: NextFunction) => {
  // Generate unique request UUID
  req.requestUuid = uuidv4();
  req.startTime = Date.now();
  
  // Extract user ID from JWT token (if present)
  req.loggedUserId = extractUserIdFromToken(req.headers.authorization);
  
  // Get client IP address
  const ipAddress = req.ip || 
    req.connection.remoteAddress || 
    req.socket.remoteAddress || 
    (req.connection as any)?.socket?.remoteAddress ||
    'unknown';

  // Prepare request data for logging
  const requestData = {
    requestUuid: req.requestUuid,
    userId: req.loggedUserId,
    method: req.method,
    url: req.originalUrl || req.url,
    headers: redactHeaders(req.headers),
    requestBody: redactSensitiveData(req.body),
    ipAddress,
    userAgent: req.headers['user-agent'] || null,
    timestamp: new Date().toISOString()
  };

  // Log request start
  logger.info(`API Request Started: ${req.method} ${req.originalUrl}`, {
    requestUuid: req.requestUuid,
    userId: req.loggedUserId,
    ipAddress
  });

  // Store original res.json and res.send methods
  const originalJson = res.json;
  const originalSend = res.send;
  const originalEnd = res.end;

  let responseBody: any = null;
  let responseCaptured = false;

  // Override res.json to capture response and add requestId
  res.json = function(body: any) {
    if (!responseCaptured) {
      // Add requestId to the response body before logging
      const bodyWithRequestId = addRequestIdToResponse(body, req.requestUuid!);
      responseBody = redactSensitiveData(bodyWithRequestId);
      responseCaptured = true;
      
      // Send the response with requestId
      return originalJson.call(this, bodyWithRequestId);
    }
    return originalJson.call(this, body);
  };

  // Override res.send to capture response and add requestId
  res.send = function(body: any) {
    if (!responseCaptured) {
      try {
        // Try to parse as JSON, add requestId if it's an object, otherwise store as string
        let parsedBody = typeof body === 'string' ? JSON.parse(body) : body;
        const bodyWithRequestId = addRequestIdToResponse(parsedBody, req.requestUuid!);
        responseBody = redactSensitiveData(bodyWithRequestId);
        
        // Send the response with requestId if it was modified
        if (bodyWithRequestId !== parsedBody) {
          return originalSend.call(this, JSON.stringify(bodyWithRequestId));
        }
      } catch {
        responseBody = typeof body === 'string' ? body : String(body);
      }
      responseCaptured = true;
    }
    return originalSend.call(this, body);
  };

  // Override res.end to capture any remaining responses
  res.end = function(chunk?: any, encoding?: any) {
    if (!responseCaptured && chunk) {
      try {
        let parsedChunk = typeof chunk === 'string' ? JSON.parse(chunk) : chunk;
        const chunkWithRequestId = addRequestIdToResponse(parsedChunk, req.requestUuid!);
        responseBody = redactSensitiveData(chunkWithRequestId);
        
        // Send the response with requestId if it was modified
        if (chunkWithRequestId !== parsedChunk) {
          return originalEnd.call(this, JSON.stringify(chunkWithRequestId), encoding);
        }
      } catch {
        responseBody = typeof chunk === 'string' ? chunk : String(chunk);
      }
      responseCaptured = true;
    }
    return originalEnd.call(this, chunk, encoding);
  };

  // Handle response completion
  res.on('finish', async () => {
    const responseTime = req.startTime ? Date.now() - req.startTime : null;
    
    // Prepare complete log data
    const logData = {
      requestUuid: req.requestUuid!,
      userId: req.loggedUserId ?? null,
      method: req.method,
      url: req.originalUrl || req.url,
      statusCode: res.statusCode,
      headers: requestData.headers,
      requestBody: requestData.requestBody,
      responseBody,
      ipAddress: requestData.ipAddress,
      userAgent: requestData.userAgent,
      responseTimeMs: responseTime,
      errorMessage: null
    };

    // Log request completion
    logger.info(`API Request Completed: ${req.method} ${req.originalUrl}`, {
      requestUuid: req.requestUuid,
      userId: req.loggedUserId,
      statusCode: res.statusCode,
      responseTime: `${responseTime}ms`
    });

    // Store in database
    await storeRequestLog(logData);
  });

  // Handle errors
  res.on('error', async (error: Error) => {
    const responseTime = req.startTime ? Date.now() - req.startTime : null;
    
    const logData = {
      requestUuid: req.requestUuid!,
      userId: req.loggedUserId ?? null,
      method: req.method,
      url: req.originalUrl || req.url,
      statusCode: res.statusCode || 500,
      headers: requestData.headers,
      requestBody: requestData.requestBody,
      responseBody,
      ipAddress: requestData.ipAddress,
      userAgent: requestData.userAgent,
      responseTimeMs: responseTime,
      errorMessage: error.message
    };

    // Log error
    logger.error(`API Request Error: ${req.method} ${req.originalUrl}`, {
      requestUuid: req.requestUuid,
      userId: req.loggedUserId,
      error: error.message
    });

    // Store in database
    await storeRequestLog(logData);
  });

  next();
};