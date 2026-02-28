import { Request, Response, NextFunction } from 'express';
import { ApiError } from '../middlewares/error.middleware';
import { HttpStatus, ApiResponse } from '../types';
import { decryptWithMasterKey } from '../utils/encryption';
import { logger } from '../utils/logger';
import { db } from '../utils/db';
import { config } from '../config';

/**
 * Get API request log by UUID (only if user ID matches)
 */
export const getApiLogByUuid = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { uuid } = req.params;

    if (!uuid) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'UUID is required');
    }

    if (!req.user) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'User not authenticated');
    }

    let logData;

    if (config.database.type === 'mysql') {
      const tableName = db.getTableName('api_request_logs');
      const [logs] = await db.execute(
        `SELECT * FROM ${tableName} WHERE request_uuid = ?`,
        [uuid]
      ) as [any[], any];

      if (logs.length === 0) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'API log not found');
      }

      logData = logs[0];
    } else {
      // Supabase implementation
      logData = await (db as any).findApiLogByUuid(uuid);

      if (!logData) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'API log not found');
      }
    }

    // Check if the authenticated user's ID matches the log's user_id
    if (logData.user_id !== parseInt(req.user.id)) {
      throw new ApiError(HttpStatus.FORBIDDEN, 'Access denied: You can only view your own API logs');
    }

    // Decrypt the encrypted fields
    let decryptedHeaders;
    let decryptedRequestBody = null;
    let decryptedResponseBody = null;

    try {
      decryptedHeaders = JSON.parse(decryptWithMasterKey(logData.encrypted_headers));
    } catch (error) {
      logger.error('Failed to decrypt headers for log:', uuid, error);
      decryptedHeaders = { error: 'Failed to decrypt headers' };
    }

    if (logData.encrypted_request_body) {
      try {
        decryptedRequestBody = JSON.parse(decryptWithMasterKey(logData.encrypted_request_body));
      } catch (error) {
        logger.error('Failed to decrypt request body for log:', uuid, error);
        decryptedRequestBody = { error: 'Failed to decrypt request body' };
      }
    }

    if (logData.encrypted_response_body) {
      try {
        decryptedResponseBody = JSON.parse(decryptWithMasterKey(logData.encrypted_response_body));
      } catch (error) {
        logger.error('Failed to decrypt response body for log:', uuid, error);
        decryptedResponseBody = { error: 'Failed to decrypt response body' };
      }
    }

    // Prepare the response data
    const responseData = {
      id: logData.id,
      requestUuid: logData.request_uuid,
      userId: logData.user_id,
      method: logData.method,
      url: logData.url,
      statusCode: logData.status_code,
      headers: decryptedHeaders,
      requestBody: decryptedRequestBody,
      responseBody: decryptedResponseBody,
      ipAddress: logData.ip_address,
      userAgent: logData.user_agent,
      responseTimeMs: logData.response_time_ms,
      errorMessage: logData.error_message,
      createdAt: logData.created_at
    };

    logger.info(`API log retrieved: ${uuid} by user: ${req.user.id}`);

    const response: ApiResponse<typeof responseData> = {
      success: true,
      data: responseData,
      message: 'API log retrieved successfully'
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};