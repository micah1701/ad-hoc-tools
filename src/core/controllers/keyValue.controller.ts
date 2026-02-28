import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { ApiError } from '../middlewares/error.middleware';
import { HttpStatus, ApiResponse } from '../types';
import { db } from '../utils/db';
import { encryptWithMasterKey, decryptWithMasterKey } from '../utils/encryption';
import { logger } from '../utils/logger';
import { config } from '../config';

interface KeyValuePair {
  key: string;
  value: string;
}

/**
 * Store a new key-value pair
 */
export const storeKeyValue = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { key, value } = req.body as KeyValuePair;
    
    if (!key || !value) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'Key and value are required');
    }

    const uuid = uuidv4();
    const encryptedValue = encryptWithMasterKey(value);

    if (config.database.type === 'mysql') {
      const keyValuesTable = db.getTableName('key_values');
      await db.execute(
        `INSERT INTO ${keyValuesTable} (uuid, key_name, encrypted_value) VALUES (?, ?, ?)`,
        [uuid, key, encryptedValue]
      );
    } else {
      await (db as any).insertKeyValue(uuid, key, encryptedValue);
    }

    logger.info(`Stored new key-value pair with UUID: ${uuid}`);

    const response: ApiResponse<{ uuid: string }> = {
      success: true,
      data: { uuid },
      message: 'Key-value pair stored successfully'
    };

    res.status(HttpStatus.CREATED).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Update an existing key-value pair
 */
export const updateKeyValue = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { uuid } = req.params;
    const { value } = req.body as Partial<KeyValuePair>;

    if (!value) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'Value is required');
    }

    const encryptedValue = encryptWithMasterKey(value);

    if (config.database.type === 'mysql') {
      const keyValuesTable = db.getTableName('key_values');
      const [result] = await db.execute(
        `UPDATE ${keyValuesTable} SET encrypted_value = ? WHERE uuid = ?`,
        [encryptedValue, uuid]
      ) as [any, any];

      if (result.affectedRows === 0) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'Key-value pair not found');
      }
    } else {
      const result = await (db as any).updateKeyValue(uuid, encryptedValue);
      if (!result || result.length === 0) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'Key-value pair not found');
      }
    }

    logger.info(`Updated key-value pair with UUID: ${uuid}`);

    const response: ApiResponse<{ uuid: string }> = {
      success: true,
      data: { uuid },
      message: 'Key-value pair updated successfully'
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Retrieve a key-value pair
 */
export const getKeyValue = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { uuid } = req.params;

    let keyValueData;

    if (config.database.type === 'mysql') {
      const keyValuesTable = db.getTableName('key_values');
      const [rows] = await db.execute(
        `SELECT key_name, encrypted_value FROM ${keyValuesTable} WHERE uuid = ?`,
        [uuid]
      ) as [any[], any];

      if (rows.length === 0) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'Key-value pair not found');
      }

      keyValueData = rows[0];

      // Increment retrieved counter
      await db.execute(
        `UPDATE ${keyValuesTable} SET retrieved = retrieved + 1 WHERE uuid = ?`,
        [uuid]
      );
    } else {
      keyValueData = await (db as any).getKeyValue(uuid);
      
      if (!keyValueData) {
        throw new ApiError(HttpStatus.NOT_FOUND, 'Key-value pair not found');
      }

      // Increment retrieved counter
      await (db as any).incrementKeyValueRetrieved(uuid);
    }

    const { key_name, encrypted_value } = keyValueData;
    const decryptedValue = decryptWithMasterKey(encrypted_value);

    logger.info(`Retrieved key-value pair with UUID: ${uuid}`);

    const response: ApiResponse<{ key: string; value: string }> = {
      success: true,
      data: {
        key: key_name,
        value: decryptedValue
      }
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};