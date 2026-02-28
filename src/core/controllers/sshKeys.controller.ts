import { Request, Response, NextFunction } from 'express';
import { ApiError } from '../middlewares/error.middleware';
import { HttpStatus, ApiResponse } from '../types';
import { generateSSHKeyPair, SSHKeyType, SSHKeyPair } from '../utils/encryption';
import { logger } from '../utils/logger';

/**
 * Generate SSH key pair
 */
export const generateSSHKeys = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { keyType } = req.params;
    
    // Validate key type
    const validKeyTypes: SSHKeyType[] = ['RSA2048', 'RSA4096', 'Ed25519', 'X25519'];
    if (!validKeyTypes.includes(keyType as SSHKeyType)) {
      throw new ApiError(
        HttpStatus.BAD_REQUEST, 
        `Invalid key type. Supported types: ${validKeyTypes.join(', ')}`
      );
    }

    if (!req.user) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'User not authenticated');
    }

    logger.info(`Generating SSH key pair of type ${keyType} for user: ${req.user.id}`);

    // Generate the key pair
    const keyPair: SSHKeyPair = generateSSHKeyPair(keyType as SSHKeyType);

    logger.info(`SSH key pair generated successfully for user: ${req.user.id}, type: ${keyType}`);

    const response: ApiResponse<SSHKeyPair> = {
      success: true,
      data: keyPair,
      message: `SSH ${keyType} key pair generated successfully`
    };

    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    logger.error(`SSH key generation failed for type ${req.params.keyType}:`, error);
    next(error);
  }
};