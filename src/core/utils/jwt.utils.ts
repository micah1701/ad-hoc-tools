import jwt, { SignOptions, JwtPayload, Secret } from 'jsonwebtoken';
import { config } from '../config';
import { UserPayload } from '../types';

/**
 * Generate a JWT token
 */
export const generateToken = (payload: UserPayload): string => {
  const secret: Secret = config.jwt.secret;
  
  const options: SignOptions = {
    algorithm: 'HS256',
    expiresIn: config.jwt.expiresIn
  };
  
  return jwt.sign(payload, secret, options);
};

/**
 * Generate a refresh token
 */
export const generateRefreshToken = (payload: UserPayload): string => {
  const secret: Secret = config.jwt.secret;

  const options: SignOptions = {
    algorithm: 'HS256',
    expiresIn: config.jwt.refreshExpiresIn
  };

  return jwt.sign(payload, secret, options);
};

/**
 * Verify a JWT token
 */
export const verifyToken = (token: string): UserPayload => {
  const secret: Secret = config.jwt.secret;

  const decoded = jwt.verify(token, secret) as JwtPayload & UserPayload;
  return {
    id: decoded.id,
    email: decoded.email,
    role: decoded.role
  };
};