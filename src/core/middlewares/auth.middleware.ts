import { Request, Response, NextFunction } from 'express';
import { ApiError } from './error.middleware';
import { HttpStatus, UserPayload } from '../types';
import { verifyToken } from '../utils/jwt.utils';

// Extend Express Request type to include user
declare global {
  namespace Express {
    interface Request {
      user?: UserPayload;
    }
  }
}

/**
 * Authentication middleware to protect routes
 */
export const authenticate = (req: Request, res: Response, next: NextFunction) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Authentication required');
    }
    
    const token = authHeader.split(' ')[1];
    
    if (!token) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Authentication token missing');
    }
    
    // Verify token
    const decoded = verifyToken(token);
    
    // Attach user to request
    req.user = decoded;
    
    next();
  } catch (error) {
    if (error instanceof ApiError) {
      next(error);
    } else {
      next(new ApiError(HttpStatus.UNAUTHORIZED, 'Invalid authentication token'));
    }
  }
};

/**
 * Authorization middleware to check user roles
 */
export const authorize = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return next(new ApiError(HttpStatus.UNAUTHORIZED, 'Authentication required'));
    }
    
    if (!roles.includes(req.user.role)) {
      return next(new ApiError(HttpStatus.FORBIDDEN, 'Insufficient permissions'));
    }
    
    next();
  };
};