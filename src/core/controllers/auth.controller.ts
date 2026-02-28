import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import { ApiError } from '../middlewares/error.middleware';
import { HttpStatus, LoginRequest, RegisterRequest, ApiResponse, TokenResponse } from '../types';
import { generateToken, generateRefreshToken } from '../utils/jwt.utils';
import { logger } from '../utils/logger';
import { db } from '../utils/db';
import { config } from '../config';

/**
 * Register a new user
 */
export const register = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email, password, name } = req.body as RegisterRequest;
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    let user;
    
    if (config.database.type === 'mysql') {
      // MySQL implementation
      const usersTable = db.getTableName('users');
      const [result] = await db.execute(
        `INSERT INTO ${usersTable} (api_user, api_secret, full_name, email) VALUES (?, ?, ?, ?)`,
        [email, hashedPassword, name, email]
      );
      
      const insertResult = result as { insertId: number };
      
      // Get the created user
      const [users] = await db.execute(
        `SELECT id, api_user, full_name, email, created_at FROM ${usersTable} WHERE id = ?`,
        [insertResult.insertId]
      ) as [any[], any];
      
      user = users[0];
    } else {
      // Supabase implementation
      const userData = {
        api_user: email,
        api_secret: hashedPassword,
        full_name: name,
        email: email
      };
      
      user = await (db as any).insertUser(userData);
    }
    
    logger.info(`User registered: ${email}`);
    
    const response: ApiResponse<typeof user> = {
      success: true,
      data: user,
      message: 'User registered successfully'
    };
    
    res.status(HttpStatus.CREATED).json(response);
  } catch (error: any) {
    if (error.code === 'ER_DUP_ENTRY' || error.code === '23505') {
      next(new ApiError(HttpStatus.BAD_REQUEST, 'User already exists'));
    } else {
      next(error);
    }
  }
};

/**
 * Login user
 */
export const login = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email, password } = req.body as LoginRequest;
    
    let user;
    
    if (config.database.type === 'mysql') {
      // MySQL implementation
      const usersTable = db.getTableName('users');
      const [users] = await db.execute(
        `SELECT * FROM ${usersTable} WHERE api_user = ?`,
        [email]
      ) as [any[], any];
      
      user = users[0];
    } else {
      // Supabase implementation
      user = await (db as any).findUserByEmail(email);
    }
    
    if (!user) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Invalid credentials');
    }
    
    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.api_secret);
    if (!isPasswordValid) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Invalid credentials');
    }
    
    // Increment total_logins
    if (config.database.type === 'mysql') {
      const usersTable = db.getTableName('users');
      await db.execute(
        `UPDATE ${usersTable} SET total_logins = total_logins + 1 WHERE id = ?`,
        [user.id]
      );
    } else {
      await (db as any).updateUserLogins(user.id);
    }
    
    // Generate tokens
    const payload = { id: user.id.toString(), email: user.email, role: 'user' };
    const accessToken = generateToken(payload);
    const refreshToken = generateRefreshToken(payload);
    
    logger.info(`User logged in: ${email}`);
    
    const response: ApiResponse<TokenResponse> = {
      success: true,
      data: { accessToken, refreshToken },
      message: 'Login successful'
    };
    
    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Get current user profile
 */
export const getProfile = async (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!req.user) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Not authenticated');
    }
    
    let user;
    
    if (config.database.type === 'mysql') {
      // MySQL implementation
      const usersTable = db.getTableName('users');
      const [users] = await db.execute(
        `SELECT id, api_user, full_name, email, total_logins, created_at FROM ${usersTable} WHERE id = ?`,
        [req.user.id]
      ) as [any[], any];
      
      user = users[0];
    } else {
      // Supabase implementation
      user = await (db as any).findUserById(req.user.id);
    }
    
    if (!user) {
      throw new ApiError(HttpStatus.NOT_FOUND, 'User not found');
    }
    
    const response: ApiResponse<typeof user> = {
      success: true,
      data: user
    };
    
    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Refresh access token
 */
export const refreshToken = (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!req.user) {
      throw new ApiError(HttpStatus.UNAUTHORIZED, 'Not authenticated');
    }
    
    const payload = { 
      id: req.user.id, 
      email: req.user.email, 
      role: req.user.role 
    };
    
    const accessToken = generateToken(payload);
    
    const response: ApiResponse<{ accessToken: string }> = {
      success: true,
      data: { accessToken },
      message: 'Token refreshed successfully'
    };
    
    res.status(HttpStatus.OK).json(response);
  } catch (error) {
    next(error);
  }
};