import { Request, Response, NextFunction } from 'express';
import { HttpStatus, ApiResponse } from '../../core/types';
import { ApiError } from '../../core/middlewares/error.middleware';

/**
 * Sample Endpoint: Hello World
 */
export const helloWorld = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { firstName, lastName } = req.body;

    if (!firstName || !lastName) {
      throw new ApiError(HttpStatus.BAD_REQUEST, 'firstName and lastName are required');
    }

    const response: ApiResponse<{ someField: string }> = {
        success: true,
        data: {
          someField: 'someValue'
        },
        message: `App created for ${firstName} ${lastName}`
      };

      res.status(HttpStatus.OK).json(response);

      } catch (error) {
    next(error);
  }
};