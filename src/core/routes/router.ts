import { Router, Request, Response, NextFunction, IRouter } from 'express';
import { logger } from '../utils/logger';

/**
 * A class to simplify route handling with support for all HTTP methods
 */
export class ApiRouter {
  private router: IRouter;
  private basePath: string;
  
  constructor(basePath: string = '') {
    this.router = Router();
    this.basePath = basePath;
    
    // Log all requests
    this.router.use((req: Request, res: Response, next: NextFunction) => {
      logger.info(`${req.method} ${req.originalUrl}`);
      next();
    });
  }
  
  /**
   * Register a GET route
   */
  public get(path: string, ...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.get(path, handlers);
    return this;
  }
  
  /**
   * Register a POST route
   */
  public post(path: string, ...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.post(path, handlers);
    return this;
  }
  
  /**
   * Register a PUT route
   */
  public put(path: string, ...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.put(path, handlers);
    return this;
  }
  
  /**
   * Register a DELETE route
   */
  public delete(path: string, ...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.delete(path, handlers);
    return this;
  }
  
  /**
   * Register an OPTIONS route
   */
  public options(path: string, ...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.options(path, handlers);
    return this;
  }
  
  /**
   * Register a middleware for all routes
   */
  public use(...handlers: Array<(req: Request, res: Response, next: NextFunction) => void>): ApiRouter {
    this.router.use(handlers);
    return this;
  }
  
  /**
   * Get the router instance
   */
  public getRouter(): IRouter {
    return this.router;
  }
}