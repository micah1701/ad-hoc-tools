import dotenv from 'dotenv';
import { SignOptions } from 'jsonwebtoken';

// Load environment variables from .env file
dotenv.config();

if (!process.env.JWT_SECRET) {
  throw new Error('JWT_SECRET environment variable must be defined');
}

if (!process.env.MASTER_ENCRYPTION_KEY) {
  throw new Error('MASTER_ENCRYPTION_KEY environment variable must be defined');
}

type JwtExpiration = SignOptions['expiresIn'];

export const config = {
  port: process.env.PORT || 3000,
  environment: process.env.NODE_ENV || 'development',
  endpointPrefix: process.env.ENDPOINT_PREFIX || '/api',
  
  database: {
    type: process.env.DATABASE_TYPE || 'postgres',
    tablePrefix: process.env.TABLE_PREFIX || '',
    
    // MySQL configuration
    mysql: {
      host: process.env.DB_HOST || 'localhost',
      user: process.env.DB_USER || 'root',
      password: process.env.DB_PASSWORD || '',
      database: process.env.DB_NAME || 'api_db',
      port: parseInt(process.env.DB_PORT || '3306')
    },
    
    // Supabase configuration
    supabase: {
      url: process.env.SUPABASE_URL || '',
      publishableKey: process.env.SUPABASE_PUBLISHABLE_KEY || '',
      schema: process.env.SUPABASE_SCHEMA || 'public',
      secretKey: process.env.SUPABASE_SECRET_KEY || ''
    }
  },
  
  jwt: {
    secret: process.env.JWT_SECRET,
    expiresIn: (process.env.JWT_EXPIRATION || '1h') as JwtExpiration,
    refreshExpiresIn: (process.env.JWT_REFRESH_EXPIRATION || '7d') as JwtExpiration
  },
  
  encryption: {
    masterKey: process.env.MASTER_ENCRYPTION_KEY
  },
  
  logging: {
    level: process.env.LOG_LEVEL || 'info'
  },

  ai: {
    baseUrl: process.env.AI_BASE_URL || 'https://api.venice.ai/api/v1',
    apiKey: process.env.AI_API_KEY || '',
    defaultModel: process.env.AI_DEFAULT_MODEL || 'llama-3.3-70b',

    // Image editing uses a separate client because providers like Venice.ai
    // support chat completions but not the /images/edits (inpainting) endpoint.
    // Point these at OpenAI (or any provider that implements images/edits).
    imageBaseUrl: process.env.AI_IMAGE_BASE_URL || 'https://api.openai.com/v1',
    imageApiKey: process.env.AI_IMAGE_API_KEY || process.env.AI_API_KEY || '',
    imageModel: process.env.AI_IMAGE_MODEL || 'dall-e-2',
  }
};