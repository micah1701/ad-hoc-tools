# TypeScript API Framework

A modular Node.js and TypeScript framework for building RESTful APIs with JWT authentication and encrypted key-value storage.

## 🏗️ Framework Architecture

This framework is designed with a modular architecture that separates core functionality from application-specific features:

### Core Framework (`src/core/`)

- **Authentication & Authorization** - JWT-based user management
- **Database Abstraction** - Support for MySQL and PostgreSQL (Supabase)
- **Encryption** - AES-CBC and RSA encryption utilities with multiple encryption methods
- **Key-Value Storage** - Encrypted data storage system
- **SSH Key Generation** - Generate RSA and Ed25519 SSH key pairs
- **API Request Logging** - Comprehensive logging of all API requests and responses
- **Middleware** - Error handling, validation, authentication, request logging
- **Utilities** - Logging, JWT handling, database connections

### Application Layer (`src/app/`)

- **Controllers** - Application-specific business logic (Keychain Management)
- **Routes** - Custom API endpoints for your application
- **Middlewares** - App-specific middleware functions
- **Types** - Application-specific TypeScript interfaces

## Features

- 🔐 JWT-based authentication with refresh tokens
- 🛣️ Modular Express routing system
- 🔄 TypeScript for complete type safety
- 🛡️ Comprehensive request validation
- 🚦 Flexible middleware architecture
- 📝 Structured logging with Winston
- 🔒 Security best practices with Helmet
- 🔑 Multiple encryption methods (AES-CBC, RSA, passphrase-based)
- 🗄️ Multi-database support (MySQL/PostgreSQL)
- 📊 Dynamic table prefixing for multi-tenancy
- 🏗️ Extensible framework architecture
- 👥 User-based access control for applications
- 🔐 Secure keychain management system with flexible encryption options
- 🔄 Public key versioning and rotation support
- 🔑 SSH key pair generation (RSA2048, RSA4096, Ed25519)
- 📋 Comprehensive API request logging with encryption

## Getting Started

### Prerequisites

- Node.js (v16+)
- npm or yarn
- MySQL server OR Supabase account

### Installation

1. Clone or copy this framework
2. Install dependencies:

   ```bash
   npm install
   ```

3. Create environment configuration:

   ```bash
   cp .env.example .env
   ```

4. Configure your `.env` file:

   ```env
   # Database Configuration
   DATABASE_TYPE=postgres  # or 'mysql'
   TABLE_PREFIX=myapp_     # or leave empty for no prefix

   # For MySQL
   DB_HOST=localhost
   DB_USER=root
   DB_PASSWORD=your_password
   DB_NAME=your_database

   # For Supabase
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

   # Security
   JWT_SECRET=your_jwt_secret_here
   MASTER_ENCRYPTION_KEY=your_master_encryption_key_here
   ```

5. Set up your database:

   ```bash
   npm run db:setup
   ```

6. Start development server:

   ```bash
   npm run dev
   ```

7. Access the application:
   - **Landing Page**: http://localhost:3000
   - **Dashboard**: http://localhost:3000/dashboard (requires login)
   - **API Health**: http://localhost:3000/api/health

## Core API Endpoints

### Authentication

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile (protected)
- `POST /api/auth/refresh-token` - Refresh access token (protected)

### Key-Value Storage

- `POST /api/key-values` - Store encrypted key-value pair (protected)
- `GET /api/key-values/:uuid` - Retrieve key-value pair (protected)
- `PUT /api/key-values/:uuid` - Update key-value pair (protected)

### SSH Key Generation

- `POST /api/ssh-keys/generate/:keyType` - Generate SSH key pair (protected)
  - Supported key types: `RSA2048`, `RSA4096`, `Ed25519`

### Keychain Management (Application Layer)

- `POST /api/keychain/authenticate` - Authenticate keychain application
- `GET /api/keychain/apps` - Get user's keychain applications (protected)
- `POST /api/keychain/apps` - Create keychain application (protected)
- `GET /api/keychain/apps/:account_id` - Get specific application (protected)
- `PUT /api/keychain/apps/:account_id` - Update application (protected)
- `POST /api/keychain/apps/:account_id/public-keys` - Add public key (protected)
- `GET /api/keychain/apps/:account_id/public-keys` - Get public keys (protected)
- `POST /api/keychain/apps/:account_id/private-keys` - Store private key (protected)
- `POST /api/keychain/apps/:account_id/private-keys/:retrieval_id/retrieve` - Get private key (protected)
- `GET /api/keychain/apps/:account_id/private-keys` - List private keys (protected)

### System

- `GET /api/health` - Health check endpoint
- `GET /` - Landing page with login functionality
- `GET /dashboard` - Management dashboard (requires authentication)

## 🔑 SSH Key Generation

The framework includes a built-in SSH key generation service that supports multiple key types:

### Supported Key Types

- **RSA2048** - 2048-bit RSA keys (good balance of security and performance) - For signatures and encryption
- **RSA4096** - 4096-bit RSA keys (higher security, slower performance) - For signatures and encryption
- **Ed25519** - Modern elliptic curve keys for signatures (recommended for SSH authentication)
- **X25519** - Modern elliptic curve keys for encryption (fast and secure)

### Usage Example

```bash
# Generate RSA2048 key pair
curl -X POST http://localhost:3000/api/ssh-keys/generate/RSA2048 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json"

# Generate Ed25519 key pair
curl -X POST http://localhost:3000/api/ssh-keys/generate/Ed25519 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json"
```

### Response Format

```json
{
  "success": true,
  "data": {
    "publicKey": "ssh-rsa AAAAB3NzaC1yc2EAAAA...",
    "privateKey": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
    "keyType": "RSA2048",
    "fingerprint": "SHA256:..."
  },
  "message": "SSH RSA2048 key pair generated successfully"
}
```

## 📋 API Request Logging

The framework automatically logs all API requests and responses for audit and monitoring purposes.

### Features

- **Comprehensive Logging** - Captures method, URL, headers, request/response bodies
- **Sensitive Data Redaction** - Automatically redacts passwords, tokens, private keys
- **User Tracking** - Extracts user ID from JWT tokens when available
- **Encrypted Storage** - All logged data is encrypted using the master encryption key
- **Performance Metrics** - Tracks response times for performance monitoring
- **Error Tracking** - Logs errors and exceptions with full context

### Logged Information

- Request UUID (unique identifier for each request)
- User ID (extracted from JWT token if present)
- HTTP method and URL
- Request/response headers (sensitive data redacted)
- Request/response bodies (sensitive data redacted)
- Client IP address and User-Agent
- Response time in milliseconds
- HTTP status code
- Error messages (if any)

### Sensitive Data Redaction

The following fields are automatically redacted from logs:

- `password`, `passphrase`, `account_secret`
- `private_key`, `api_secret`, `secret`
- `token`, `authorization`, `cookie`
- `x-api-key` and other sensitive headers

### Database Schema

The `api_request_logs` table stores:

- `request_uuid` - Unique identifier for the request
- `user_id` - User ID (if authenticated)
- `method` - HTTP method
- `url` - Request URL
- `status_code` - HTTP response status
- `encrypted_headers` - Encrypted request headers
- `encrypted_request_body` - Encrypted request body
- `encrypted_response_body` - Encrypted response body
- `ip_address` - Client IP address
- `user_agent` - Client User-Agent string
- `response_time_ms` - Response time in milliseconds
- `error_message` - Error message (if any)
- `created_at` - Timestamp

## 🔐 Encryption Methods

The framework supports three different encryption methods:

### 1. Default Encryption

- **Description**: Uses the master encryption key from environment variables
- **Use Case**: Convenient for most applications where server-side encryption is sufficient
- **Security**: Server can decrypt data (good for automated processes)
- **Setup**: No additional configuration required

### 2. Passphrase Encryption

- **Description**: Requires a user-provided passphrase for encryption/decryption
- **Use Case**: When users want control over their encryption keys
- **Security**: Server cannot decrypt without the passphrase (higher security)
- **Setup**: Passphrase must be provided with each store/retrieve operation

### 3. Public Key Encryption

- **Description**: Uses public key encryption (RSA or Ed25519) with the application's active public key
  - **Supported Keys**: RSA and X25519 (Ed25519 is a signature algorithm, not suitable for encryption)
- **Use Case**: End-to-end encryption where server never has access to decrypted data
  - **Security**: Highest security - server cannot decrypt data
  - **Setup**: Requires providing a public key (RSA PEM, SSH RSA, or X25519) during application creation or adding one later
  - **Formats Supported**: RSA PEM, SSH RSA (ssh-rsa), X25519 (x25519:base64key)

**Security Note:** For RSA public key encryption, the server returns encrypted data that must be decrypted client-side using the corresponding private key. This ensures true end-to-end encryption where the server never has access to the decrypted data.

**X25519 Support:** X25519 provides modern elliptic curve encryption with excellent performance and security. X25519 keys are smaller and faster than RSA while providing equivalent security.

**Important:** Ed25519 keys are not supported for encryption because Ed25519 is a digital signature algorithm, not an encryption algorithm.

## Extending the Framework

### Adding Application-Specific Features

1. **Create Controllers** in `src/app/controllers/`:

   ```typescript
   // src/app/controllers/product.controller.ts
   import { Request, Response, NextFunction } from "express";
   import { ApiResponse, HttpStatus } from "../../core/types";

   export const getProducts = async (
     req: Request,
     res: Response,
     next: NextFunction,
   ) => {
     // Your application logic here
     // Access authenticated user via req.user
   };
   ```

2. **Define Routes** in `src/app/routes/`:

   ```typescript
   // src/app/routes/product.routes.ts
   import { Router } from "express";
   import * as productController from "../controllers/product.controller";
   import { authenticate } from "../../core/middlewares/auth.middleware";

   const router = Router();
   router.get("/", authenticate, productController.getProducts);
   export default router;
   ```

3. **Register Routes** in `src/app/routes/index.ts`:

   ```typescript
   import { Router } from "express";
   import productRoutes from "./product.routes";

   const router = Router();
   router.use("/products", productRoutes);
   export default router;
   ```

### Using Core Framework Features

```typescript
// Import core utilities
import { db } from "../../core/utils/db";
import { logger } from "../../core/utils/logger";
import { authenticate } from "../../core/middlewares/auth.middleware";
import { ApiError } from "../../core/middlewares/error.middleware";
import { HttpStatus } from "../../core/types";

// Use database abstraction
const tableName = db.getTableName("my_table");
const result = await db.execute(`SELECT * FROM ${tableName}`);

// Access authenticated user
export const myController = async (
  req: Request,
  res: Response,
  next: NextFunction,
) => {
  const userId = req.user?.id; // Available after authenticate middleware
  // Your logic here
};

// Use encryption
import {
  encryptWithMasterKey,
  decryptWithMasterKey,
  encryptWithPassphrase,
  decryptWithPassphrase,
  encryptWithPublicKey,
  decryptWithPrivateKey,
  generateSSHKeyPair,
} from "../../core/utils/encryption";

const encrypted = encryptWithMasterKey("sensitive data");
const passphraseEncrypted = encryptWithPassphrase("data", "my-passphrase");
const sshKeys = generateSSHKeyPair("RSA2048");
```

## Project Structure

```
src/
├── core/                 # Core framework functionality
│   ├── config/          # Configuration management
│   ├── controllers/     # Core controllers (auth, key-value, ssh-keys)
│   ├── middlewares/     # Core middleware functions
│   ├── routes/          # Core API routes
│   ├── scripts/         # Database setup scripts
│   ├── types/           # Core TypeScript definitions
│   └── utils/           # Core utilities (db, encryption, jwt, logger)
├── app/                 # Application-specific code (Keychain Management)
│   ├── controllers/     # Keychain management controllers
│   ├── routes/          # Keychain API routes
│   ├── types/           # Keychain-specific types
│   └── README.md        # Keychain application documentation
├── routes/              # Route orchestration
├── index.ts             # Application entry point
public/                  # Static web interface
├── index.html          # Landing page
└── dashboard.html      # Management dashboard
```

## Database Support

### MySQL

- Full SQL support with connection pooling
- Automatic reconnection handling
- Transaction support
- Foreign key constraints

### PostgreSQL (Supabase)

- Supabase client integration
- Row Level Security (RLS) ready
- Real-time subscriptions available
- Custom database methods for complex operations

### Table Prefixing

Configure `TABLE_PREFIX` in your `.env` to support:

- Multi-tenancy
- Environment separation
- Database organization

### Database Schema

The framework creates the following tables:

- `users` - User authentication and profiles
- `key_values` - Encrypted key-value storage
- `api_request_logs` - Encrypted API request and response logs

## Security Features

- **Password Hashing** - bcrypt with configurable salt rounds
- **JWT Authentication** - Access and refresh token system
- **Request Validation** - express-validator integration
- **Multiple Encryption Methods** - AES-CBC, RSA, and passphrase-based encryption
- **Security Headers** - Helmet.js integration with CSP
- **CORS** - Configurable cross-origin resource sharing
- **User Access Control** - Role-based permissions for applications
- **Audit Logging** - Comprehensive logging of all operations
- **Sensitive Data Redaction** - Automatic redaction in logs
- **Encrypted Log Storage** - All logs encrypted with master key

## Development Scripts

```bash
npm run dev          # Start development server with hot reload
npm run build        # Build TypeScript to JavaScript
npm run start        # Start production server
npm run db:setup     # Set up database tables
npm run lint         # Run ESLint
npm run test         # Run tests
```

## Environment Variables

See `.env.example` for all available configuration options including:

- Database configuration (MySQL/Supabase)
- JWT secrets and expiration times
- Encryption keys
- Logging levels
- Server configuration

### Security

- Client-side token management
- Automatic token refresh
- Secure form handling
- XSS protection

## Contributing

This framework is designed as a template and is meant to be extended.

Where possible, create your application-specific features within `src/app/`.

If you must edit or add core features:

1. Add them to `src/core/`
2. Maintain backward compatibility
3. Update documentation
4. Add appropriate tests
5. Follow the established patterns for database abstraction
