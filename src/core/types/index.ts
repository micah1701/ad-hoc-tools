export interface UserPayload {
  id: string;
  email: string;
  role: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  requestId?: string;
}

export enum HttpStatus {
  OK = 200,
  CREATED = 201,
  BAD_REQUEST = 400,
  UNAUTHORIZED = 401,
  FORBIDDEN = 403,
  NOT_FOUND = 404,
  NOT_IMPLEMENTED = 501,
  INTERNAL_SERVER_ERROR = 500
}

export interface TokenResponse {
  accessToken: string;
  refreshToken: string;
}

export interface User {
  id: string;
  email: string;
  password: string;
  name: string;
  role: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface SSHKeyPair {
  publicKey: string;
  privateKey: string;
  keyType: 'RSA2048' | 'RSA4096' | 'Ed25519';
  fingerprint: string;
}

export interface ApiRequestLog {
  id: number;
  requestUuid: string;
  userId: string | null;
  method: string;
  url: string;
  statusCode: number | null;
  encryptedHeaders: string;
  encryptedRequestBody: string | null;
  encryptedResponseBody: string | null;
  ipAddress: string;
  userAgent: string | null;
  responseTimeMs: number | null;
  errorMessage: string | null;
  createdAt: Date;
}