import * as crypto from 'node:crypto';
import * as nacl from 'tweetnacl';
import * as naclUtil from 'tweetnacl-util';
import { config } from '../config';

/**
 * Generates a random IV for encryption (16 bytes)
 */
const generateIV = (): Buffer => {
  return crypto.randomBytes(16);
};

/**
 * Derives a 32-byte AES key from a password and salt using PBKDF2
 */
const deriveKey = (password: string, salt: Buffer): Buffer => {
  return crypto.pbkdf2Sync(password, salt, 10000, 32, 'sha1');
};

/**
 * Encrypts data using the master encryption key
 */
export const encryptWithMasterKey = (data: string): string => {
  const iv = generateIV();
  const key = deriveKey(config.encryption.masterKey, iv);

  const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
  const encryptedBuffer = Buffer.concat([
    cipher.update(data, 'utf8'),
    cipher.final()
  ]);

  // Combine IV + encrypted data and encode as base64
  const combined = Buffer.concat([iv, encryptedBuffer]);
  return combined.toString('base64');
};

/**
 * Decrypts data using the master encryption key
 */
export const decryptWithMasterKey = (encryptedData: string): string => {
  const combined = Buffer.from(encryptedData, 'base64');
  const iv = combined.subarray(0, 16);
  const encrypted = combined.subarray(16);

  const key = deriveKey(config.encryption.masterKey, iv);

  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  const decryptedBuffer = Buffer.concat([
    decipher.update(encrypted),
    decipher.final()
  ]);

  return decryptedBuffer.toString('utf8');
};

/**
 * Encrypts data using a passphrase
 */
export const encryptWithPassphrase = (data: string, passphrase: string): string => {
  const iv = generateIV();
  const key = deriveKey(passphrase, iv);

  const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
  const encryptedBuffer = Buffer.concat([
    cipher.update(data, 'utf8'),
    cipher.final()
  ]);

  const combined = Buffer.concat([iv, encryptedBuffer]);
  return combined.toString('base64');
};

/**
 * Decrypts data using a passphrase
 */
export const decryptWithPassphrase = (encryptedData: string, passphrase: string): string => {
  const combined = Buffer.from(encryptedData, 'base64');
  const iv = combined.subarray(0, 16);
  const encrypted = combined.subarray(16);

  const key = deriveKey(passphrase, iv);

  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  const decryptedBuffer = Buffer.concat([
    decipher.update(encrypted),
    decipher.final()
  ]);

  return decryptedBuffer.toString('utf8');
};

/**
 * Generates a new RSA-2048 key pair in PEM format
 */
export const generateKeyPair = (): { publicKey: string; privateKey: string } => {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
  });

  return { publicKey, privateKey };
};

/**
 * Detects the type of public key (RSA PEM, SSH RSA, SSH Ed25519, or X25519)
 */
const detectKeyType = (publicKey: string): 'rsa-pem' | 'ssh-rsa' | 'ssh-ed25519' | 'x25519' | 'unknown' => {
  const trimmedKey = publicKey.trim();

  if (trimmedKey.includes('-----BEGIN PUBLIC KEY-----') && trimmedKey.includes('-----END PUBLIC KEY-----')) {
    return 'rsa-pem';
  }
  if (trimmedKey.startsWith('ssh-rsa ')) {
    return 'ssh-rsa';
  }
  if (trimmedKey.startsWith('ssh-ed25519 ')) {
    return 'ssh-ed25519';
  }
  // X25519 keys are identified by our custom prefix
  if (trimmedKey.startsWith('x25519:')) {
    return 'x25519';
  }
  return 'unknown';
};

/**
 * Converts an SSH RSA public key to PEM format (SPKI)
 */
const convertSSHRSAToPEM = (sshKey: string): string => {
  const parts = sshKey.trim().split(' ');
  if (parts.length < 2) {
    throw new Error('Invalid SSH RSA key format');
  }

  const keyData = Buffer.from(parts[1], 'base64');
  let offset = 0;

  const readUint32 = (): number => {
    const val = keyData.readUInt32BE(offset);
    offset += 4;
    return val;
  };

  const readBytes = (length: number): Buffer => {
    const val = keyData.subarray(offset, offset + length);
    offset += length;
    return val;
  };

  // Read key type
  const keyTypeLength = readUint32();
  const keyType = readBytes(keyTypeLength).toString('utf8');
  if (keyType !== 'ssh-rsa') {
    throw new Error('Not an SSH RSA key');
  }

  // Read public exponent (e)
  const eLength = readUint32();
  const eBytes = readBytes(eLength);

  // Read modulus (n)
  const nLength = readUint32();
  const nBytes = readBytes(nLength);

  // Reconstruct via crypto.createPublicKey using JWK
  const n = nBytes[0] === 0 ? nBytes.subarray(1) : nBytes; // strip leading zero byte if present
  const e = eBytes;

  const jwk: crypto.JsonWebKey = {
    kty: 'RSA',
    n: n.toString('base64url'),
    e: e.toString('base64url')
  };

  const publicKey = crypto.createPublicKey({ key: jwk, format: 'jwk' });
  return publicKey.export({ type: 'spki', format: 'pem' }) as string;
};

/**
 * Encrypts data using X25519 public key (NaCl box)
 */
const encryptWithX25519 = (data: string, x25519PublicKey: string): string => {
  try {
    const publicKeyB64 = x25519PublicKey.replace('x25519:', '');
    const publicKeyBytes = naclUtil.decodeBase64(publicKeyB64);

    if (publicKeyBytes.length !== 32) {
      throw new Error('Invalid X25519 public key length. Expected 32 bytes.');
    }

    // Generate ephemeral key pair for this encryption
    const ephemeralKeyPair = nacl.box.keyPair();

    // Generate random nonce
    const nonce = nacl.randomBytes(24);

    // Encrypt the data
    const dataBytes = naclUtil.decodeUTF8(data);
    const encrypted = nacl.box(dataBytes, nonce, publicKeyBytes, ephemeralKeyPair.secretKey);

    if (!encrypted) {
      throw new Error('X25519 encryption failed');
    }

    // Combine ephemeral public key + nonce + encrypted data
    const combined = new Uint8Array(32 + 24 + encrypted.length);
    combined.set(ephemeralKeyPair.publicKey, 0);
    combined.set(nonce, 32);
    combined.set(encrypted, 56);

    return 'X25519:' + naclUtil.encodeBase64(combined);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`X25519 encryption failed: ${errorMessage}`);
  }
};

/**
 * Decrypts data that was encrypted with X25519
 */
const decryptWithX25519 = (encryptedData: string, x25519PrivateKey: string): string => {
  try {
    if (!encryptedData.startsWith('X25519:')) {
      throw new Error('Data was not encrypted with X25519');
    }

    const actualEncryptedData = encryptedData.substring(7); // Remove 'X25519:' prefix

    const privateKeyB64 = x25519PrivateKey.replace('x25519:', '');
    const privateKeyBytes = naclUtil.decodeBase64(privateKeyB64);

    if (privateKeyBytes.length !== 32) {
      throw new Error('Invalid X25519 private key length. Expected 32 bytes.');
    }

    const combined = naclUtil.decodeBase64(actualEncryptedData);

    if (combined.length < 56) {
      throw new Error('Invalid X25519 encrypted data format');
    }

    const ephemeralPublicKey = combined.slice(0, 32);
    const nonce = combined.slice(32, 56);
    const encrypted = combined.slice(56);

    const decrypted = nacl.box.open(encrypted, nonce, ephemeralPublicKey, privateKeyBytes);

    if (!decrypted) {
      throw new Error('X25519 decryption failed - invalid key or corrupted data');
    }

    return naclUtil.encodeUTF8(decrypted);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`X25519 decryption failed: ${errorMessage}`);
  }
};

/**
 * Encrypts data using a public key (supports RSA PEM, SSH RSA, and X25519)
 * Note: Ed25519 is not supported for encryption as it is a signature algorithm only.
 */
export const encryptWithPublicKey = (data: string, publicKey: string): string => {
  const keyType = detectKeyType(publicKey);

  switch (keyType) {
    case 'rsa-pem': {
      try {
        const encrypted = crypto.publicEncrypt(
          { key: publicKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
          Buffer.from(data, 'utf8')
        );
        return 'RSA:' + encrypted.toString('base64');
      } catch (error) {
        throw new Error(`RSA PEM encryption failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    case 'ssh-rsa': {
      try {
        const pemKey = convertSSHRSAToPEM(publicKey);
        const encrypted = crypto.publicEncrypt(
          { key: pemKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
          Buffer.from(data, 'utf8')
        );
        return 'RSA:' + encrypted.toString('base64');
      } catch (error) {
        throw new Error(`SSH RSA encryption failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    case 'x25519':
      return encryptWithX25519(data, publicKey);

    case 'ssh-ed25519':
      throw new Error(
        'Ed25519 keys are not supported for encryption. Ed25519 is a digital signature algorithm. ' +
        'For encryption, use RSA keys or X25519 for elliptic curve encryption.'
      );

    default:
      throw new Error(
        'Unsupported public key format. Supported formats for encryption: ' +
        'RSA PEM (-----BEGIN PUBLIC KEY-----), SSH RSA (ssh-rsa), X25519 (x25519:base64key)'
      );
  }
};

/**
 * Decrypts data using a private key (supports RSA PEM and X25519)
 */
export const decryptWithPrivateKey = (encryptedData: string, privateKey: string): string => {
  if (encryptedData.startsWith('RSA:')) {
    const actualEncryptedData = encryptedData.substring(4);
    try {
      const decrypted = crypto.privateDecrypt(
        { key: privateKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
        Buffer.from(actualEncryptedData, 'base64')
      );
      return decrypted.toString('utf8');
    } catch (error) {
      throw new Error(`RSA decryption failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else if (encryptedData.startsWith('X25519:')) {
    return decryptWithX25519(encryptedData, privateKey);
  } else if (encryptedData.startsWith('ED25519:')) {
    throw new Error(
      'Ed25519 encrypted data detected, but Ed25519 is a signature algorithm, not an encryption algorithm. ' +
      'This data may have been created incorrectly.'
    );
  } else {
    // Legacy format — assume RSA
    try {
      const decrypted = crypto.privateDecrypt(
        { key: privateKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
        Buffer.from(encryptedData, 'base64')
      );
      return decrypted.toString('utf8');
    } catch (error) {
      throw new Error(`Legacy RSA decryption failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
};

/**
 * Generates an X25519 key pair for encryption
 */
export const generateX25519KeyPair = (): { publicKey: string; privateKey: string } => {
  const keyPair = nacl.box.keyPair();

  return {
    publicKey: 'x25519:' + naclUtil.encodeBase64(keyPair.publicKey),
    privateKey: 'x25519:' + naclUtil.encodeBase64(keyPair.secretKey)
  };
};

/**
 * SSH Key Generation Types
 */
export type SSHKeyType = 'RSA2048' | 'RSA4096' | 'Ed25519' | 'X25519';

/**
 * SSH Key Pair Interface
 */
export interface SSHKeyPair {
  publicKey: string;
  privateKey: string;
  keyType: SSHKeyType;
  fingerprint: string;
  pemPublicKey?: string; // For RSA keys, also provide PEM format
}

/**
 * Converts an RSA public key (crypto.KeyObject or PEM) to SSH wire format
 */
const rsaPublicKeyToSSH = (publicKeyPem: string): string => {
  const keyObject = crypto.createPublicKey(publicKeyPem);
  const jwk = keyObject.export({ format: 'jwk' }) as { n: string; e: string };

  const keyType = 'ssh-rsa';
  const keyTypeBytes = Buffer.from(keyType, 'utf8');

  // Decode n and e from base64url
  let nBytes = Buffer.from(jwk.n, 'base64url');
  let eBytes = Buffer.from(jwk.e, 'base64url');

  // SSH RSA requires a leading 0x00 byte if the high bit is set (to indicate positive integer)
  if (nBytes[0] & 0x80) {
    nBytes = Buffer.concat([Buffer.from([0x00]), nBytes]);
  }
  if (eBytes[0] & 0x80) {
    eBytes = Buffer.concat([Buffer.from([0x00]), eBytes]);
  }

  const buf = Buffer.alloc(
    4 + keyTypeBytes.length +
    4 + eBytes.length +
    4 + nBytes.length
  );

  let pos = 0;
  buf.writeUInt32BE(keyTypeBytes.length, pos); pos += 4;
  keyTypeBytes.copy(buf, pos); pos += keyTypeBytes.length;

  buf.writeUInt32BE(eBytes.length, pos); pos += 4;
  eBytes.copy(buf, pos); pos += eBytes.length;

  buf.writeUInt32BE(nBytes.length, pos); pos += 4;
  nBytes.copy(buf, pos);

  return `ssh-rsa ${buf.toString('base64')}`;
};

/**
 * Generates an Ed25519 SSH key pair using native crypto
 */
const generateEd25519KeyPair = (): { publicKey: string; privateKey: string } => {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519', {
    publicKeyEncoding: { type: 'spki', format: 'der' },
    privateKeyEncoding: { type: 'pkcs8', format: 'der' }
  });

  // Ed25519 SPKI DER: last 32 bytes are the raw public key
  const rawPublicKey = publicKey.subarray(publicKey.length - 32);

  const keyType = 'ssh-ed25519';
  const keyTypeBytes = Buffer.from(keyType, 'utf8');

  const sshBuf = Buffer.alloc(4 + keyTypeBytes.length + 4 + rawPublicKey.length);
  let pos = 0;
  sshBuf.writeUInt32BE(keyTypeBytes.length, pos); pos += 4;
  keyTypeBytes.copy(sshBuf, pos); pos += keyTypeBytes.length;
  sshBuf.writeUInt32BE(rawPublicKey.length, pos); pos += 4;
  rawPublicKey.copy(sshBuf, pos);

  const sshPublicKey = `ssh-ed25519 ${sshBuf.toString('base64')}`;

  // Encode private key as OpenSSH PEM (DER → base64 wrapped in header)
  const privateKeyB64 = privateKey.toString('base64');
  const wrapped = privateKeyB64.match(/.{1,64}/g)?.join('\n') ?? privateKeyB64;
  const sshPrivateKey = `-----BEGIN PRIVATE KEY-----\n${wrapped}\n-----END PRIVATE KEY-----`;

  return { publicKey: sshPublicKey, privateKey: sshPrivateKey };
};

/**
 * Calculates SSH key fingerprint (SHA-256, base64)
 */
const calculateFingerprint = (publicKey: string): string => {
  const parts = publicKey.split(' ');
  if (parts.length < 2) {
    throw new Error('Invalid public key format');
  }

  const keyData = Buffer.from(parts[1], 'base64');
  const hash = crypto.createHash('sha256').update(keyData).digest();
  return `SHA256:${hash.toString('base64')}`;
};

/**
 * Generates an SSH key pair of the specified type
 */
export const generateSSHKeyPair = (keyType: SSHKeyType): SSHKeyPair => {
  let publicKey: string;
  let privateKey: string;
  let pemPublicKey: string | undefined;

  switch (keyType) {
    case 'RSA2048': {
      const { publicKey: pubPem, privateKey: privPem } = crypto.generateKeyPairSync('rsa', {
        modulusLength: 2048,
        publicKeyEncoding: { type: 'spki', format: 'pem' },
        privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
      });
      publicKey = rsaPublicKeyToSSH(pubPem);
      privateKey = privPem;
      pemPublicKey = pubPem;
      break;
    }

    case 'RSA4096': {
      const { publicKey: pubPem, privateKey: privPem } = crypto.generateKeyPairSync('rsa', {
        modulusLength: 4096,
        publicKeyEncoding: { type: 'spki', format: 'pem' },
        privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
      });
      publicKey = rsaPublicKeyToSSH(pubPem);
      privateKey = privPem;
      pemPublicKey = pubPem;
      break;
    }

    case 'Ed25519': {
      const keypair = generateEd25519KeyPair();
      publicKey = keypair.publicKey;
      privateKey = keypair.privateKey;
      break;
    }

    case 'X25519': {
      const keypair = generateX25519KeyPair();
      publicKey = keypair.publicKey;
      privateKey = keypair.privateKey;
      break;
    }

    default:
      throw new Error(`Unsupported key type: ${keyType}`);
  }

  const fingerprint = calculateFingerprint(publicKey);

  return {
    publicKey,
    privateKey,
    keyType,
    fingerprint,
    pemPublicKey
  };
};
