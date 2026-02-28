import mysql from 'mysql2/promise';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import { logger } from '../utils/logger';
import { config } from '../config';

dotenv.config();

const supabaseSchema = process.env.SUPABASE_SCHEMA;

// MySQL table creation queries
const createUsersTableMySQL = (tableName: string) => `
CREATE TABLE IF NOT EXISTS ${tableName} (
  id INT AUTO_INCREMENT PRIMARY KEY,
  api_user VARCHAR(255) NOT NULL UNIQUE,
  api_secret VARCHAR(255) NOT NULL,
  full_name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  total_logins INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
`;

const createKeyValuesTableMySQL = (tableName: string) => `
CREATE TABLE IF NOT EXISTS ${tableName} (
  id INT AUTO_INCREMENT PRIMARY KEY,
  uuid VARCHAR(36) NOT NULL UNIQUE,
  key_name VARCHAR(255) NOT NULL,
  encrypted_value TEXT NOT NULL,
  retrieved INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_uuid (uuid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
`;

const createApiRequestLogsTableMySQL = (tableName: string) => `
CREATE TABLE IF NOT EXISTS ${tableName} (
  id INT AUTO_INCREMENT PRIMARY KEY,
  request_uuid VARCHAR(36) NOT NULL UNIQUE,
  user_id INT NULL,
  method VARCHAR(10) NOT NULL,
  url VARCHAR(2048) NOT NULL,
  status_code INT NULL,
  encrypted_headers TEXT NOT NULL,
  encrypted_request_body TEXT NULL,
  encrypted_response_body TEXT NULL,
  ip_address VARCHAR(45) NOT NULL,
  user_agent TEXT NULL,
  response_time_ms INT NULL,
  error_message TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_request_uuid (request_uuid),
  INDEX idx_user_id (user_id),
  INDEX idx_method (method),
  INDEX idx_status_code (status_code),
  INDEX idx_created_at (created_at),
  INDEX idx_ip_address (ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
`;

// PostgreSQL table creation queries
const createUsersTablePostgreSQL = (tableName: string) => {
  const t = supabaseSchema ? `${supabaseSchema}.${tableName}` : tableName;
  return `
CREATE TABLE IF NOT EXISTS ${t} (
  id SERIAL PRIMARY KEY,
  api_user VARCHAR(255) NOT NULL UNIQUE,
  api_secret VARCHAR(255) NOT NULL,
  full_name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  total_logins INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE ${t} ENABLE ROW LEVEL SECURITY;

-- Create trigger for modified_at update
CREATE OR REPLACE FUNCTION update_modified_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_${tableName}_modified_at ON ${t};
CREATE TRIGGER update_${tableName}_modified_at
    BEFORE UPDATE ON ${t}
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_at_column();
`;
};

const createKeyValuesTablePostgreSQL = (tableName: string) => {
  const t = supabaseSchema ? `${supabaseSchema}.${tableName}` : tableName;
  return `
CREATE TABLE IF NOT EXISTS ${t} (
  id SERIAL PRIMARY KEY,
  uuid VARCHAR(36) NOT NULL UNIQUE,
  key_name VARCHAR(255) NOT NULL,
  encrypted_value TEXT NOT NULL,
  retrieved INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE ${t} ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_${tableName}_uuid ON ${t} (uuid);

DROP TRIGGER IF EXISTS update_${tableName}_modified_at ON ${t};
CREATE TRIGGER update_${tableName}_modified_at
    BEFORE UPDATE ON ${t}
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_at_column();
`;
};

const createApiRequestLogsTablePostgreSQL = (tableName: string) => {
  const t = supabaseSchema ? `${supabaseSchema}.${tableName}` : tableName;
  return `
CREATE TABLE IF NOT EXISTS ${t} (
  id SERIAL PRIMARY KEY,
  request_uuid VARCHAR(36) NOT NULL UNIQUE,
  user_id INTEGER NULL,
  method VARCHAR(10) NOT NULL,
  url VARCHAR(2048) NOT NULL,
  status_code INTEGER NULL,
  encrypted_headers TEXT NOT NULL,
  encrypted_request_body TEXT NULL,
  encrypted_response_body TEXT NULL,
  ip_address VARCHAR(45) NOT NULL,
  user_agent TEXT NULL,
  response_time_ms INTEGER NULL,
  error_message TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE ${t} ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_${tableName}_request_uuid ON ${t} (request_uuid);
CREATE INDEX IF NOT EXISTS idx_${tableName}_user_id ON ${t} (user_id);
CREATE INDEX IF NOT EXISTS idx_${tableName}_method ON ${t} (method);
CREATE INDEX IF NOT EXISTS idx_${tableName}_status_code ON ${t} (status_code);
CREATE INDEX IF NOT EXISTS idx_${tableName}_created_at ON ${t} (created_at);
CREATE INDEX IF NOT EXISTS idx_${tableName}_ip_address ON ${t} (ip_address);
`;
};

async function setupMySQLDatabase() {
  try {
    const connection = await mysql.createConnection({
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      port: parseInt(process.env.DB_PORT || '3306')
    });

    logger.info('Connected to MySQL database');

    const tablePrefix = config.database.tablePrefix;

    // Create core tables
    await connection.execute(createUsersTableMySQL(`${tablePrefix}users`));
    logger.info('Users table created successfully');

    await connection.execute(createKeyValuesTableMySQL(`${tablePrefix}key_values`));
    logger.info('Key-values table created successfully');

    // Create API request logs table
    await connection.execute(createApiRequestLogsTableMySQL(`${tablePrefix}api_request_logs`));
    logger.info('API request logs table created successfully');

    // Add foreign key constraint for API request logs
    await connection.execute(`
      ALTER TABLE ${tablePrefix}api_request_logs 
      ADD CONSTRAINT fk_api_request_logs_user_id 
      FOREIGN KEY (user_id) REFERENCES ${tablePrefix}users(id) 
      ON DELETE SET NULL;
    `).catch(() => {
      // Constraint might already exist
      logger.info('Foreign key constraint for api_request_logs user_id already exists or failed to create');
    });

    await connection.end();
    logger.info('MySQL database setup completed');
  } catch (error) {
    logger.error('Error setting up MySQL database:', error);
    throw error;
  }
}

function generateSupabaseSQL() {
  const tablePrefix = config.database.tablePrefix;
  const schemaPrefix = supabaseSchema ? `${supabaseSchema}.` : '';

  const schemaQueries = supabaseSchema ? [
    `-- Create schema and grant access to Supabase roles
CREATE SCHEMA IF NOT EXISTS ${supabaseSchema};

GRANT USAGE ON SCHEMA ${supabaseSchema} TO postgres, anon, authenticated, service_role;

ALTER DEFAULT PRIVILEGES IN SCHEMA ${supabaseSchema} GRANT ALL ON TABLES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA ${supabaseSchema} GRANT ALL ON SEQUENCES TO postgres, anon, authenticated, service_role;`
  ] : [];

  const allQueries = [
    createUsersTablePostgreSQL(`${tablePrefix}users`),
    createKeyValuesTablePostgreSQL(`${tablePrefix}key_values`),
    createApiRequestLogsTablePostgreSQL(`${tablePrefix}api_request_logs`)
  ];

  const constraintQueries = [
    `ALTER TABLE ${schemaPrefix}${tablePrefix}api_request_logs
     ADD CONSTRAINT fk_api_request_logs_user_id
     FOREIGN KEY (user_id) REFERENCES ${schemaPrefix}${tablePrefix}users(id)
     ON DELETE SET NULL;`
  ];

  return [...schemaQueries, ...allQueries, ...constraintQueries].join('\n\n');
}

async function setupSupabaseDatabase() {
  try {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SECRET_KEY) {
      throw new Error('Supabase configuration missing');
    }

    logger.info('='.repeat(80));
    logger.info('SUPABASE DATABASE SETUP REQUIRED');
    logger.info('='.repeat(80));
    logger.info('');
    logger.info('Supabase does not allow programmatic table creation through the API.');
    logger.info('You need to manually create the database tables using the Supabase SQL Editor.');
    logger.info('');
    logger.info('Please follow these steps:');
    logger.info('');
    logger.info('1. Go to your Supabase project dashboard');
    logger.info('2. Navigate to the SQL Editor');
    logger.info('3. Copy and paste the following SQL commands:');
    logger.info('');
    logger.info('-'.repeat(80));
    logger.info('SQL COMMANDS TO EXECUTE:');
    logger.info('-'.repeat(80));
    logger.info('');
    
    const sqlCommands = generateSupabaseSQL();
    console.log(sqlCommands);
    
    logger.info('');
    logger.info('-'.repeat(80));
    logger.info('');
    logger.info('4. Execute the SQL commands in the Supabase SQL Editor');
    logger.info('5. Verify that all tables have been created successfully');
    if (process.env.SUPABASE_SCHEMA && process.env.SUPABASE_SCHEMA !== 'public') {
      logger.info('');
      logger.info('IMPORTANT - Custom schema requires one additional step:');
      logger.info(`6. In the Supabase dashboard go to: Settings > API > Exposed schemas`);
      logger.info(`   Add "${process.env.SUPABASE_SCHEMA}" to the list and click Save.`);
      logger.info('   This allows PostgREST (the Supabase API layer) to access your schema.');
    }
    logger.info('');
    logger.info('After completing these steps, your Supabase database will be ready to use.');
    logger.info('='.repeat(80));

    // Test connection to verify Supabase is accessible
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SECRET_KEY,
      { 
        db: { schema: process.env.SUPABASE_SCHEMA }
      }
    );

    // Simple connection test
    const { error } = await supabase.from('information_schema.tables').select('table_name').limit(1);
    if (error && !error.message.includes('relation "information_schema.tables" does not exist')) {
      logger.error('Supabase connection test failed:', error);
      throw error;
    }

    logger.info('Supabase connection verified successfully');
  } catch (error) {
    logger.error('Error with Supabase setup:', error);
    throw error;
  }
}

async function setupDatabase() {
  try {
    if (config.database.type === 'mysql') {
      await setupMySQLDatabase();
    } else if (config.database.type === 'postgres') {
      await setupSupabaseDatabase();
    } else {
      throw new Error(`Unsupported database type: ${config.database.type}`);
    }
    
    logger.info('Database setup process completed successfully');
  } catch (error) {
    logger.error('Error setting up database:', error);
    process.exit(1);
  }
}

setupDatabase();