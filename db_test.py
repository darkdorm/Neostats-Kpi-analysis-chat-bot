# db_test.py
from sqlalchemy import create_engine, text

# PostgreSQL credentials
DB_USER = "postgres"
DB_PASSWORD = "21012004"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "neo_stats"

def get_engine():
    """Return SQLAlchemy engine"""
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return engine

def init_users_table():
    """Create users table if it does not exist"""
    engine = get_engine()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with engine.begin() as conn:  # ensures commit
            conn.execute(text(create_table_query))
            print("✅ Users table is ready.")
    except Exception as e:
        print("❌ Failed to create users table:", e)
