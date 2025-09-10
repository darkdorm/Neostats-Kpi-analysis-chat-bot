import psycopg2
from config.config import CONFIG
from datetime import datetime

# ----------------- PostgreSQL Connection -----------------
def get_conn():
    """
    Establish a connection to the PostgreSQL database using CONFIG variables.
    """
    return psycopg2.connect(
        host=CONFIG.get("POSTGRES_HOST", "localhost"),
        database=CONFIG.get("POSTGRES_DB", "showdb"),
        user=CONFIG.get("POSTGRES_USER", "postgres"),
        password=CONFIG.get("POSTGRES_PASSWORD", "")
    )

# ----------------- Save KPI -----------------
def save_kpi(name, value, growth):
    """
    Save a KPI record into the 'kpis' table.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kpis (name, value, growth, timestamp)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (name, value, growth, datetime.now())
                )
    except Exception as e:
        print("Error saving KPI:", e)

# ----------------- Save Chat -----------------
def save_chat(role, message):
    """
    Save a chat message into the 'chat_history' table.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_history (role, message, timestamp)
                    VALUES (%s, %s, %s)
                    """,
                    (role, message, datetime.now())
                )
    except Exception as e:
        print("Error saving chat:", e)

# ----------------- Save Insight -----------------
def save_insight(insight):
    """
    Save an auto-generated insight into the 'insights' table.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO insights (insight, timestamp)
                    VALUES (%s, %s)
                    """,
                    (insight, datetime.now())
                )
    except Exception as e:
        print("Error saving insight:", e)
