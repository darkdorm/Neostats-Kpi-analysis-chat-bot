# config/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

CONFIG = {
    # ----------------- LLM Providers -----------------
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "openai").lower(),

    # OpenAI
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o"),

    # Groq
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),

    # ----------------- Web Search -----------------
    "CSE_API_KEY": os.getenv("CSE_API_KEY", ""),
    "CSE_CX": os.getenv("CSE_CX", ""),

    # ----------------- RAG Backend -----------------
    "RAG_BACKEND": os.getenv("RAG_BACKEND", "faiss").lower(),

    # ----------------- PostgreSQL -----------------
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "POSTGRES_PORT": int(os.getenv("POSTGRES_PORT", 5432)),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "neo_stats"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),

    # ----------------- General -----------------
    "DEBUG": os.getenv("DEBUG", "true").lower() in ("1", "true", "yes"),

    # ----------------- GitHub OAuth -----------------
    "GITHUB_CLIENT_ID": os.getenv("GITHUB_CLIENT_ID", ""),
    "GITHUB_CLIENT_SECRET": os.getenv("GITHUB_CLIENT_SECRET", ""),
    "GITHUB_REDIRECT_URI": os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8501/github_callback"),

    # ----------------- Google OAuth -----------------
    "GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID", ""),
    "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
    "GOOGLE_REDIRECT_URI": os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/google_callback"),
}
