from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI File Processing Website"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    API_KEY: str = os.getenv("API_KEY", "")
    MODEL: str = os.getenv("MODEL", "gpt-4o")

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ai_file_processing.db")

    API_BASE: str = os.getenv("API_BASE", "https://api.openai.com/v1")

    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    CHROMA_MEMORY_COLLECTION: str = os.getenv("CHROMA_MEMORY_COLLECTION", "user_memories")
    CHROMA_RAG_COLLECTION: str = os.getenv("CHROMA_RAG_COLLECTION", "rag_documents")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
