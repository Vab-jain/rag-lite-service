from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    COLLECTION_NAME: str = "rag-lite"

    class Config:
        env_file = ".env"

settings = Settings()
