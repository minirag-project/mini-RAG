from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    APP_DESCRIPTION: str

    FILE_TYPE: list[str]
    FILE_SIZE_LIMIT: int
    FILE_CHUNK_SIZE: int

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_DB: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: str
    OPENAI_API_URL: str

    COHERE_API_KEY: str

    GEMINI_API_KEY: str

    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int

    INPUT_DEFAULT_MAX_TOKENS: int
    GENERATION_DEFAULT_MAX_TOKENS: int
    GENERATION_DEFAULT_TEMPERATURE: float

    VECTOR_DB_BACKEND_LITERAL : List[str] = None
    VECTOR_DB_BACKEND: str  
    VECTOR_DB_PATH: str 
    VECTOR_DB_DISTANCE_METRIC: str
    VECTOR_DB_INDEX_THRESHOLD: int

    DEFAULT_LANGUAGE: str = "en"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


def get_settings():
    return Settings()