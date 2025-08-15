from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import Optional


class Settings(BaseSettings):
    """
    Manages application settings and secrets by loading them from environment
    variables or a .env file. It provides validation and type casting.
    """
    # --- Core API Keys (Required) ---
    # These must be set in your environment or .env file.
    PINECONE_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    TAVILY_API_KEY: str 
    
    # --- LangSmith Tracing (Optional) ---
    # Set to "true" or "1" in your .env file to enable LangSmith tracing.
    LANGSMITH_TRACING_V2: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    

    @model_validator(mode='after')
    def _check_langsmith_settings(self) -> 'Settings':
        """
        If LangSmith tracing is enabled, this validator ensures that the
        necessary API key and project name are also provided.
        """
        if self.LANGSMITH_TRACING_V2 == "true":
            if not self.LANGSMITH_API_KEY:
                raise ValueError(
                    "LANGSMITH_API_KEY must be set if LANGSMITH_TRACING_V2 is enabled."
                )
            if not self.LANGSMITH_PROJECT:
                raise ValueError(
                    "LANGSMITH_PROJECT must be set if LANGSMITH_TRACING_V2 is enabled."
                )
        return self

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",        # Load from a .env file
        env_file_encoding="utf-8",
        extra="ignore"          # Ignore extra fields from the environment
    )


# Create a single, importable instance of the settings
settings = Settings()