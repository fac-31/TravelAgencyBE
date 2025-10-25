from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Travel Agency Backend"
    api_version: str = "v1"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS Settings
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    allowed_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: list[str] = ["*"]

    # OpenAI Settings (for LangChain)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # LangGraph Settings
    max_iterations: int = 10
    recursion_limit: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
