from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    database_url: str = "postgresql+asyncpg://edumap:edumap_dev@localhost:5432/edumap"
    redis_url: str = "redis://localhost:6379/1"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = "spark"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
