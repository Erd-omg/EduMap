from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    database_url: str = "postgresql://edumap:edumap_dev@localhost:5432/edumap"
    redis_url: str = "redis://localhost:6379/1"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = "spark"
    api_key: str = ""  # If set, all API requests must include Authorization: Bearer <api_key>

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}


settings = Settings()
