from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    default_timeout_seconds: int = 5
    default_memory_limit_mb: int = 256
    network_enabled: bool = False

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
