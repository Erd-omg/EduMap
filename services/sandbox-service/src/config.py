from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    default_timeout_seconds: int = 5
    default_memory_limit_mb: int = 256
    network_enabled: bool = False
    api_key: str = ""  # If set, all API requests must include Authorization: Bearer <api_key>
    max_concurrent_executions: int = 5  # Max concurrent sandbox executions
    rate_limit_per_minute: int = 10  # Max executions per IP per minute

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()

# In-memory rate limiter state
_IP_REQUEST_COUNTS: dict[str, tuple[int, float]] = {}
_CONCURRENT_SEMAPHORE = None
