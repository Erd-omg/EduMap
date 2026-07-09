from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "edumap_dev"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    redis_url: str = "redis://localhost:6379/0"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = "spark"
    database_url: str = ""
    llm_timeout: int = 60
    llm_max_retries: int = 3
    llm_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
