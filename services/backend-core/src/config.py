from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "DEBUG"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "edumap_dev"
    chroma_host: str = "localhost"
    chroma_port: int = 8003
    redis_url: str = "redis://localhost:6379/0"
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = "spark"
    database_url: str = "postgresql://edumap:edumap_dev@localhost:5432/edumap"
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    api_key: str = ""  # If set, all API requests must include Authorization: Bearer <api_key>
    sandbox_url: str = "http://localhost:8002"

    # ── Chunking ───────────────────────────────────────────────────────────
    chunking_strategy: str = "auto"  # "auto" | "fixed" | "recursive" | "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Reranking ──────────────────────────────────────────────────────────
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Query Expansion ─────────────────────────────────────────────────────
    expand_query_enabled: bool = False
    expand_query_max_terms: int = 5

    # ── Memory ─────────────────────────────────────────────────────────────
    session_ttl_hours: int = 1
    max_conversation_turns: int = 50

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}


settings = Settings()
