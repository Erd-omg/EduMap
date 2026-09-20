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
    # 默认关闭：Cross-Encoder 需对 (query, doc) 逐对前向，会显著抬升 P95 延迟，
    # 在延迟敏感的实时问答路径上不划算。
    #
    # ⚠️ 指标口径（重要）：benchmark_results/ 下所有已记录报告均为
    #    reranker_enabled=False。因此仓库中的 MRR / HitRate 数值
    #    （n=20: MRR 0.900；n=200: MRR 0.841 / HR@3 0.930 / P@1 0.755）
    #    属于「未启用重排」的基线，**不可归因于重排收益**。
    #    重排 vs 基线的对照实验尚未执行；复现方式见
    #    scripts/run_rag_benchmark.py 的 --reranker 开关。
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Hybrid retrieval fusion ──────────────────────────────────────────────
    # ChromaDB 提供 1.0 - cosine_distance 的 [0, 1] 相似度；
    # Neo4j 关键字检索给出离散匹配质量（exact / prefix / substring / token-overlap），
    # 二者量纲不一致——直接拼接按 score 排序会让一边压另一边。
    #
    # 三种融合策略（默认 "rrf"）：
    #   - "score"  : 按各自原始 score 排序去重（保留各源最大值）；保持旧行为。
    #   - "minmax" : 各自在源内 min-max 归一化到 [0, 1] 后再排序；适合各源量纲差很大。
    #   - "rrf"    : Reciprocal Rank Fusion，score(d) = Σ 1/(k + rank)，只依赖顺序；
    #                不需要 score 校准，对单源故障最稳健（k=60 经典默认）。
    hybrid_fusion_method: str = "rrf"
    hybrid_rrf_k: int = 60

    # ── Query Expansion ─────────────────────────────────────────────────────
    # 默认关闭：图谱查询扩展会额外引入一次图遍历，且在评测集上收益有限，
    # 并存在放大噪声的风险。复现方式见 run_rag_benchmark.py 的 --expand-query。
    expand_query_enabled: bool = False
    expand_query_max_terms: int = 5

    # ── Memory ─────────────────────────────────────────────────────────────
    session_ttl_hours: int = 1
    max_conversation_turns: int = 50

    # ── Profile service ────────────────────────────────────────────────────
    # backend-core 读取用户画像（通知开关等）的服务地址。
    # host 模式（本机 uvicorn，默认）连 localhost；容器模式由 docker-compose
    # 显式覆盖为 http://profile-service:8001。
    profile_service_base: str = "http://localhost:8001"

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}


settings = Settings()
