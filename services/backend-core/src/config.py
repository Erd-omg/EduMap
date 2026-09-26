from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# services/backend-core/src/config.py → parents[3] is the monorepo root.
_SERVICE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_env_file() -> str:
    """Locate the .env file independent of the current working directory.

    Preference order: the service directory, then the monorepo root.  Returns
    the first path that exists, or ``".env"`` (pydantic's default) if neither
    does, so behaviour is unchanged when no .env is present at all.
    """
    candidates = (_SERVICE_DIR / ".env", _REPO_ROOT / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ".env"


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
    # ── Cross-Encoder Reranker ──────────────────────────────────────────────
    # 默认关闭，且**实测证明必须保持关闭**（2026-09-25，n=200，缓存绕过）：
    #
    #   reranker=off : MRR 0.8945 / NDCG@1 0.8600 / P50 24.5ms
    #   reranker=on  : MRR 0.3250 / NDCG@1 0.2250 / P50 63.6ms   ← 崩了一半以上
    #   ΔMRR −0.5695，延迟 ×2.60
    #
    # 原因与 NVIDIA 的评测一致（arXiv 2409.07691）：**过小的 cross-encoder 会主动
    # 伤害检索**。当前默认模型 ms-marco-MiniLM-L-6-v2 只有 ~22M 参数，比该研究
    # 中已被证明有害的 33M MiniLM-L-12 还小。
    #
    # **若将来要启用，必须先换更大的模型再测**（如 BAAI/bge-reranker-v2-m3，568M，
    # 该研究中相对基线 +3.7~5 NDCG@10），不能只改开关。
    #
    # 复现：python scripts/run_reranker_ablation.py --dataset expanded
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Hybrid retrieval fusion ──────────────────────────────────────────────
    # ChromaDB 提供 1.0 - cosine_distance 的 [0, 1] 相似度；
    # Neo4j 关键字检索给出离散匹配质量（exact / prefix / substring / token-overlap），
    # 二者量纲不一致——直接拼接按 score 排序会让一边压另一边。
    #
    # 三种融合策略：
    #   - "score"  : 按各自原始 score 排序去重（保留各源最大值）。
    #   - "minmax" : 各自在源内 min-max 归一化到 [0, 1] 后再排序；适合各源量纲差很大。
    #   - "rrf"    : Reciprocal Rank Fusion，score(d) = Σ 1/(k + rank)，只依赖顺序；
    #                不需要 score 校准，对单源故障最稳健（k=60 是 Cormack 2009 的
    #                惯例值，原文自述"该选择并不关键"）。
    #
    # 默认值依据 **本仓 n=200 实测**（scripts/run_fusion_ablation.py，缓存绕过）：
    #   score  MRR 0.9002 / P@1 0.8600 / P50 18.6ms   ← 最优
    #   rrf    MRR 0.8569 / P@1 0.7700 / P50 22.1ms
    #   minmax MRR 0.8499 / P@1 0.7600 / P50 18.6ms
    # 即 score 相对 rrf 的 MRR 高 0.043（相对 +5%）、P@1 高 0.09，且延迟不劣。
    # 与 Bruch et al.（arXiv 2210.11934, ACM TOIS）"调优的加权分数融合优于 RRF"
    # 的结论一致。复现：python scripts/run_fusion_ablation.py --dataset expanded
    #
    # ⚠️ **证据边界（重要）**：上述数字全部来自**单一语料** —— `expanded_queries.json`
    #    的 `_meta.course_id == "cs201"`（数据结构与算法，22 个 KP，含自动生成条目）。
    #
    #    2026-09-27 补测 cs301（操作系统，10 KP）：
    #      cs201 (n=200): score 0.8851 > rrf 0.8370 > minmax 0.8197
    #      cs301 (n=40) : minmax 0.9875 > score 0.9750 > rrf 0.9542   ← 次序变化
    #    可确证的结论：**score 的优势不是跨语料普适的**，不存在无条件的默认最优。
    #    ⚠️ 但 cs301 语料的标注**尚未人工复核**（见其 _meta.review_note），所以
    #    "minmax 反超"的**具体幅度**属初步结果，可能因标注修正而消失——不要据此改动默认值。
    #
    #    因此：**当前默认值是在 cs201 上的选择，不是"最优策略"的声明**。
    #    换语料前请重跑 `scripts/run_fusion_ablation.py --course <id>`，不要外推。
    #
    # 回滚：改这个默认值，或用 `HYBRID_FUSION_METHOD` 环境变量单向覆盖（无需改代码）。
    # 值必须是 FUSION_METHODS 之一——拼错会在**启动时**报错（或在直接赋值时抛
    # ValueError），而不是静默退化成 rrf。
    hybrid_fusion_method: str = "score"
    hybrid_rrf_k: int = 60

    @field_validator("hybrid_fusion_method")
    @classmethod
    def _validate_fusion_method(cls, value: str) -> str:
        """Reject unknown fusion strategies at construction time.

        ``RAGRetrievalService._merge_and_rank`` dispatches on this string and
        falls through to RRF for anything unrecognised.  That fallback is a
        reasonable *runtime* policy for a programmatic caller, but as a
        *configuration* behaviour it is dangerous: a typo (``HYBRID_FUSION_METHOD=scoer``)
        would silently serve RRF, and a benchmark run against that deployment
        would report numbers attributable to a strategy nobody selected — with
        nothing in the logs to say so.  Failing loudly here means the mistake
        surfaces at boot instead of in a results table.
        """
        normalised = value.strip().lower()
        # Single source of truth: the strategies the dispatch actually
        # implements.  A second hard-coded list here could drift from it.
        from src.rag.rag_service import FUSION_METHODS

        if normalised not in FUSION_METHODS:
            raise ValueError(
                f"hybrid_fusion_method must be one of {FUSION_METHODS}, got {value!r}. "
                "An unrecognised value used to fall back to 'rrf' silently; "
                "set it explicitly instead."
            )
        return normalised

    # ── Query Expansion ─────────────────────────────────────────────────────
    # 默认关闭：图谱查询扩展会额外引入一次图遍历，且在评测集上收益有限，
    # 并存在放大噪声的风险。复现方式见 run_rag_benchmark.py 的 --expand-query。
    expand_query_enabled: bool = False
    expand_query_max_terms: int = 5

    # ── LLM Query Rewrite ───────────────────────────────────────────────────
    # 默认关闭，依据实测：n=20 评测集上 LLM 改写把 MRR 从 0.9056 降到 0.825
    # （−0.08），平均延迟从 38ms 涨到 1546ms（×9.8）。改写对"教科书式问法"
    # 的评测集是负收益——原句的语义结构反而被拆成了关键词。
    # 保留开关以便后续在"多轮指代"场景（"它和上一个有什么区别"）重测，
    # 那类查询必须结合短期记忆才有意义。
    # 复现：scripts/run_strategy_comparison.py --strategies direct,hybrid,rewrite
    rag_rewrite_enabled: bool = False
    rag_rewrite_timeout: float = 8.0

    # ── Retrieval result cache ──────────────────────────────────────────────
    # 对同一 (query, top_k) 的检索结果做 TTL 缓存，降低重复检索开销。
    # 知识更新的一致性由 TTL 兜底（默认 5 分钟）。
    rag_cache_enabled: bool = True
    rag_cache_ttl_seconds: int = 300
    rag_cache_size: int = 256

    # ── Memory ─────────────────────────────────────────────────────────────
    session_ttl_hours: int = 1
    max_conversation_turns: int = 50

    # ── Profile service ────────────────────────────────────────────────────
    # backend-core 读取用户画像（通知开关等）的服务地址。
    # host 模式（本机 uvicorn，默认）连 localhost；容器模式由 docker-compose
    # 显式覆盖为 http://profile-service:8001。
    profile_service_base: str = "http://localhost:8001"

    # ``env_file`` is resolved relative to the *current working directory*,
    # which silently breaks: the repo's only .env lives at the monorepo root,
    # but every script's usage says ``cd services/backend-core``.  Running from
    # the service directory used to fall back to defaults — llm_model="spark"
    # with no API key — i.e. a MOCK LLM returning canned text, while benchmarks
    # still printed plausible-looking tables.  Search both locations explicitly.
    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
        "env_file": _resolve_env_file(),
        "extra": "ignore",
    }


settings = Settings()
