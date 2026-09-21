"""Tests for RAGRetrievalService — merge, dedup, context assembly."""

from __future__ import annotations

import pytest

from src.rag.models import RAGResult
from src.rag.rag_service import RAGRetrievalService


def _make_result(
    source_id: str,
    source_type: str = "chroma",
    score: float = 0.8,
    content: str = "测试内容",
    source_name: str = "测试来源",
) -> RAGResult:
    return RAGResult(
        content=content,
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        source_name=source_name,
        score=score,
    )


class TestMergeAndRank:
    """_merge_and_rank — dedup + sorting logic."""

    def test_dedup_by_source_id_across_sources(self) -> None:
        """One document retrieved by both legs collapses to a single entry.

        ``source_id`` is the document identity; ``source_type`` is only
        provenance.  The same knowledge point arriving from Chroma (vector)
        and Neo4j (keyword) must NOT be counted twice — that used to leak
        duplicates into results and made recall@k exceed 1.0.
        """
        chroma_hit = _make_result("kp-1", "chroma", score=0.9)
        chroma_lower = _make_result("kp-1", "chroma", score=0.5)
        neo4j_hit = _make_result("kp-1", "neo4j", score=0.7)

        merged = RAGRetrievalService._merge_and_rank(
            [chroma_hit, chroma_lower, neo4j_hit], "test", method="score"
        )
        assert len(merged) == 1
        assert merged[0].source_id == "kp-1"
        # Highest score wins regardless of which leg produced it.
        assert merged[0].score == 0.9
        assert merged[0].source_type == "chroma"

    def test_rrf_accumulates_across_sources(self) -> None:
        """A document hit by BOTH legs outranks one hit by only one leg.

        This is the documented behaviour of RRF (``Σ 1/(k+rank)`` over
        sources) and it only works if results are grouped by ``source_id``.
        """
        # kp-both is retrieved by chroma (rank 1) and neo4j (rank 1);
        # kp-vector-only only by chroma (rank 2) — but with a higher raw score.
        both_chroma = _make_result("kp-both", "chroma", score=0.5)
        both_neo4j = _make_result("kp-both", "neo4j", score=0.9)
        vector_only = _make_result("kp-vector-only", "chroma", score=0.99)

        merged = RAGRetrievalService._merge_and_rank(
            [both_chroma, both_neo4j, vector_only], "test", method="rrf", k=60
        )
        ids = [r.source_id for r in merged]
        assert len(ids) == len(set(ids)), f"duplicates leaked: {ids}"
        assert ids[0] == "kp-both", f"two-leg hit should win, got {ids}"

    def test_sorted_by_score_desc(self) -> None:
        """Results are sorted by score descending."""
        results = [
            _make_result("a", score=0.5),
            _make_result("b", score=0.9),
            _make_result("c", score=0.7),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="score")
        scores = [r.score for r in merged]
        assert scores == [0.9, 0.7, 0.5]

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert RAGRetrievalService._merge_and_rank([], "test") == []


class TestAssembleContext:
    """assemble_context — context string assembly from results."""

    async def test_basic_assembly(self) -> None:
        """Context includes source headers + content snippets."""
        results = [
            _make_result("kp-1", content="数组插入O(n)", source_name="数组"),
            _make_result("kp-2", content="链表插入O(1)", source_name="链表"),
        ]
        ctx = await RAGRetrievalService().assemble_context(results, max_chars=500)
        assert "[1]" in ctx.context_str
        assert "[2]" in ctx.context_str
        assert "数组" in ctx.context_str
        assert len(ctx.sources) == 2

    async def test_max_chars_truncation(self) -> None:
        """Context is truncated at max_chars boundary."""
        results = [
            _make_result("kp-1", content="A" * 300, source_name="长内容"),
            _make_result("kp-2", content="B" * 300, source_name="更多内容"),
        ]
        ctx = await RAGRetrievalService().assemble_context(results, max_chars=100)
        assert len(ctx.context_str) <= 300  # 100 chars + overhead for headers

    async def test_no_results(self) -> None:
        """Empty results produce fallback message."""
        ctx = await RAGRetrievalService().assemble_context([], max_chars=500)
        assert "未找到相关参考资料" in ctx.context_str
        assert len(ctx.sources) == 0


class TestNeo4jKeywordScore:
    """_neo4j_keyword_score 匹配质量打分。"""

    def test_exact_match_highest(self) -> None:
        assert RAGRetrievalService._neo4j_keyword_score("数组", "数组") == 1.0

    def test_prefix_match(self) -> None:
        assert RAGRetrievalService._neo4j_keyword_score("数组", "数组基础") == 0.85
        assert RAGRetrievalService._neo4j_keyword_score("数组基础", "数组") == 0.85

    def test_substring_match(self) -> None:
        assert RAGRetrievalService._neo4j_keyword_score("数组", "动态数组") == 0.7

    def test_no_overlap_fallback(self) -> None:
        """完全不相关的两词回退到兜底分。"""
        assert RAGRetrievalService._neo4j_keyword_score("xyz", "数组") == 0.4

    def test_case_insensitive(self) -> None:
        assert RAGRetrievalService._neo4j_keyword_score("HashMap", "hashmap") == 1.0

    def test_empty_input_returns_zero(self) -> None:
        assert RAGRetrievalService._neo4j_keyword_score("", "数组") == 0.0
        assert RAGRetrievalService._neo4j_keyword_score("数组", "") == 0.0

    def test_token_overlap_scales(self) -> None:
        """部分词项重叠时分值在 [0.5, 0.95] 之间。"""
        score_full = RAGRetrievalService._neo4j_keyword_score("数组 链表", "数组")
        score_partial = RAGRetrievalService._neo4j_keyword_score("数组 哈希表 红黑树", "数组")
        # 全 token 命中 → 高分；仅部分命中 → 中等分
        assert 0.5 <= score_partial <= 0.95
        assert score_full >= score_partial


class TestMergeAndRankRRF:
    """Reciprocal Rank Fusion 跨源融合。"""

    def test_default_method_is_rrf(self) -> None:
        assert RAGRetrievalService.default_fusion_method == "rrf"

    def test_single_source_preserves_order(self) -> None:
        """单源时 RRF 仅取名次，排序结果与 score 排序一致。"""
        chroma = [_make_result(f"k{i}", "chroma", score=1 - i * 0.1) for i in range(5)]
        merged = RAGRetrievalService._merge_and_rank(chroma, "test", method="rrf", k=60)
        ids = [r.source_id for r in merged]
        assert ids == [f"k{i}" for i in range(5)]
        # rrf scores 严格单调递减
        scores = [r.score for r in merged]
        assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))

    def test_cross_source_rrf_combines_rankings(self) -> None:
        """跨源同一文档 RRF 分累加。"""
        # chroma: k1(高) > k2；neo4j: k2(高) > k1
        results = [
            _make_result("k1", "chroma", score=0.9),
            _make_result("k2", "chroma", score=0.5),
            _make_result("k1", "neo4j", score=0.4),
            _make_result("k2", "neo4j", score=0.95),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="rrf")
        # k1 rank 1 (chroma) + rank 2 (neo4j) = 1/(60+1) + 1/(60+2)
        # k2 rank 2 (chroma) + rank 1 (neo4j) = 1/(60+2) + 1/(60+1)  → 平局
        # 平局时 stable，验证二者 rrf 相等，且 score 都介于 1/62 与 2/61 之间
        scores = {r.source_id: r.score for r in merged}
        assert abs(scores["k1"] - scores["k2"]) < 1e-9
        assert 1 / 62 <= scores["k1"] <= 2 / 61

    def test_rrf_robust_to_score_scale_mismatch(self) -> None:
        """RRF 对量纲差异免疫：chroma 普遍高分、neo4j 普遍低分时仍按名次融合。"""
        results = [
            _make_result("k1", "chroma", score=10.0),
            _make_result("k2", "chroma", score=5.0),
            _make_result("k3", "neo4j", score=0.9),
            _make_result("k4", "neo4j", score=0.5),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="rrf", k=60)
        # chroma 内部 rank: k1=1, k2=2；neo4j 内部 rank: k3=1, k4=2
        # k1=1/(60+1); k2=1/(60+2); k3=1/(60+1); k4=1/(60+2)
        ids_top2 = {merged[0].source_id, merged[1].source_id}
        assert ids_top2 == {"k1", "k3"}

    def test_rrf_differs_from_raw_score_method(self) -> None:
        """RRF 与 score 策略在同一数据集上产生不同结果（量纲差异场景）。"""
        # chroma 全部 10.0；neo4j 全部 0.01 → score 法 chroma 全在前；rrf 平局按名次
        results = [
            _make_result("k1", "chroma", score=10.0),
            _make_result("k2", "chroma", score=10.0),
            _make_result("k1", "neo4j", score=0.01),
            _make_result("k2", "neo4j", score=0.01),
        ]
        by_score = RAGRetrievalService._merge_and_rank(results, "test", method="score")
        by_rrf = RAGRetrievalService._merge_and_rank(results, "test", method="rrf")
        # score 法 chroma 分高 → 在前两条；rrf 法 k1/k2 平局但分远低于 1.0
        assert [r.source_type for r in by_score[:2]] == ["chroma", "chroma"]
        assert max(r.score for r in by_rrf) < 1.0  # 1/61 < 1.0
        assert max(r.score for r in by_score) == 10.0


class TestMergeAndRankMinMax:
    """Min-Max 归一化融合。"""

    def test_normalizes_per_source(self) -> None:
        """chroma / neo4j 分别归一化后再统一排序。"""
        results = [
            _make_result("k1", "chroma", score=10.0),
            _make_result("k2", "chroma", score=5.0),
            _make_result("k3", "neo4j", score=0.01),
            _make_result("k4", "neo4j", score=0.99),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="minmax")
        # k1 在 chroma 内最大 → 1.0；k3 在 neo4j 内最小 → 0.0
        scores_by_id = {r.source_id: r.score for r in merged}
        assert scores_by_id["k1"] == pytest.approx(1.0)
        assert scores_by_id["k3"] == pytest.approx(0.0)
        assert scores_by_id["k2"] == pytest.approx(0.0)
        assert scores_by_id["k4"] == pytest.approx(1.0)

    def test_single_element_source(self) -> None:
        """单元素源归一化为 0.5。"""
        results = [
            _make_result("k1", "chroma", score=10.0),
            _make_result("k2", "neo4j", score=0.99),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="minmax")
        for r in merged:
            assert r.score == pytest.approx(0.5)


class TestFusionMethodSelectable:
    """method 参数可切换，并随类默认值生效。"""

    def test_default_method_inherits_from_class(self) -> None:
        """未传 method 时使用类默认 (rrf)。"""
        results = [_make_result("k1", "chroma", score=0.5)]
        merged = RAGRetrievalService._merge_and_rank(results, "test")
        # 1/(60+1) ≈ 0.01639 —— 与 score=0.5 不同，确认不是 score 法
        assert merged[0].score == pytest.approx(1 / 61)

    def test_invalid_method_falls_back_to_rrf(self) -> None:
        """未知 method 字符串回退到 RRF（不抛错）。"""
        results = [_make_result("k1", "chroma", score=0.5)]
        merged = RAGRetrievalService._merge_and_rank(results, "test", method="bogus")
        assert merged[0].score == pytest.approx(1 / 61)

    def test_instance_fusion_method_used_by_dispatch(self) -> None:
        """实例属性 self.fusion_method 控制 search() 内分发。"""
        svc = RAGRetrievalService()
        svc.fusion_method = "score"
        # 通过类方法验证：使用 instance dispatch 路径不会破坏 default
        assert svc.fusion_method == "score"
        # 改回默认也工作
        svc.fusion_method = "rrf"
        assert svc.fusion_method == "rrf"


class TestExtractSearchTerms:
    """_extract_search_terms — 自然语言查询的分词抽取。"""

    def test_chinese_natural_language_query(self) -> None:
        """中文自然语言查询抽出内容词，停用/短词被过滤。"""
        terms = RAGRetrievalService._extract_search_terms("什么是二叉树的遍历")
        assert "二叉树" in terms or "遍历" in terms
        # 单字与空串不会出现
        assert all(len(t) >= 2 for t in terms)
        # 去重保序
        assert len(terms) == len(set(terms))

    def test_ascii_terms_lowercased(self) -> None:
        """英文词统一小写。"""
        terms = RAGRetrievalService._extract_search_terms("BFS and DFS Graph")
        assert "bfs" in terms and "dfs" in terms and "graph" in terms


class TestNeo4jSearchTermFallback:
    """_neo4j_search 整句匹配失败后的分词兜底。"""

    @staticmethod
    def _kp(kp_id: str, name: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=kp_id,
            name=name,
            description=f"{name}的概念",
            category="数据结构",
            difficulty=2,
            prerequisites=[],
        )

    async def test_falls_back_to_term_search(self) -> None:
        """整句 CONTAINS 无结果时按分词逐个检索，且去重。"""
        from unittest.mock import AsyncMock

        kp_tree = self._kp("kp-bst", "二叉搜索树")
        kp_graph = self._kp("kp-graph", "图")
        calls: list[str] = []

        async def fake_search(term, **kwargs):
            calls.append(term)
            if term == "什么是二叉搜索树和图":
                return []  # 整句无命中 → 触发兜底
            hits = []
            if "二叉" in term or "搜索" in term:
                hits.append(kp_tree)
            if "图" in term:
                hits.extend([kp_graph, kp_tree])  # 重复出现，验证去重
            return hits

        kp_repo = AsyncMock()
        kp_repo.search_by_name = fake_search
        svc = RAGRetrievalService()
        svc._kp_repo = kp_repo

        results = await svc._neo4j_search("什么是二叉搜索树和图", top_k=5)
        ids = [r.source_id for r in results]
        # kp-bst 出现在多个分词结果中，只保留一次
        assert ids.count("kp-bst") == 1
        assert set(ids) == {"kp-bst", "kp-graph"}
        assert all(r.source_type == "neo4j" for r in results)
        # 第一次是整句查询，其后为分词查询
        assert calls[0] == "什么是二叉搜索树和图"
        assert len(calls) >= 2

    async def test_no_fallback_when_full_query_hits(self) -> None:
        """整句已命中时不再触发分词检索。"""
        from unittest.mock import AsyncMock

        kp_repo = AsyncMock()
        kp_repo.search_by_name = AsyncMock(return_value=[self._kp("kp-bst", "二叉搜索树")])
        svc = RAGRetrievalService()
        svc._kp_repo = kp_repo

        results = await svc._neo4j_search("二叉搜索树", top_k=5)
        assert len(results) == 1
        assert kp_repo.search_by_name.call_count == 1

    async def test_two_gram_fallback_when_all_terms_miss(self) -> None:
        """jieba 分出的词全部未命中时，朴素 2-gram 片段兜底仍可召回。

        口语化整句「什么是二叉树」分出的词（'什么'/'二叉树' 等）都匹配不到
        图中节点「二叉搜索树」（CONTAINS 语义下 "二叉树" 不是其子串）——
        降级用 2-gram「二叉」仍可命中。
        """
        from unittest.mock import AsyncMock

        calls: list[str] = []
        kp_bst = self._kp("kp-bst", "二叉搜索树")

        async def fake_search(term, **kwargs):
            calls.append(term)
            if term == "二叉":  # 只有 2-gram 片段能命中
                return [kp_bst]
            return []

        kp_repo = AsyncMock()
        kp_repo.search_by_name = fake_search
        svc = RAGRetrievalService()
        svc._kp_repo = kp_repo

        results = await svc._neo4j_search("什么是二叉树", top_k=5)

        assert [r.source_id for r in results] == ["kp-bst"]
        assert "二叉" in calls  # 2-gram 兜底确实被触发


class TestSearchResultCache:
    """The retrieval-result TTL cache on RAGRetrievalService.search()."""

    async def test_cache_hit_avoids_second_retrieval(self) -> None:
        from unittest.mock import AsyncMock

        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc.enable_result_cache(maxsize=8, ttl_seconds=300.0)
        calls = {"n": 0}

        async def fake_chroma(query, top_k):
            calls["n"] += 1
            return [_make_result("kp-1", "chroma", score=0.9)]

        svc._chroma_search = fake_chroma  # type: ignore[assignment]

        first = await svc.search("什么是时间复杂度", top_k=5)
        second = await svc.search("什么是时间复杂度", top_k=5)

        assert calls["n"] == 1, "second identical query must be served from cache"
        assert [r.source_id for r in first] == [r.source_id for r in second]

    async def test_cache_returns_copies_not_shared_objects(self) -> None:
        """A caller mutating its result must not poison the cached entry."""
        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc.enable_result_cache(maxsize=8, ttl_seconds=300.0)

        async def fake_chroma(query, top_k):
            return [_make_result("kp-1", "chroma", score=0.9)]

        svc._chroma_search = fake_chroma  # type: ignore[assignment]

        warm = await svc.search("什么是栈", top_k=5)   # populates cache
        original_score = warm[0].score
        warm[0].score = -123.0  # caller mutates the object it was handed
        second = await svc.search("什么是栈", top_k=5)  # served from cache
        assert second[0].score == original_score, (
            "cached entry must not be mutated by callers"
        )

    async def test_use_cache_false_bypasses_cache(self) -> None:
        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc.enable_result_cache(maxsize=8, ttl_seconds=300.0)
        calls = {"n": 0}

        async def fake_chroma(query, top_k):
            calls["n"] += 1
            return [_make_result("kp-1", "chroma", score=0.9)]

        svc._chroma_search = fake_chroma  # type: ignore[assignment]

        await svc.search("什么是队列", top_k=5, use_cache=False)
        await svc.search("什么是队列", top_k=5, use_cache=False)
        assert calls["n"] == 2, "use_cache=False must re-run retrieval every time"

    async def test_cache_is_keyed_on_top_k(self) -> None:
        """Different top_k values must not share a cache entry."""
        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc.enable_result_cache(maxsize=8, ttl_seconds=300.0)
        calls: list[int] = []

        async def fake_chroma(query, top_k):
            calls.append(top_k)
            return [_make_result(f"kp-{top_k}", "chroma", score=0.9)]

        svc._chroma_search = fake_chroma  # type: ignore[assignment]

        await svc.search("什么是图", top_k=3)
        await svc.search("什么是图", top_k=5)
        assert calls == [3, 5]

    async def test_no_cache_configured_still_works(self) -> None:
        """Default service (cache disabled) must behave exactly as before."""
        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        calls = {"n": 0}

        async def fake_chroma(query, top_k):
            calls["n"] += 1
            return [_make_result("kp-1", "chroma", score=0.9)]

        svc._chroma_search = fake_chroma  # type: ignore[assignment]
        await svc.search("什么是树", top_k=5)
        await svc.search("什么是树", top_k=5)
        assert calls["n"] == 2


class TestSearchQueryRewrite:
    """LLM rewrite integration in search() — vector leg only."""

    async def test_rewrite_applies_to_vector_leg_only(self) -> None:
        from unittest.mock import AsyncMock

        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc._kp_repo = object()       # enables the KG leg
        svc._rewrite_enabled = True
        svc._rewriter = AsyncMock()
        svc._rewriter.rewrite = AsyncMock(return_value="数组 链表 区别 插入复杂度")

        seen: dict[str, str] = {}

        async def fake_chroma(query, top_k):
            seen["chroma"] = query
            return [_make_result("kp-1", "chroma", score=0.9)]

        async def fake_neo4j(query, top_k):
            seen["neo4j"] = query
            return []

        svc._chroma_search = fake_chroma  # type: ignore[assignment]
        svc._neo4j_search = fake_neo4j  # type: ignore[assignment]

        await svc.search("它和上一个有什么区别？", top_k=5)

        assert seen["chroma"] == "数组 链表 区别 插入复杂度", "vector leg uses rewrite"
        assert seen["neo4j"] == "它和上一个有什么区别？", "KG leg keeps the original query"

    async def test_rewrite_failure_falls_back_to_original(self) -> None:
        from unittest.mock import AsyncMock

        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc._rewrite_enabled = True
        svc._rewriter = AsyncMock()
        svc._rewriter.rewrite = AsyncMock(side_effect=RuntimeError("boom"))

        seen: dict[str, str] = {}

        async def fake_chroma(query, top_k):
            seen["chroma"] = query
            return []

        svc._chroma_search = fake_chroma  # type: ignore[assignment]

        await svc.search("什么是红黑树", top_k=5)
        assert seen["chroma"] == "什么是红黑树"

    async def test_rewrite_disabled_by_default(self) -> None:
        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        assert svc._rewrite_enabled is False
        assert svc._rewriter is None


class TestFusionStrategyViaSearch:
    """The fusion strategy must be honoured through ``search()``.

    ``TestFusionMethodSelectable`` calls ``_merge_and_rank`` as a staticmethod
    with ``method=`` passed explicitly, so it validates the merge maths but not
    the *dispatch*: production calls it as
    ``self._merge_and_rank(results, chroma_query, method=self.fusion_method,
    k=self.fusion_k)``, where those are instance attributes set from
    ``settings.hybrid_fusion_method`` (src/main.py:110).  A regression in that
    wiring — or in which query string is passed — is invisible to those tests.

    These drive ``search()`` with a configured instance, mirroring
    ``TestSearchResultCache`` in this file.
    """

    @staticmethod
    def _svc_with_two_legs(fusion: str):
        """A service whose chroma and neo4j legs both return known hits."""
        from unittest.mock import AsyncMock

        svc = RAGRetrievalService()
        svc._vector_index = object()  # enables the vector leg
        svc.fusion_method = fusion
        svc._kp_repo = object()  # enables the neo4j leg

        async def fake_chroma(query, top_k):
            # Only the chroma leg sees the (possibly rewritten/expanded) query.
            return [_make_result("kp-chroma", "chroma", score=0.9)]

        async def fake_neo4j(query, top_k):
            return [_make_result("kp-neo4j", "neo4j", score=0.8)]

        svc._chroma_search = fake_chroma          # type: ignore[assignment]
        svc._neo4j_search = AsyncMock(side_effect=fake_neo4j)
        return svc

    async def test_search_uses_instance_fusion_method(self) -> None:
        """The configured strategy is the one applied (not the class default)."""
        from unittest.mock import patch

        svc = self._svc_with_two_legs("minmax")
        with patch.object(
            RAGRetrievalService, "_merge_and_rank", autospec=True,
            return_value=[],
        ) as spy:
            await svc.search("什么是栈", top_k=5)

        assert spy.called, "merge was never reached through search()"
        assert spy.call_args.kwargs["method"] == "minmax", (
            "search() must pass the instance's fusion_method"
        )

    async def test_search_passes_instance_fusion_k(self) -> None:
        svc = self._svc_with_two_legs("rrf")
        svc.fusion_k = 17
        from unittest.mock import patch

        with patch.object(
            RAGRetrievalService, "_merge_and_rank", autospec=True,
            return_value=[],
        ) as spy:
            await svc.search("什么是栈", top_k=5)
        assert spy.call_args.kwargs["k"] == 17

    async def test_both_legs_contribute_results(self) -> None:
        """Both recall paths reach the merge — neither is silently skipped."""
        svc = self._svc_with_two_legs("rrf")
        results = await svc.search("什么是栈", top_k=5)
        ids = {r.source_id for r in results}
        assert ids == {"kp-chroma", "kp-neo4j"}, ids

    async def test_default_fusion_is_rrf(self) -> None:
        """A fresh service defaults to RRF (the documented default)."""
        svc = RAGRetrievalService()
        assert svc.fusion_method == "rrf"

    async def test_neo4j_leg_receives_original_query_not_rewritten(self) -> None:
        """The keyword leg must get the user's real words, not a rewrite.

        ``_neo4j_search`` matches on exact names, so a rewritten query would
        break precise matching for knowledge-point names.
        """
        svc = self._svc_with_two_legs("rrf")
        from unittest.mock import AsyncMock

        svc._rewriter = AsyncMock()
        svc._rewriter.rewrite = AsyncMock(return_value="REWRITTEN")
        svc._rewrite_enabled = True

        await svc.search("原始查询", top_k=5)

        # chroma leg (via _chroma_search) may see the rewrite; neo4j must not.
        assert svc._neo4j_search.call_args.args[0] == "原始查询"
