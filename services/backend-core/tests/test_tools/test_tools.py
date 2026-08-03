"""Comprehensive tests for the Tool system — ToolRegistry, BaseTool, and built-in tools.

Tests use mocks for all external dependencies (Neo4j, ChromaDB, RAG, forgetting curve).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import BaseTool, ToolParameter, ToolResult, ToolSpec
from src.tools.builtin.forgetting_check import ForgettingCheckTool
from src.tools.builtin.kg_search import KGTool
from src.tools.builtin.resource_search import ResourceSearchTool
from src.tools.registry import ToolRegistry


# ======================================================================
# Helper: concrete test tools for BaseTool/Registry testing
# ======================================================================


class _ConcreteTestTool(BaseTool):
    """Minimal concrete tool for testing BaseTool interface & Registry."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="input",
                    type="string",
                    description="Input value",
                    required=True,
                ),
                ToolParameter(
                    name="optional_param",
                    type="number",
                    description="Optional value",
                    required=False,
                ),
            ],
        )

    async def execute(self, input: str = "", optional_param: int = 0, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"executed: {input}")


class _OtherTestTool(BaseTool):
    """Second concrete tool for multi-tool tests."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="other_tool",
            description="Another test tool",
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="other done")


class _FailingTestTool(BaseTool):
    """Tool that always fails for testing error handling."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="failing_tool", description="Always fails")

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool execution failed")


# ======================================================================
# Helper: KnowledgePoint factory for KGTool tests
# ======================================================================


def _make_kp(
    id: str,
    name: str = "",
    difficulty: int = 3,
    category: str = "基本概念",
    description: str = "",
    prerequisites: list[str] | None = None,
):
    """Create a KnowledgePoint instance for test data."""
    from src.kg.models import KnowledgePoint

    return KnowledgePoint(
        id=id,
        name=name or id,
        description=description or f"Description for {id}",
        difficulty=difficulty,
        category=category,
        prerequisites=prerequisites or [],
    )


# ======================================================================
# TestToolRegistry
# ======================================================================


class TestToolRegistry:
    """ToolRegistry: register, get, list, execute, and prompt-building."""

    # ── Registration & retrieval ──────────────────────────────────────

    def test_register_and_get(self) -> None:
        """Register a tool and retrieve it by name."""
        registry = ToolRegistry()
        tool = _ConcreteTestTool()
        registry.register(tool)

        assert registry.get("test_tool") is tool
        assert registry.get("nonexistent") is None

    def test_register_overwrite_warning(self, caplog) -> None:
        """Registering the same tool name twice logs a warning."""
        caplog.set_level(logging.WARNING)

        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())
        registry.register(_ConcreteTestTool())  # duplicate name

        assert "Overwriting existing tool" in caplog.text

    def test_list_tools_empty(self) -> None:
        """list_tools returns empty list when no tools registered."""
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_list_tools_after_registration(self) -> None:
        """list_tools returns names of registered tools."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())
        assert registry.list_tools() == ["test_tool"]

    def test_list_tools_multiple(self) -> None:
        """list_tools returns all registered tool names."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())
        registry.register(_OtherTestTool())
        assert sorted(registry.list_tools()) == ["other_tool", "test_tool"]

    # ── Execution ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """Execute a registered tool with arguments."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())

        result = await registry.execute("test_tool", input="hello")

        assert result.success is True
        assert result.output == "executed: hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        """Executing an unregistered tool returns an error ToolResult."""
        registry = ToolRegistry()

        result = await registry.execute("unknown")

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_catches_exception(self) -> None:
        """When a tool raises, the registry wraps it in an error ToolResult."""
        registry = ToolRegistry()
        registry.register(_FailingTestTool())

        result = await registry.execute("failing_tool")

        assert result.success is False
        assert "tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_error_not_none(self) -> None:
        """Unknown tool result has a non-None error message."""
        registry = ToolRegistry()
        result = await registry.execute("does_not_exist")
        assert result.error is not None
        assert len(result.error) > 0

    # ── Build prompt block ────────────────────────────────────────────

    def test_build_prompt_block_empty(self) -> None:
        """With no registered tools, prompt block is an empty string."""
        registry = ToolRegistry()
        assert registry.build_prompt_block() == ""

    def test_build_prompt_block_single_tool(self) -> None:
        """A single tool appears in the prompt block."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())

        block = registry.build_prompt_block()

        assert block.startswith("## 可用工具")
        assert "### test_tool" in block
        assert "A test tool" in block
        assert "`input`" in block
        assert "`optional_param`" in block
        assert "（必填）" in block
        assert "（可选）" in block
        assert "!tool:{name}" in block

    def test_build_prompt_block_multiple_tools(self) -> None:
        """Multiple tools each contribute a section to the prompt block."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())
        registry.register(_OtherTestTool())

        block = registry.build_prompt_block()

        assert "### test_tool" in block
        assert "### other_tool" in block
        assert block.count("### ") == 2
        assert block.count("!tool:") == 2

    def test_build_prompt_block_includes_note(self) -> None:
        """The prompt block includes the 'one tool at a time' note."""
        registry = ToolRegistry()
        registry.register(_ConcreteTestTool())

        block = registry.build_prompt_block()

        assert "每次只调用一个工具" in block
        assert "直接回答即可" in block


# ======================================================================
# TestBaseTool / ToolSpec / ToolParameter / ToolResult
# ======================================================================


class TestBaseToolInterface:
    """BaseTool abstract interface enforcement."""

    def test_cannot_instantiate_abstract(self) -> None:
        """BaseTool cannot be instantiated directly (abstract methods)."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_subclass_can_instantiate(self) -> None:
        """A concrete subclass of BaseTool instantiates successfully."""
        tool = _ConcreteTestTool()
        assert isinstance(tool, BaseTool)

    def test_spec_property_returns_tool_spec(self) -> None:
        """spec property returns a ToolSpec instance."""
        tool = _ConcreteTestTool()
        spec = tool.spec
        assert isinstance(spec, ToolSpec)
        assert spec.name == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self) -> None:
        """execute returns a ToolResult."""
        tool = _ConcreteTestTool()
        result = await tool.execute(input="direct")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "executed: direct"

    def test_to_prompt_block_heading(self) -> None:
        """to_prompt_block starts with the tool heading."""
        tool = _ConcreteTestTool()
        block = tool.to_prompt_block()
        assert block.startswith("### test_tool")

    def test_to_prompt_block_contains_section_labels(self) -> None:
        """to_prompt_block contains parameter section and call format."""
        tool = _ConcreteTestTool()
        block = tool.to_prompt_block()
        assert "参数：" in block
        assert "调用格式：" in block

    def test_to_prompt_block_contains_parameter_details(self) -> None:
        """to_prompt_block lists each parameter with type and requirement."""
        tool = _ConcreteTestTool()
        block = tool.to_prompt_block()
        assert "`input` (string) （必填）" in block
        assert "`optional_param` (number) （可选）" in block

    def test_to_prompt_block_call_format(self) -> None:
        """to_prompt_block shows the call format for the tool."""
        tool = _ConcreteTestTool()
        block = tool.to_prompt_block()
        # The template uses literal {name} (not an f-string substitution)
        assert "!tool:{name}" in block


class TestToolSpec:
    """ToolSpec dataclass."""

    def test_default_parameters_empty(self) -> None:
        """ToolSpec defaults to empty parameters list."""
        spec = ToolSpec(name="test", description="desc")
        assert spec.parameters == []

    def test_with_parameters(self) -> None:
        """ToolSpec stores provided parameters."""
        params = [ToolParameter(name="p1", type="string")]
        spec = ToolSpec(name="test", description="desc", parameters=params)
        assert len(spec.parameters) == 1
        assert spec.parameters[0].name == "p1"

    def test_name_and_description(self) -> None:
        """ToolSpec stores name and description."""
        spec = ToolSpec(name="search", description="Search tool")
        assert spec.name == "search"
        assert spec.description == "Search tool"


class TestToolParameter:
    """ToolParameter dataclass."""

    def test_default_values(self) -> None:
        """Default required is False, description is empty."""
        param = ToolParameter(name="test", type="string")
        assert param.required is False
        assert param.description == ""

    def test_required_parameter(self) -> None:
        """Required parameter has required=True."""
        param = ToolParameter(name="test", type="string", description="A param", required=True)
        assert param.required is True

    def test_type_and_name(self) -> None:
        """Stores name and type correctly."""
        param = ToolParameter(name="count", type="number")
        assert param.name == "count"
        assert param.type == "number"


class TestToolResult:
    """ToolResult dataclass."""

    def test_success_defaults(self) -> None:
        """Minimal success result has empty output and no error."""
        result = ToolResult(success=True)
        assert result.success is True
        assert result.output == ""
        assert result.data is None
        assert result.error is None

    def test_failure_result(self) -> None:
        """Failure result carries an error message."""
        result = ToolResult(success=False, error="something went wrong")
        assert result.success is False
        assert result.error == "something went wrong"

    def test_all_fields(self) -> None:
        """ToolResult with all fields populated."""
        result = ToolResult(
            success=True,
            output="completed",
            data={"key": "value"},
            error=None,
        )
        assert result.success is True
        assert result.output == "completed"
        assert result.data == {"key": "value"}
        assert result.error is None


# ======================================================================
# TestKGTool — Knowledge Graph search
# ======================================================================


class TestKGTool:
    """KGTool: spec, execution with mocked repository."""

    # ── Spec ──────────────────────────────────────────────────────────

    def test_spec_name_and_description(self) -> None:
        """KGTool spec has correct name and relevant description."""
        tool = KGTool()
        spec = tool.spec
        assert spec.name == "knowledge_graph_search"
        assert "知识图谱" in spec.description

    def test_spec_parameters(self) -> None:
        """KGTool defines query (required) and max_results (optional)."""
        tool = KGTool()
        params = tool.spec.parameters
        param_names = [p.name for p in params]

        assert "query" in param_names
        assert "max_results" in param_names

        query_param = next(p for p in params if p.name == "query")
        assert query_param.required is True
        assert query_param.type == "string"

        max_param = next(p for p in params if p.name == "max_results")
        assert max_param.required is False
        assert max_param.type == "number"

    # ── Execution: no repo ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_no_repo_returns_error(self) -> None:
        """When kp_repo is None, execute returns an error."""
        tool = KGTool(kp_repo=None)
        result = await tool.execute(query="test")
        assert result.success is False
        assert "not available" in result.error

    # ── Execution: with results ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_with_results(self) -> None:
        """Successful search returns formatted knowledge point info."""
        mock_repo = AsyncMock()
        kps = [
            _make_kp("kp-1", name="基础知识", difficulty=2, category="基础", description="基础描述"),
            _make_kp("kp-2", name="进阶知识", difficulty=4, category="进阶", description="进阶描述", prerequisites=["kp-1"]),
        ]
        mock_repo.search_by_name = AsyncMock(return_value=kps)

        tool = KGTool(kp_repo=mock_repo)
        result = await tool.execute(query="知识")

        assert result.success is True
        assert "2 个相关知识点" in result.output
        assert "基础知识" in result.output
        assert "进阶知识" in result.output
        # Note: uses equality not identity because kps[:max_results] creates a new list
        assert result.data == kps
        mock_repo.search_by_name.assert_awaited_once_with("知识")

    # ── Execution: empty results ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_empty_results(self) -> None:
        """When no KPs match, returns success with a 'not found' message."""
        mock_repo = AsyncMock()
        mock_repo.search_by_name = AsyncMock(return_value=[])

        tool = KGTool(kp_repo=mock_repo)
        result = await tool.execute(query="nonexistent")

        assert result.success is True
        assert "未找到" in result.output
        assert result.data is None

    # ── Execution: max_results limit ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_respects_max_results(self) -> None:
        """max_results limits how many KPs are returned."""
        mock_repo = AsyncMock()
        many_kps = [_make_kp(f"kp-{i}", name=f"Point {i}") for i in range(10)]
        mock_repo.search_by_name = AsyncMock(return_value=many_kps)

        tool = KGTool(kp_repo=mock_repo)
        result = await tool.execute(query="Point", max_results=3)

        assert result.success is True
        assert "3 个相关知识点" in result.output
        assert len(result.data) == 3
        mock_repo.search_by_name.assert_awaited_once_with("Point")

    # ── Execution: exception handling ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_repo_exception_caught(self) -> None:
        """When the repo raises, the tool returns an error ToolResult."""
        mock_repo = AsyncMock()
        mock_repo.search_by_name = AsyncMock(side_effect=ConnectionError("Neo4j unavailable"))

        tool = KGTool(kp_repo=mock_repo)
        result = await tool.execute(query="test")

        assert result.success is False
        assert "Neo4j unavailable" in result.error

    # ── Edge case: default max_results ────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_default_max_results(self) -> None:
        """Default max_results is 5 when not specified."""
        mock_repo = AsyncMock()
        many_kps = [_make_kp(f"kp-{i}", name=f"Point {i}") for i in range(10)]
        mock_repo.search_by_name = AsyncMock(return_value=many_kps)

        tool = KGTool(kp_repo=mock_repo)
        result = await tool.execute(query="Point")

        assert result.success is True
        assert len(result.data) == 5  # default


# ======================================================================
# TestResourceSearchTool
# ======================================================================


class TestResourceSearchTool:
    """ResourceSearchTool: spec, execution with mocked RAG service."""

    # ── Spec ──────────────────────────────────────────────────────────

    def test_spec_name_and_description(self) -> None:
        """ResourceSearchTool spec has correct name and relevant description."""
        tool = ResourceSearchTool()
        spec = tool.spec
        assert spec.name == "resource_search"
        assert "学习资料" in spec.description

    def test_spec_parameters(self) -> None:
        """ResourceSearchTool defines query (required) and top_k (optional)."""
        tool = ResourceSearchTool()
        params = tool.spec.parameters
        param_names = [p.name for p in params]
        assert "query" in param_names
        assert "top_k" in param_names

        query_param = next(p for p in params if p.name == "query")
        assert query_param.required is True

        top_k_param = next(p for p in params if p.name == "top_k")
        assert top_k_param.required is False

    # ── Execution: no RAG service ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_no_rag_service(self) -> None:
        """When rag_service is None, execute returns an error."""
        tool = ResourceSearchTool(rag_service=None)
        result = await tool.execute(query="test")
        assert result.success is False
        assert "not available" in result.error

    # ── Execution: with results ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_with_results(self) -> None:
        """Successful search returns formatted resource info."""
        from src.rag.models import RAGContext, RAGResult

        mock_rag = AsyncMock()
        results = [
            RAGResult(
                content="这是一段示例内容",
                source_type="chroma",
                source_id="1",
                source_name="文档1",
                score=0.95,
            ),
            RAGResult(
                content="另一段内容",
                source_type="neo4j",
                source_id="2",
                source_name="文档2",
                score=0.85,
            ),
        ]
        mock_rag.search = AsyncMock(return_value=results)
        mock_context = RAGContext(context_str="context text", sources=results)
        mock_rag.assemble_context = AsyncMock(return_value=mock_context)

        tool = ResourceSearchTool(rag_service=mock_rag)
        result = await tool.execute(query="示例", top_k=3)

        assert result.success is True
        assert "2 条相关资料" in result.output
        assert "文档1" in result.output
        assert "文档2" in result.output
        # Note: uses equality not identity (Pydantic may copy the list during validation)
        assert result.data == results
        mock_rag.search.assert_awaited_once_with("示例", top_k=3)
        mock_rag.assemble_context.assert_awaited_once_with(results)

    # ── Execution: empty results ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_empty_results(self) -> None:
        """When no resources match, returns success with a 'not found' message."""
        from src.rag.models import RAGContext

        mock_rag = AsyncMock()
        mock_rag.search = AsyncMock(return_value=[])
        mock_context = RAGContext(context_str="", sources=[])
        mock_rag.assemble_context = AsyncMock(return_value=mock_context)

        tool = ResourceSearchTool(rag_service=mock_rag)
        result = await tool.execute(query="nonexistent")

        assert result.success is True
        assert "未找到" in result.output
        assert result.data is None

    # ── Execution: exception handling ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_rag_exception_caught(self) -> None:
        """When RAG service raises, the tool returns an error ToolResult."""
        mock_rag = AsyncMock()
        mock_rag.search = AsyncMock(side_effect=ConnectionError("ChromaDB unavailable"))

        tool = ResourceSearchTool(rag_service=mock_rag)
        result = await tool.execute(query="test")

        assert result.success is False
        assert "ChromaDB unavailable" in result.error

    # ── Edge case: default top_k ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_default_top_k(self) -> None:
        """Default top_k is 3 when not specified."""
        from src.rag.models import RAGContext, RAGResult

        mock_rag = AsyncMock()
        results = [RAGResult(content=f"result {i}", source_type="chroma", source_id=str(i), source_name=f"Doc {i}", score=0.5) for i in range(5)]
        mock_rag.search = AsyncMock(return_value=results)
        mock_context = RAGContext(context_str="", sources=results)
        mock_rag.assemble_context = AsyncMock(return_value=mock_context)

        tool = ResourceSearchTool(rag_service=mock_rag)
        result = await tool.execute(query="test")

        assert result.success is True
        # Verify search was called with default top_k=3
        mock_rag.search.assert_awaited_once_with("test", top_k=3)


# ======================================================================
# TestForgettingCheckTool
# ======================================================================


class TestForgettingCheckTool:
    """ForgettingCheckTool: spec, execution with mocked ForgettingCurveService."""

    # ── Spec ──────────────────────────────────────────────────────────

    def test_spec_name_and_description(self) -> None:
        """ForgettingCheckTool spec has correct name and relevant description."""
        tool = ForgettingCheckTool()
        spec = tool.spec
        assert spec.name == "forgetting_check"
        assert "遗忘曲线" in spec.description

    def test_spec_parameters(self) -> None:
        """ForgettingCheckTool defines user_id and kp_id, both required."""
        tool = ForgettingCheckTool()
        params = tool.spec.parameters
        param_names = [p.name for p in params]
        assert "user_id" in param_names
        assert "kp_id" in param_names
        for p in params:
            assert p.required is True

    # ── Execution: no service ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_no_service_returns_error(self) -> None:
        """When forgetting_service is None, execute returns an error."""
        tool = ForgettingCheckTool(forgetting_service=None)
        result = await tool.execute(user_id="user1", kp_id="kp-1")
        assert result.success is False
        assert "not available" in result.error

    # ── Execution: urgent recall ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_urgent_recall(self) -> None:
        """Recall probability < URGENT_RECALL_THRESHOLD shows urgent alert."""
        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=0.2)
        mock_service.get_state = AsyncMock(return_value=None)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🔴" in result.output
        assert "紧急复习" in result.output
        assert "20.00%" in result.output
        assert result.data["recall_probability"] == 0.2
        assert result.data["strength"] == 0
        assert result.data["review_count"] == 0
        mock_service.predict_recall.assert_awaited_once_with(kp_id="kp-1", user_id="user1")
        mock_service.get_state.assert_awaited_once_with(kp_id="kp-1", user_id="user1")

    # ── Execution: warning recall ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_warning_recall(self) -> None:
        """Recall probability < ALERT_RECALL_THRESHOLD shows warning alert."""
        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=0.5)
        mock_service.get_state = AsyncMock(return_value=None)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🟡" in result.output
        assert "建议复习" in result.output

    # ── Execution: good recall ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_good_recall(self) -> None:
        """Recall probability >= ALERT_RECALL_THRESHOLD shows good status."""
        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=0.8)
        mock_service.get_state = AsyncMock(return_value=None)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🟢" in result.output
        assert "记忆状态良好" in result.output

    # ── Execution: with state data ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_with_state(self) -> None:
        """When get_state returns a state, strength and review_count are used."""
        from src.learning_path.forgetting_curve import ForgettingState

        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=0.45)
        state = ForgettingState(kp_id="kp-1", user_id="user1", strength=12.5, review_count=3)
        mock_service.get_state = AsyncMock(return_value=state)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🟡" in result.output
        assert result.data["strength"] == 12.5
        assert result.data["review_count"] == 3

    # ── Execution: boundary recall values ─────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_boundary_alert_threshold(self) -> None:
        """Recall at exactly ALERT_RECALL_THRESHOLD is 'good' (>= threshold)."""
        from src.learning_path.forgetting_curve import ALERT_RECALL_THRESHOLD

        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=ALERT_RECALL_THRESHOLD)
        mock_service.get_state = AsyncMock(return_value=None)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🟢" in result.output  # >= threshold => good

    @pytest.mark.asyncio
    async def test_execute_boundary_urgent_threshold(self) -> None:
        """Recall just below ALERT_RECALL_THRESHOLD is 'warning'."""
        from src.learning_path.forgetting_curve import ALERT_RECALL_THRESHOLD

        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(return_value=ALERT_RECALL_THRESHOLD - 0.001)
        mock_service.get_state = AsyncMock(return_value=None)

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is True
        assert "🟡" in result.output  # below threshold => warning

    # ── Execution: exception handling ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_exception_caught(self) -> None:
        """When the service raises, the tool returns an error ToolResult."""
        mock_service = AsyncMock()
        mock_service.predict_recall = AsyncMock(side_effect=RuntimeError("Forgetting curve service crashed"))

        tool = ForgettingCheckTool(forgetting_service=mock_service)
        result = await tool.execute(user_id="user1", kp_id="kp-1")

        assert result.success is False
        assert "Forgetting curve service crashed" in result.error
