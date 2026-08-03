"""Knowledge Graph search tool — queries Neo4j for knowledge points."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.tools.base import BaseTool, ToolParameter, ToolResult, ToolSpec

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository

logger = logging.getLogger(__name__)


class KGTool(BaseTool):
    """Tool for searching the knowledge graph."""

    def __init__(self, kp_repo: KnowledgePointRepository | None = None) -> None:
        self._kp_repo = kp_repo

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_graph_search",
            description="搜索知识图谱，查找知识点及其关系。可用于了解某个主题的前置知识、相关概念和难度等级。",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="要搜索的知识点名称或关键词",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="number",
                    description="最大返回结果数（默认 5）",
                    required=False,
                ),
            ],
        )

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        if self._kp_repo is None:
            return ToolResult(success=False, error="Knowledge graph repository not available")

        try:
            kps = await self._kp_repo.search_by_name(query)
            kps = kps[:max_results]

            if not kps:
                return ToolResult(success=True, output=f"未找到与「{query}」相关的知识点")

            lines = [f"找到 {len(kps)} 个相关知识点：\n"]
            for i, kp in enumerate(kps, 1):
                prereq = kp.prerequisites or []
                lines.append(
                    f"{i}. {kp.name}\n"
                    f"   难度: {kp.difficulty}/5  |  分类: {kp.category}\n"
                    f"   描述: {kp.description}\n"
                    f"   前置知识: {', '.join(prereq) if prereq else '无'}\n"
                )

            return ToolResult(success=True, output="\n".join(lines), data=kps)
        except Exception as exc:
            logger.exception("KGTool search failed")
            return ToolResult(success=False, error=str(exc))
