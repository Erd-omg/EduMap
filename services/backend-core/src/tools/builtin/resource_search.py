"""Resource search tool — queries ChromaDB for uploaded resource chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.tools.base import BaseTool, ToolParameter, ToolResult, ToolSpec

if TYPE_CHECKING:
    from src.rag.rag_service import RAGRetrievalService

logger = logging.getLogger(__name__)


class ResourceSearchTool(BaseTool):
    """Tool for searching uploaded learning resources."""

    def __init__(self, rag_service: RAGRetrievalService | None = None) -> None:
        self._rag = rag_service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="resource_search",
            description="搜索已上传的学习资料（PDF、文档、代码等）中的相关内容。用于查找特定主题的学习资料。",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="要搜索的内容关键词或问题",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    type="number",
                    description="返回结果数量（默认 3）",
                    required=False,
                ),
            ],
        )

    async def execute(self, query: str, top_k: int = 3, **kwargs) -> ToolResult:
        if self._rag is None:
            return ToolResult(success=False, error="RAG service not available")

        try:
            results = await self._rag.search(query, top_k=top_k)
            context = await self._rag.assemble_context(results)

            if not context.sources:
                return ToolResult(success=True, output=f"未找到与「{query}」相关的资料")

            lines = [f"找到 {len(context.sources)} 条相关资料：\n"]
            for i, s in enumerate(context.sources, 1):
                lines.append(
                    f"{i}. {s.source_name} (来源: {s.source_type}, 相关性: {s.score:.2f})\n"
                    f"   {s.content[:300]}\n"
                )

            return ToolResult(success=True, output="\n".join(lines), data=context.sources)
        except Exception as exc:
            logger.exception("ResourceSearchTool failed")
            return ToolResult(success=False, error=str(exc))
