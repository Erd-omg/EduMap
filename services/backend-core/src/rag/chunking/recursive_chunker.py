"""Recursive chunker — splits text using a hierarchy of separators.

Inspired by LangChain's RecursiveCharacterTextSplitter.
Attempts to split text at the highest-level separator first, then
recursively splits sub-chunks using the next separator in the hierarchy.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """Split text recursively using a separator hierarchy.

    The default separator hierarchy is designed for general text:
        ["\n\n", "\n", "。", ".", " ", ""]

    For code, use code-specific separators:
        ["\n\n", "\n", "def ", "class ", "    ", "  ", " "]

    Args:
        separators: Ordered list of separators to try (most preferred first).
        chunk_size: Target character count per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
        self._chunk_size = chunk_size
        self._chunk_overlap = min(chunk_overlap, chunk_size // 2)

    def chunk(self, text: str) -> list[str]:
        """Split text recursively into chunks."""
        if not text or not text.strip():
            return []

        chunks = self._split_recursive(text, self._separators, self._chunk_size)
        # Apply overlap
        if self._chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        logger.debug(
            "Recursive chunker: %d chars → %d chunks (size=%d, overlap=%d)",
            len(text), len(chunks), self._chunk_size, self._chunk_overlap,
        )
        return chunks

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
        max_size: int,
    ) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not text or not text.strip():
            return []

        # If text fits in one chunk, return it
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        # Try current separator level
        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level splitting as last resort
            return self._split_by_chars(text, max_size)

        segments = text.split(sep)
        # Filter empty segments
        segments = [s.strip() for s in segments if s.strip()]

        # If separator didn't help (no splits), try next level
        if len(segments) <= 1:
            return self._split_recursive(text, remaining_seps, max_size)

        # Merge small segments and split large ones
        merged = self._merge_segments(segments, sep, max_size)

        # Further split any segment that's still too large
        result: list[str] = []
        for segment in merged:
            if len(segment) <= max_size:
                result.append(segment)
            else:
                result.extend(
                    self._split_recursive(segment, remaining_seps, max_size)
                )

        return result

    def _merge_segments(
        self,
        segments: list[str],
        separator: str,
        max_size: int,
    ) -> list[str]:
        """Merge consecutive segments that fit within max_size."""
        merged: list[str] = []
        current = ""

        for seg in segments:
            candidate = (current + separator + seg) if current else seg
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = seg  # Start new segment with current one

        if current:
            merged.append(current)

        return merged

    def _split_by_chars(self, text: str, max_size: int) -> list[str]:
        """Split text into chunks of exact character length (last resort)."""
        return [text[i:i + max_size].strip() for i in range(0, len(text), max_size) if text[i:i + max_size].strip()]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply character-level overlap between chunks."""
        if self._chunk_overlap <= 0:
            return chunks

        overlapped: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Take last N characters of previous chunk as overlap prefix
                prev_end = chunks[i - 1][-self._chunk_overlap:] if len(chunks[i - 1]) > self._chunk_overlap else chunks[i - 1]
                overlapped.append(prev_end + chunk)

        return overlapped

    @classmethod
    def for_code(cls, chunk_size: int = 512) -> RecursiveChunker:
        """Create a chunker optimized for code files."""
        return cls(
            separators=["\n\n", "\n", "def ", "class ", "    ", "  ", " "],
            chunk_size=chunk_size,
        )

    @classmethod
    def for_markdown(cls, chunk_size: int = 512) -> RecursiveChunker:
        """Create a chunker optimized for Markdown files."""
        return cls(
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
        )

    @classmethod
    def default(cls, chunk_size: int = 512) -> RecursiveChunker:
        """Create a default chunker."""
        return cls(chunk_size=chunk_size)
