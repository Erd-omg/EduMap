"""Semantic chunker — splits text at semantic boundaries using embedding similarity.

Detects natural breakpoints in text by measuring the cosine similarity between
consecutive sentence embeddings. A drop in similarity indicates a topic shift,
which becomes a chunk boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split text into chunks based on semantic similarity between sentences.

    Uses sentence-transformers to embed sentences, then detects boundaries
    where consecutive sentence similarity falls below a threshold.

    Args:
        embedding_model: Name of a sentence-transformers model.
        min_chunk_size: Minimum characters per chunk (prevents tiny chunks).
        max_chunk_size: Approximate maximum characters per chunk.
        similarity_threshold: Boundary threshold (0-1). Lower = fewer boundaries,
            larger chunks. Default 0.7 means a boundary is created when similarity
            between consecutive sentences drops below 70%.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
        similarity_threshold: float = 0.7,
    ) -> None:
        self._model_name = embedding_model
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._similarity_threshold = similarity_threshold
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not available, falling back to "
                "simple sentence splitting for semantic chunker"
            )
            self._model = None

    def chunk(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks."""
        if not text or not text.strip():
            return []

        self._load_model()

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # If no embedding model, use simple heuristic (paragraph-based)
        if self._model is None:
            return self._paragraph_fallback(text)

        # Compute embeddings
        try:
            embeddings = self._model.encode(sentences, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Embedding failed, falling back: %s", exc)
            return self._paragraph_fallback(text)

        # Find boundaries where similarity drops below threshold
        boundaries = self._find_boundaries(embeddings, sentences)

        # Build chunks from boundaries
        chunks = self._build_chunks(sentences, boundaries)

        logger.debug(
            "Semantic chunker: %d sentences → %d chunks",
            len(sentences), len(chunks),
        )
        return chunks

    def _find_boundaries(
        self,
        embeddings: np.ndarray,
        sentences: list[str],
    ) -> list[int]:
        """Detect semantic boundaries using cosine similarity drops."""
        boundaries: list[int] = [0]
        running_chars = 0

        for i in range(len(embeddings) - 1):
            # Cosine similarity between consecutive sentences
            sim = float(np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            ))

            running_chars += len(sentences[i])

            # Create boundary if: similarity below threshold OR max_chunk_size exceeded
            exceeds_max = running_chars > self._max_chunk_size and i + 1 - boundaries[-1] > 1
            topic_shift = sim < self._similarity_threshold
            meets_min = running_chars >= self._min_chunk_size

            if (topic_shift and meets_min) or exceeds_max:
                boundaries.append(i + 1)
                running_chars = 0

        # End boundary
        boundaries.append(len(sentences))
        return boundaries

    def _build_chunks(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """Assemble chunks from sentence groups defined by boundary indices."""
        chunks: list[str] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_text = "".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Handles Chinese (。！？) and English (. ! ?) sentence boundaries.
        """
        import re
        # Split on sentence-ending punctuation while keeping the delimiter
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        # Filter out empty strings
        return [p for p in parts if p.strip()]

    def _paragraph_fallback(self, text: str) -> list[str]:
        """Fallback: split by double newlines (paragraphs), merge small ones."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > self._max_chunk_size and current:
                chunks.append(current)
                current = para
            else:
                current = (current + "\n\n" + para) if current else para
        if current:
            chunks.append(current)

        return chunks if chunks else [text]
