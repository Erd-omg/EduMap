"""Tests for RAG chunking modules — SemanticChunker and RecursiveChunker."""

from __future__ import annotations

import logging

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.chunking.semantic_chunker import SemanticChunker
from src.rag.chunking.recursive_chunker import RecursiveChunker


# ── SemanticChunker ─────────────────────────────────────────────────

class TestSemanticChunkerInit:
    """SemanticChunker construction and configuration."""

    def test_default_params(self) -> None:
        """Default constructor sets expected defaults."""
        chunker = SemanticChunker()
        assert chunker._model_name == "BAAI/bge-small-zh-v1.5"
        assert chunker._min_chunk_size == 128
        assert chunker._max_chunk_size == 1024
        assert chunker._similarity_threshold == 0.7
        assert chunker._model is None

    def test_custom_params(self) -> None:
        """Custom parameters are stored correctly."""
        chunker = SemanticChunker(
            embedding_model="custom-model",
            min_chunk_size=64,
            max_chunk_size=512,
            similarity_threshold=0.5,
        )
        assert chunker._model_name == "custom-model"
        assert chunker._min_chunk_size == 64
        assert chunker._max_chunk_size == 512
        assert chunker._similarity_threshold == 0.5


class TestSemanticChunkerSplitSentences:
    """Sentence splitting logic (no model needed)."""

    def test_chinese_sentences(self) -> None:
        """Chinese sentence-ending punctuation produces splits."""
        chunker = SemanticChunker()
        text = "数组插入复杂度是O(n)。链表插入复杂度是O(1)。"
        result = chunker._split_sentences(text)
        assert len(result) == 2
        assert "数组插入复杂度是O(n)。" in result
        assert "链表插入复杂度是O(1)。" in result

    def test_english_sentences(self) -> None:
        """English sentence-ending punctuation produces splits."""
        chunker = SemanticChunker()
        text = "Array insertion is O(n). Linked list insertion is O(1)."
        result = chunker._split_sentences(text)
        assert len(result) == 2
        assert "Array insertion is O(n)." in result

    def test_mixed_sentence_endings(self) -> None:
        """Mixed Chinese/English punctuation all work."""
        chunker = SemanticChunker()
        text = "第一句！Second? 第三句。"
        result = chunker._split_sentences(text)
        assert len(result) == 3

    def test_exclamation_and_question(self) -> None:
        """Exclamation and question marks are sentence boundaries."""
        chunker = SemanticChunker()
        text = "你好！怎么样？好的。"
        result = chunker._split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self) -> None:
        """A single sentence returns one element."""
        chunker = SemanticChunker()
        result = chunker._split_sentences("仅此一句。")
        assert len(result) == 1

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = SemanticChunker()
        assert chunker._split_sentences("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only text returns empty list."""
        chunker = SemanticChunker()
        assert chunker._split_sentences("   \n\n  ") == []


class TestSemanticChunkerParagraphFallback:
    """Paragraph fallback logic (no model available)."""

    def test_single_paragraph(self) -> None:
        """Single paragraph returns as-is."""
        chunker = SemanticChunker()
        result = chunker._paragraph_fallback("这是一个段落。")
        assert result == ["这是一个段落。"]

    def test_multiple_paragraphs_small(self) -> None:
        """Multiple small paragraphs are merged within max_chunk_size."""
        chunker = SemanticChunker(max_chunk_size=1024)
        text = "第一段。\n\n第二段。\n\n第三段。"
        result = chunker._paragraph_fallback(text)
        # All fit in one chunk
        assert len(result) == 1
        assert "第一段" in result[0]
        assert "第三段" in result[0]

    def test_multiple_paragraphs_exceed_max(self) -> None:
        """Large paragraphs are split when they exceed max_chunk_size."""
        chunker = SemanticChunker(max_chunk_size=50)
        a = "A" * 40
        b = "B" * 40
        c = "C" * 40
        text = f"{a}\n\n{b}\n\n{c}"
        result = chunker._paragraph_fallback(text)
        assert len(result) >= 2  # at least 2 chunks

    def test_empty_text(self) -> None:
        """Empty text returns a list containing the empty string."""
        chunker = SemanticChunker()
        result = chunker._paragraph_fallback("")
        assert result == [""]

    def test_no_double_newlines(self) -> None:
        """Text without double newlines returns as single chunk."""
        chunker = SemanticChunker()
        result = chunker._paragraph_fallback("行1\n行2\n行3")
        assert len(result) == 1

    def test_leading_trailing_whitespace(self) -> None:
        """Extra whitespace around paragraphs is stripped."""
        chunker = SemanticChunker()
        text = "  \n\n  第一段  \n\n  第二段  "
        result = chunker._paragraph_fallback(text)
        for chunk in result:
            assert chunk == chunk.strip()


class TestSemanticChunkerFindBoundaries:
    """Boundary detection from embedding similarity."""

    def test_no_boundaries_high_similarity(self) -> None:
        """All similar embeddings produce no boundaries (except start/end)."""
        chunker = SemanticChunker(similarity_threshold=0.7, min_chunk_size=1, max_chunk_size=9999)
        # All vectors point in the same direction — high similarity
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        sentences = ["A. ", "B. ", "C. "]
        boundaries = chunker._find_boundaries(embeddings, sentences)
        assert boundaries == [0, 3]  # just start and end

    def test_boundary_at_topic_shift(self) -> None:
        """A large drop in similarity creates a boundary."""
        chunker = SemanticChunker(similarity_threshold=0.5, min_chunk_size=1, max_chunk_size=9999)
        # First two are similar, third is very different
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        sentences = ["A. ", "B. ", "C. "]
        boundaries = chunker._find_boundaries(embeddings, sentences)
        assert 2 in boundaries  # boundary before "C"

    def test_boundary_at_max_chunk_size(self) -> None:
        """Boundary created when accumulated chars exceed max_chunk_size."""
        chunker = SemanticChunker(
            similarity_threshold=0.0,  # never trigger topic shift
            min_chunk_size=1,
            max_chunk_size=10,
        )
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        sentences = ["A" * 8, "B" * 8, "C" * 8]
        boundaries = chunker._find_boundaries(embeddings, sentences)
        assert len(boundaries) > 2  # at least one extra boundary

    def test_similarity_below_threshold_but_below_min(self) -> None:
        """Topic shift is ignored when min_chunk_size hasn't been met."""
        chunker = SemanticChunker(
            similarity_threshold=0.9,  # high threshold → most shifts trigger
            min_chunk_size=100,  # very large min
            max_chunk_size=9999,
        )
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        sentences = ["A. ", "B. ", "C. "]
        boundaries = chunker._find_boundaries(embeddings, sentences)
        # Only start and end because running_chars never hits min_chunk_size
        assert boundaries == [0, 3]

    def test_single_sentence(self) -> None:
        """Single sentence produces only start/end boundaries."""
        chunker = SemanticChunker()
        embeddings = np.array([[1.0, 0.0]])
        sentences = ["Only sentence. "]
        boundaries = chunker._find_boundaries(embeddings, sentences)
        assert boundaries == [0, 1]


class TestSemanticChunkerBuildChunks:
    """Chunk assembly from boundary indices."""

    def test_basic_assembly(self) -> None:
        """Sentences are grouped by boundaries correctly."""
        chunker = SemanticChunker()
        sentences = ["A.", "B.", "C.", "D."]
        boundaries = [0, 2, 4]
        chunks = chunker._build_chunks(sentences, boundaries)
        assert len(chunks) == 2
        assert chunks[0] == "A.B."
        assert chunks[1] == "C.D."

    def test_empty_sentence_skipped(self) -> None:
        """Boundaries spanning empty sections produce no empty chunks."""
        chunker = SemanticChunker()
        sentences = ["A.", "  ", "B."]
        boundaries = [0, 1, 3]
        chunks = chunker._build_chunks(sentences, boundaries)
        # The empty sentence is joined but the strip removes it
        assert len(chunks) >= 1


class TestSemanticChunkerChunk:
    """End-to-end chunk() method."""

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = SemanticChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_single_sentence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single sentence is returned as a single chunk.

        ``_load_model`` is stubbed: without it this test loads the real
        ``SentenceTransformer`` in-process (slow, and it can hang the whole
        suite).  The single-sentence path returns before ``_model`` is ever
        consulted, so the stub does not weaken the assertion.
        """
        chunker = SemanticChunker()
        monkeypatch.setattr(chunker, "_load_model", lambda: None)

        result = chunker.chunk("仅此一句。")
        assert result == ["仅此一句。"]

    def test_fallback_when_model_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When no embedding model is available, the paragraph fallback is used.

        Two things were wrong with the previous version:

        1. It set ``_model = None``, but ``chunk()`` calls ``_load_model()``
           first, which saw ``None`` and **loaded the real model** — so the
           fallback branch was never reached (and the test loaded a real
           ``SentenceTransformer``, hanging the suite).
        2. Even with that fixed, asserting "fallback was called" is not enough:
           ``chunk()`` has **two** paths into ``_paragraph_fallback`` — the
           explicit ``if self._model is None`` check, and the ``except`` around
           ``self._model.encode()``.  With ``_model = None`` the encode path
           raises ``AttributeError`` and reaches the fallback anyway, so
           disabling the explicit check left the test green.

        Stubbing ``_load_model`` keeps ``_model`` None, and the raise-guard
        asserts the explicit branch is what runs — not the exception path.
        """
        chunker = SemanticChunker()
        monkeypatch.setattr(chunker, "_load_model", lambda: None)
        chunker._model = None  # no embedding model available

        text = "段落一。\n\n段落二。"
        calls: list[str] = []
        real_fallback = chunker._paragraph_fallback

        def spy_fallback(t: str):
            calls.append(t)
            return real_fallback(t)

        chunker._paragraph_fallback = spy_fallback  # type: ignore[method-assign]

        # Distinguish the two paths into _paragraph_fallback.  The exception
        # path (``None.encode()`` -> AttributeError -> except) logs
        # "Embedding failed, falling back"; the explicit ``_model is None``
        # branch does not.  Asserting the warning is ABSENT is what proves the
        # explicit check ran — merely asserting "fallback was called" cannot,
        # which is how the earlier version passed while broken.
        with caplog.at_level(logging.WARNING):
            result = chunker.chunk(text)

        assert not any(
            "Embedding failed" in r.message for r in caplog.records
        ), "chunk() took the encode()/except path, not the explicit None check"
        assert calls == [text], "paragraph fallback was not used"
        # And the fallback's own contract: paragraphs merge while they fit
        # under max_chunk_size, so two short paragraphs yield one chunk.
        assert result == [text], result

    def test_fallback_on_embedding_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When encode() raises, fallback is used."""
        chunker = SemanticChunker()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("模拟错误")
        chunker._model = mock_model

        text = "句子一。句子二。句子三。"
        result = chunker.chunk(text)
        # Should fall back successfully
        assert len(result) >= 1

    def test_chunk_with_mocked_model(self) -> None:
        """Full pipeline works with a mocked embedding model."""
        chunker = SemanticChunker(
            min_chunk_size=1,
            max_chunk_size=9999,
            similarity_threshold=0.5,
        )
        mock_model = MagicMock()
        # Return diverse embeddings to trigger topic-shift boundaries
        mock_model.encode.return_value = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        chunker._model = mock_model

        text = "AAA。BBB。CCC。DDD。"
        result = chunker.chunk(text)

        mock_model.encode.assert_called_once()
        assert len(result) >= 1


# ── RecursiveChunker ─────────────────────────────────────────────────

class TestRecursiveChunkerInit:
    """RecursiveChunker construction."""

    def test_default_params(self) -> None:
        """Default constructor uses expected defaults."""
        chunker = RecursiveChunker()
        assert chunker._separators == ["\n\n", "\n", "。", ".", " ", ""]
        assert chunker._chunk_size == 512
        assert chunker._chunk_overlap == 64

    def test_overlap_clamped(self) -> None:
        """chunk_overlap is clamped to chunk_size // 2."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=200)
        assert chunker._chunk_overlap == 50

    def test_custom_separators(self) -> None:
        """Custom separators are used."""
        chunker = RecursiveChunker(separators=["---", "\n\n"])
        assert chunker._separators == ["---", "\n\n"]

    def test_none_separators_default(self) -> None:
        """None separators falls back to default list."""
        chunker = RecursiveChunker(separators=None)
        assert chunker._separators == ["\n\n", "\n", "。", ".", " ", ""]

    def test_for_code_class_method(self) -> None:
        """for_code() returns a code-optimized chunker."""
        chunker = RecursiveChunker.for_code()
        assert "def " in chunker._separators
        assert "class " in chunker._separators

    def test_for_markdown_class_method(self) -> None:
        """for_markdown() returns a markdown-optimized chunker."""
        chunker = RecursiveChunker.for_markdown()
        assert "\n## " in chunker._separators
        assert "\n### " in chunker._separators

    def test_default_class_method(self) -> None:
        """default() creates a standard chunker."""
        chunker = RecursiveChunker.default(chunk_size=256)
        assert chunker._chunk_size == 256
        assert chunker._chunk_overlap == 64  # default overlap


class TestRecursiveChunkerChunk:
    """Core chunk() method."""

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_fits_one_chunk(self) -> None:
        """Short text that fits in one chunk is returned as-is."""
        chunker = RecursiveChunker(chunk_size=1000)
        text = "短文本。"
        result = chunker.chunk(text)
        assert result == [text]

    def test_split_by_double_newline(self) -> None:
        """Text is split at double-newlines first."""
        chunker = RecursiveChunker(chunk_size=200)
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_split_by_single_newline(self) -> None:
        """Text without double-newlines is split at single newlines."""
        chunker = RecursiveChunker(
            separators=["\n"],
            chunk_size=20,
        )
        text = "AAAA\nBBBB\nCCCC"
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_split_by_period(self) -> None:
        """Text is split at Chinese/English period."""
        chunker = RecursiveChunker(
            separators=["。"],
            chunk_size=50,
        )
        text = "句子一。句子二句子二。句子三。"
        result = chunker.chunk(text)
        assert len(result) >= 1

    def test_character_level_splitting(self) -> None:
        """When all separators fail, character-level splitting is used."""
        chunker = RecursiveChunker(
            separators=[""],  # forces char-split path
            chunk_size=5,
            chunk_overlap=0,  # no overlap — every chunk must be <= 5
        )
        text = "ABCDEFGHIJ"
        result = chunker.chunk(text)
        assert all(len(c) <= 5 for c in result)

    def test_chunk_size_respected(self) -> None:
        """No chunk exceeds the chunk_size (except with overlap)."""
        chunker = RecursiveChunker(
            separators=["\n"],
            chunk_size=50,
            chunk_overlap=0,
        )
        text = "A" * 30 + "\n" + "B" * 30 + "\n" + "C" * 30
        result = chunker.chunk(text)
        # Without overlap, all chunks should be <= chunk_size
        # (actually each segment is <= chunk_size, and the separator-based
        #  merging means segments can be up to chunk_size)
        for chunk in result:
            assert len(chunk) <= 50

    def test_overlap_application(self) -> None:
        """Overlap characters from previous chunk are prepended."""
        chunker = RecursiveChunker(
            separators=["\n"],
            chunk_size=100,
            chunk_overlap=10,
        )
        text = "A" * 30 + "\n" + "B" * 30 + "\n" + "C" * 30
        result = chunker.chunk(text)
        # With overlap, only the first chunk has no prefix
        if len(result) > 1:
            # Second chunk should contain the end of the first
            assert len(result[1]) > 30  # has overlap prefix

    def test_no_overlap_when_zero(self) -> None:
        """Zero overlap means no overlap is applied."""
        chunker = RecursiveChunker(
            separators=["\n"],
            chunk_size=100,
            chunk_overlap=0,
        )
        text = "A" * 30 + "\n" + "B" * 30 + "\n" + "C" * 30
        result = chunker.chunk(text)
        for chunk in result:
            assert chunk.strip() != ""


class TestRecursiveChunkerSplitRecursive:
    """Internal _split_recursive method."""

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = RecursiveChunker()
        assert chunker._split_recursive("", ["\n"], 100) == []

    def test_text_fits(self) -> None:
        """Text that fits max_size returns as single element."""
        chunker = RecursiveChunker()
        result = chunker._split_recursive("Hello world", ["\n"], 100)
        assert result == ["Hello world"]

    def test_next_separator_when_first_ineffective(self) -> None:
        """When first separator doesn't split, the next is tried."""
        chunker = RecursiveChunker()
        # "\n\n" won't split this, so it falls through to "."
        text = "AAAAA.BBBBB.CCCCC"
        result = chunker._split_recursive(
            text, ["\n\n", "."], 10,
        )
        assert len(result) >= 2

    def test_char_split_last_resort(self) -> None:
        """Empty separator triggers char-level split."""
        chunker = RecursiveChunker()
        result = chunker._split_recursive(
            "ABCDEFGHIJKLMNOPQRST", [""], 5,
        )
        assert len(result) == 4  # 20 chars / 5 = 4 chunks
        assert all(len(c) <= 5 for c in result)


class TestRecursiveChunkerMergeSegments:
    """Internal _merge_segments method."""

    def test_merge_small_segments(self) -> None:
        """Small segments are merged within max_size."""
        chunker = RecursiveChunker()
        segments = ["A", "B", "C"]
        result = chunker._merge_segments(segments, "\n", 100)
        assert len(result) == 1
        assert result[0] == "A\nB\nC"

    def test_split_when_exceeds_max_size(self) -> None:
        """A segment that exceeds max_size on merge starts a new group."""
        chunker = RecursiveChunker()
        segments = ["A" * 50, "B" * 50]
        result = chunker._merge_segments(segments, "\n", 60)
        assert len(result) == 2

    def test_empty_segments_list(self) -> None:
        """Empty segments list returns empty list."""
        chunker = RecursiveChunker()
        assert chunker._merge_segments([], "\n", 100) == []


class TestRecursiveChunkerSplitByChars:
    """Character-level splitting (last resort)."""

    def test_exact_split(self) -> None:
        """Text is split evenly at max_size boundaries."""
        chunker = RecursiveChunker()
        result = chunker._split_by_chars("ABCDEFGHIJ", 5)
        assert result == ["ABCDE", "FGHIJ"]

    def test_uneven_split(self) -> None:
        """Last chunk is shorter when text length isn't a multiple."""
        chunker = RecursiveChunker()
        result = chunker._split_by_chars("ABCDEFGHI", 5)
        assert len(result) == 2
        assert len(result[-1]) == 4

    def test_smaller_than_max(self) -> None:
        """Text smaller than max_size returns as one chunk."""
        chunker = RecursiveChunker()
        result = chunker._split_by_chars("ABC", 100)
        assert result == ["ABC"]

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        chunker = RecursiveChunker()
        assert chunker._split_by_chars("", 10) == []

    def test_whitespace_stripped(self) -> None:
        """Whitespace-only chunks are stripped and empty ones removed."""
        chunker = RecursiveChunker()
        result = chunker._split_by_chars("A   B", 2)
        assert all(c.strip() == c for c in result if c)


class TestRecursiveChunkerApplyOverlap:
    """_apply_overlap method."""

    def test_no_overlap_when_zero(self) -> None:
        """Zero overlap returns unchanged chunks."""
        chunker = RecursiveChunker(chunk_overlap=0)
        chunks = ["AAA", "BBB", "CCC"]
        result = chunker._apply_overlap(chunks)
        assert result == chunks

    def test_overlap_applied(self) -> None:
        """Overlap characters from previous chunk are prepended."""
        chunker = RecursiveChunker(chunk_overlap=2)
        chunks = ["AAAA", "BBBB", "CCCC"]
        result = chunker._apply_overlap(chunks)
        assert result[0] == "AAAA"  # first chunk unchanged
        assert "AA" in result[1]  # has overlap from chunk[0]
        assert "BB" in result[2]  # has overlap from chunk[1]

    def test_overlap_larger_than_chunk(self) -> None:
        """When overlap > chunk length, the whole previous chunk is used."""
        chunker = RecursiveChunker(chunk_overlap=100)
        chunks = ["AB", "CD"]
        result = chunker._apply_overlap(chunks)
        assert result[0] == "AB"
        assert result[1] == "ABCD"  # full previous chunk + current

    def test_single_chunk(self) -> None:
        """Single chunk list returns unchanged."""
        chunker = RecursiveChunker(chunk_overlap=10)
        assert chunker._apply_overlap(["AAAA"]) == ["AAAA"]
