"""Tests for character-level chunk provenance.

The property that matters: a reported span must point at the **actual** text of
its chunk. A wrong offset is worse than a missing one — it would highlight the
wrong passage in the UI and let a reviewer "verify" a citation against text
that was never the source. So the tests below assert round-trip correctness
(``source[start:end] == chunk``), not merely that some numbers came back.
"""

from __future__ import annotations

import pytest

from src.resources.provenance import (
    ChunkSpan,
    locate_chunks,
    spans_to_metadata,
)


class TestBasicLocating:
    def test_single_chunk_round_trips(self) -> None:
        source = "hello world"
        chunks = ["hello world"]
        spans = locate_chunks(source, chunks)
        assert spans[0].char_start == 0
        assert spans[0].char_end == 11
        assert source[spans[0].char_start : spans[0].char_end] == chunks[0]

    def test_consecutive_chunks_round_trip(self) -> None:
        source = "aaabbbccc"
        chunks = ["aaa", "bbb", "ccc"]
        spans = locate_chunks(source, chunks)
        for span, chunk in zip(spans, chunks):
            assert source[span.char_start : span.char_end] == chunk

    def test_offsets_are_in_document_order(self) -> None:
        source = "one two three four"
        chunks = ["one", "two", "three", "four"]
        spans = locate_chunks(source, chunks)
        starts = [s.char_start for s in spans]
        assert starts == sorted(starts)

    def test_empty_source(self) -> None:
        spans = locate_chunks("", [])
        assert spans == []

    def test_empty_chunk_is_unlocated_not_misplaced(self) -> None:
        spans = locate_chunks("hello", ["", "hello"])
        assert spans[0].char_start is None
        assert not spans[0].located
        # the real chunk is still placed correctly
        assert spans[1].char_start == 0


class TestOverlapHandling:
    def test_overlapping_chunks_are_not_located_twice(self) -> None:
        """With overlap, a naive search would re-find the earlier text.

        Forward-only search from the previous match's end is what prevents the
        second chunk from being pinned to the first occurrence.
        """
        source = "ABCDEFGHIJ"
        # second chunk overlaps the first
        chunks = ["ABCDE", "DEFGHIJ"]
        spans = locate_chunks(source, chunks)

        assert source[spans[0].char_start : spans[0].char_end] == "ABCDE"
        assert source[spans[1].char_start : spans[1].char_end] == "DEFGHIJ"
        assert spans[1].char_start == 3

    def test_repeated_text_advances(self) -> None:
        """Identical consecutive chunks must map to distinct spans."""
        source = "xyz xyz xyz"
        chunks = ["xyz", "xyz", "xyz"]
        spans = locate_chunks(source, chunks)
        starts = [s.char_start for s in spans]
        assert starts == [0, 4, 8]

    def test_fully_overlapping_chunks_either_locate_or_report_none(self) -> None:
        """Overlap means there may be no *distinct* span for the later chunk.

        The contract under test is that any offset reported must be **true**
        (``source[start:end]`` really is the chunk). Whether the second chunk
        of a full overlap resolves to a shared span or to ``None`` is a
        judgement call; what must never happen is a span pointing at text that
        is not the chunk.
        """
        source = "aaaa"
        chunks = ["aaa", "aaa"]
        spans = locate_chunks(source, chunks)

        assert spans[0].char_start == 0
        for span in spans:
            if span.located:
                assert source[span.char_start : span.char_end] == "aaa"

    def test_overlap_does_not_duplicate_the_first_span(self) -> None:
        """Non-identical overlapping chunks must get distinct, correct spans."""
        source = "ABCDEFGHIJ"
        spans = locate_chunks(source, ["ABCDE", "DEFGHIJ"])
        assert spans[0].char_start == 0
        assert spans[1].char_start == 3
        for span in spans:
            assert span.located


class TestWhitespaceTolerance:
    def test_chunk_that_was_stripped_is_located(self) -> None:
        source = "  hello world  "
        chunks = ["hello world"]  # chunker stripped it
        spans = locate_chunks(source, chunks)
        assert spans[0].char_start == 2
        assert source[spans[0].char_start : spans[0].char_end] == "hello world"

    def test_collapsed_blank_lines_are_tolerated(self) -> None:
        """The parser collapses 3+ newlines to 2; chunks reflect the normalised form."""
        source = "para one\n\n\n\npara two"
        chunks = ["para one\n\npara two"]
        spans = locate_chunks(source, chunks)
        assert spans[0].located, "a whitespace-differing chunk should still be found"
        # the located start must be correct even if the end lands short/long
        assert spans[0].char_start == 0

    def test_newline_vs_space_variation(self) -> None:
        source = "alpha\nbeta"
        chunks = ["alpha beta"]
        spans = locate_chunks(source, chunks)
        assert spans[0].char_start == 0

    def test_trailing_whitespace_in_chunk(self) -> None:
        source = "abc def"
        chunks = ["abc def   "]
        spans = locate_chunks(source, chunks)
        assert spans[0].located
        assert spans[0].char_start == 0


class TestUnlocatableChunks:
    def test_absent_text_is_reported_not_guessed(self) -> None:
        """A chunk not in the source must not be given a plausible-looking span."""
        source = "hello world"
        chunks = ["goodbye moon"]
        spans = locate_chunks(source, chunks)
        assert spans[0].char_start is None
        assert spans[0].char_end is None
        assert not spans[0].located

    def test_partial_dataset_still_places_the_rest(self) -> None:
        source = "one two three"
        chunks = ["one", "NOT PRESENT", "three"]
        spans = locate_chunks(source, chunks)
        assert spans[0].located
        assert not spans[1].located
        # "three" is found by the global fallback even though the cursor had
        # advanced past it
        assert spans[2].located
        assert source[spans[2].char_start : spans[2].char_end] == "three"

    def test_out_of_order_chunk_found_by_fallback(self) -> None:
        source = "alpha beta gamma"
        chunks = ["gamma", "alpha"]
        spans = locate_chunks(source, chunks)
        assert spans[0].located and spans[1].located
        assert source[spans[0].char_start : spans[0].char_end] == "gamma"
        assert source[spans[1].char_start : spans[1].char_end] == "alpha"


class TestMetadata:
    def test_metadata_shape(self) -> None:
        spans = locate_chunks("hello world", ["hello", "world"])
        meta = spans_to_metadata(spans)
        assert len(meta) == 2
        assert set(meta[0]) == {"chunk_index", "char_start", "char_end", "char_count"}
        assert meta[0]["chunk_index"] == 0
        assert meta[1]["chunk_index"] == 1

    def test_unlocated_chunk_stores_none_not_zero(self) -> None:
        """Defaulting to 0 would silently point every citation at the document start."""
        spans = locate_chunks("hello", ["missing"])
        meta = spans_to_metadata(spans)
        assert meta[0]["char_start"] is None
        assert meta[0]["char_end"] is None

    def test_char_count_is_the_chunk_length(self) -> None:
        spans = locate_chunks("hello world", ["hello"])
        meta = spans_to_metadata(spans)
        assert meta[0]["char_count"] == 5


class TestRealisticDocument:
    """A document resembling parser output: normalised newlines, many chunks."""

    def test_overlapping_chunks_are_located(self) -> None:
        """Overlap-prepended chunks must still get a span.

        This is the case that the first implementation failed on badly: the
        recursive chunker builds each chunk as ``prev_tail + current``, so the
        chunk is *not* a contiguous substring of the source. On a real
        13k-character Chinese document only 1 of 30 chunks satisfied the naive
        containment test; after splitting and locating the parts, 25 of 30 do
        (the other 5 are ASCII-art tables whose characters are interleaved by
        the fixed-width split and have no contiguous span — correctly None).
        """
        paragraphs = [f"第 {i} 段：这是关于数据结构与算法的说明文字。" for i in range(40)]
        source = "\n\n".join(paragraphs)
        # simulate recursive-style chunking with overlap prepended
        size, overlap = 120, 20
        base: list[str] = []
        start = 0
        while start < len(source):
            piece = source[start : start + size].strip()
            if piece:
                base.append(piece)
            start += size - overlap
        chunks = [
            base[0],
            *[base[i - 1][-overlap:] + base[i] for i in range(1, len(base))],
        ]

        spans = locate_chunks(source, chunks)

        located = [s for s in spans if s.located]
        assert len(located) == len(chunks), (
            f"{len(chunks) - len(located)} of {len(chunks)} chunks unlocated"
        )

    def test_every_reported_span_is_within_bounds(self) -> None:
        """A span must be a usable range — never inverted or past the end.

        The invariant worth asserting is *usability*, not that the span text
        equals the chunk: an overlap-prepended chunk is wider than any single
        contiguous source range, so the span covers the whole cited region
        (including the seam) rather than reproducing the chunk verbatim.
        """
        paragraphs = [f"第 {i} 段：这是关于数据结构与算法的说明文字。" for i in range(40)]
        source = "\n\n".join(paragraphs)
        size, overlap = 120, 20
        base: list[str] = []
        start = 0
        while start < len(source):
            piece = source[start : start + size].strip()
            if piece:
                base.append(piece)
            start += size - overlap
        chunks = [
            base[0],
            *[base[i - 1][-overlap:] + base[i] for i in range(1, len(base))],
        ]

        for span in locate_chunks(source, chunks):
            if span.located:
                assert 0 <= span.char_start < span.char_end <= len(source)
                # the span must be non-trivially wide
                assert span.char_end - span.char_start >= 20

    def test_span_covers_the_chunks_own_text(self) -> None:
        """The span must include the chunk's distinctive portion.

        For an overlap-prepended chunk the span starts at the previous chunk's
        tail, so ``source[start:end]`` is longer than the chunk; what must hold
        is that the chunk's *own* trailing content appears inside the span.
        """
        source = "A" * 200 + "\n\n" + "B" * 200
        chunks = ["A" * 30 + "A" * 200 + "\n\n" + "B" * 200]

        span = locate_chunks(source, chunks)[0]
        assert span.located
        region = source[span.char_start : span.char_end]
        assert "B" * 50 in region
