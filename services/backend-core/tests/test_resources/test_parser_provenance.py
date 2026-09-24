"""Tests that chunk provenance actually reaches the vector index.

``test_provenance.py`` covers the locator in isolation. This file covers the
wiring: ``DocumentParser._embed_and_index`` must write the offsets into each
chunk's metadata, because a correct locator that is never called produces the
same user-visible result as no locator at all.

The failure mode worth guarding: a span silently defaulting to 0. That would
make every citation point at the document's first character, which looks like
working provenance and is worse than none.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.resources.parser import DocumentParser


def _capturing_collection() -> tuple[MagicMock, dict]:
    """A ChromaDB collection mock that records what was upserted."""
    captured: dict = {}

    def _upsert(ids, embeddings, documents, metadatas):
        captured.setdefault("ids", []).extend(ids)
        captured.setdefault("documents", []).extend(documents)
        captured.setdefault("metadatas", []).extend(metadatas)

    collection = MagicMock()
    collection.upsert = MagicMock(side_effect=_upsert)
    return collection, captured


def _parser_with_stub_model() -> DocumentParser:
    """Parser whose embedding model is stubbed to avoid loading 100MB of weights."""
    parser = DocumentParser()
    model = MagicMock()
    # encode(list) -> array-like of one 3-dim vector per text
    model.encode = MagicMock(
        side_effect=lambda texts, **_: [MagicMock(tolist=lambda: [0.1, 0.2, 0.3]) for _ in texts]
    )
    parser._model = model
    return parser


class TestEmbedAndIndexWritesSpans:
    @pytest.mark.asyncio
    async def test_char_offsets_are_written_into_metadata(self) -> None:
        from src.resources.provenance import locate_chunks

        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        chunks = ["alpha", "beta", "gamma"]
        full_text = "alpha beta gamma"

        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            await parser._embed_and_index(
                chunks=chunks,
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
                spans=locate_chunks(full_text, chunks),
            )

        metas = captured["metadatas"]
        assert len(metas) == 3
        for meta, chunk in zip(metas, chunks):
            assert "char_start" in meta and "char_end" in meta
            assert full_text[meta["char_start"] : meta["char_end"]] == chunk

    @pytest.mark.asyncio
    async def test_offsets_are_ordered(self) -> None:
        from src.resources.provenance import locate_chunks

        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        full_text = "one two three four"
        chunks = ["one", "two", "three", "four"]
        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            await parser._embed_and_index(
                chunks=chunks,
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
                spans=locate_chunks(full_text, chunks),
            )

        starts = [m["char_start"] for m in captured["metadatas"]]
        assert starts == sorted(starts)

    @pytest.mark.asyncio
    async def test_unlocatable_chunk_stores_none_not_zero(self) -> None:
        """A missing span must be distinguishable from "starts at offset 0"."""
        from src.resources.provenance import locate_chunks

        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        chunks = ["not in the document at all"]
        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            await parser._embed_and_index(
                chunks=chunks,
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
                spans=locate_chunks("completely different content", chunks),
            )

        meta = captured["metadatas"][0]
        assert meta["char_start"] is None
        assert meta["char_end"] is None

    @pytest.mark.asyncio
    async def test_no_spans_omits_span_fields(self) -> None:
        """Older callers that pass no spans get no (wrong) offsets."""
        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            await parser._embed_and_index(
                chunks=["alpha"],
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
            )

        meta = captured["metadatas"][0]
        assert "char_start" not in meta
        assert "char_end" not in meta

    @pytest.mark.asyncio
    async def test_existing_metadata_is_preserved(self) -> None:
        """Adding spans must not drop resource_id / kp fields."""
        from src.resources.provenance import locate_chunks

        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            await parser._embed_and_index(
                chunks=["alpha"],
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
                kp_id="kp1",
                kp_name="数组",
                spans=locate_chunks("alpha", ["alpha"]),
            )

        meta = captured["metadatas"][0]
        assert meta["resource_id"] == "res1"
        assert meta["chunk_index"] == 0
        assert meta["kp_id"] == "kp1"
        assert meta["kp_name"] == "数组"
        assert meta["text_preview"] == "alpha"


    @pytest.mark.asyncio
    async def test_index_count_matches_chunks_upserted(self) -> None:
        """A partial final batch must not be counted as full.

        The batch loop computed its count as ``batch_end - batch_start`` with
        ``batch_end`` unclamped, so a 19-chunk document was reported as
        "indexed: 100" (one full batch_size). Slice truncation hid the bug from
        the upsert itself — only the number was wrong, which is the kind of
        error that makes a report untrustworthy without ever failing.
        """
        from src.resources.provenance import locate_chunks

        parser = _parser_with_stub_model()
        collection, captured = _capturing_collection()

        chunks = [f"chunk{i}" for i in range(19)]  # < batch_size (100)
        source = " ".join(chunks)

        with patch.object(parser, "_get_or_create_resource_collection", return_value=collection):
            indexed = await parser._embed_and_index(
                chunks=chunks,
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=MagicMock(),
                spans=locate_chunks(source, chunks),
            )

        assert indexed == len(chunks), (
            f"reported {indexed} indexed but upserted {len(chunks)}"
        )
        assert len(captured["metadatas"]) == len(chunks)


class TestParseResultCarriesSpans:
    """The chunk previews returned to the API must carry offsets too.

    The previews are what ``/chunks`` serves, so without offsets there the
    frontend has nothing to highlight with — even though the vector-store
    metadata would have them.
    """

    @pytest.mark.asyncio
    async def test_in_memory_chunks_include_offsets(self, tmp_path) -> None:
        from unittest.mock import patch as _patch

        doc = tmp_path / "doc.txt"
        doc.write_text("alpha beta gamma delta", encoding="utf-8")

        parser = DocumentParser()
        with _patch.object(parser, "_extract_text", return_value="alpha beta gamma delta"):
            result = await parser.parse_and_index(
                file_path=str(doc),
                resource_id="res1",
                resource_name="doc.txt",
                vector_index=None,
            )

        chunks = result["in_memory_chunks"]
        assert chunks, "expected chunk previews"
        for entry in chunks:
            assert "char_start" in entry
            assert "char_end" in entry
            if entry["char_start"] is not None:
                assert (
                    "alpha beta gamma delta"[entry["char_start"] : entry["char_end"]]
                    == entry["text"]
                )

    @pytest.mark.asyncio
    async def test_unlocated_preview_uses_none(self) -> None:
        """A preview whose chunk cannot be found must report None, not 0."""
        parser = DocumentParser()
        with patch.object(parser, "_chunk_text", return_value=["text not in source"]):
            with patch.object(parser, "_extract_text", return_value="something else entirely"):
                import tempfile, os

                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
                    fh.write("x")
                    path = fh.name
                try:
                    result = await parser.parse_and_index(
                        file_path=path,
                        resource_id="r1",
                        resource_name="f.txt",
                        vector_index=None,
                    )
                finally:
                    os.unlink(path)

        entry = result["in_memory_chunks"][0]
        assert entry["char_start"] is None
        assert entry["char_end"] is None


class TestMentorSourceSpanPassthrough:
    """The span must survive the trip from chunk metadata to the API model."""

    def test_span_extracted_when_present(self) -> None:
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": 10, "char_end": 20}) == {
            "char_start": 10,
            "char_end": 20,
        }

    def test_missing_metadata_yields_no_keys(self) -> None:
        from src.agents.mentor.agent import _source_span

        assert _source_span(None) == {}
        assert _source_span({}) == {}

    def test_partial_span_is_rejected_entirely(self) -> None:
        """One-sided spans are dropped, not passed through.

        A span with only ``char_start`` cannot be highlighted, and emitting it
        would make the frontend's "is the span known?" test disagree with what
        the values imply (``start`` known, ``end`` missing). Absent is the
        honest answer.
        """
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": 5}) == {}
        assert _source_span({"char_end": 5}) == {}

    def test_int_valued_float_offsets_are_accepted(self) -> None:
        """Offsets travel through JSON/numpy, so 1200 may arrive as 1200.0.

        ``isinstance(v, int)`` rejects that, which previously dropped the span
        silently — leaving half a span that no consumer could render.
        """
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": 1200.0, "char_end": 1800.0}) == {
            "char_start": 1200,
            "char_end": 1800,
        }

    def test_fractional_offsets_are_rejected(self) -> None:
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": 1.5, "char_end": 9}) == {}

    def test_inverted_span_is_rejected(self) -> None:
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": 100, "char_end": 50}) == {}
        assert _source_span({"char_start": 10, "char_end": 10}) == {}

    def test_bool_is_not_an_offset(self) -> None:
        """bool is an int subclass; True must not become offset 1."""
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": True, "char_end": 5}) == {}

    def test_non_integer_offsets_are_ignored(self) -> None:
        """A malformed value must not become a bogus highlight."""
        from src.agents.mentor.agent import _source_span

        assert _source_span({"char_start": "5", "char_end": None}) == {}

    def test_mentor_source_defaults_to_none(self) -> None:
        from src.rag.models import MentorSource

        source = MentorSource(id="s", name="n")
        assert source.char_start is None
        assert source.char_end is None

    def test_mentor_source_accepts_span(self) -> None:
        from src.rag.models import MentorSource

        source = MentorSource(id="s", name="n", char_start=1, char_end=9)
        assert source.char_start == 1
        assert source.char_end == 9
