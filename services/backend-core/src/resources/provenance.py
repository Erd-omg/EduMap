"""Locate chunks within their source text — character-level provenance.

Why this exists
---------------
Chunks were produced as bare strings (``list[str]``), so by the time they
reached ChromaDB the link back to *where in the document* each one came from
was gone.  A citation could name the chunk but not point at the passage —
which makes a faithfulness score impossible to audit by hand: a reviewer
cannot check whether an answer is grounded if they cannot see the source span.

Design note — why this searches instead of changing the chunkers
---------------------------------------------------------------
The three chunking strategies (``fixed`` / ``recursive`` / ``semantic``) are
pluggable and each returns plain strings.  Threading offset tracking through
all of them would mean editing every chunker's internals, including the
recursive and semantic ones whose splitting logic is not offset-aware.

Instead the offsets are recovered by searching for each chunk in the source,
which:

* works uniformly for every strategy, present and future;
* leaves the splitting behaviour completely untouched (no regression risk in
  the chunkers themselves);
* is cheap — the search is linear in practice, and one pass per chunk.

The cost is that overlapping or normalised chunks need care.  Both cases are
handled explicitly below, and ``locate_chunks`` reports how many chunks it
could not place rather than silently returning wrong offsets — a wrong offset
is worse than an absent one, because it would highlight the wrong passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ChunkSpan:
    """One chunk together with its position in the source document."""

    text: str
    # Character offsets into the *normalised* source text.  ``None`` when the
    # chunk could not be located (see ``locate_chunks``).
    char_start: int | None
    char_end: int | None
    index: int

    @property
    def located(self) -> bool:
        return self.char_start is not None and self.char_end is not None

    @property
    def char_count(self) -> int:
        return len(self.text)


def _search_from(haystack: str, needle: str, start: int) -> int:
    """``haystack.find(needle, start)``, falling back to a whitespace-flexible search.

    Chunkers strip leading/trailing whitespace and the parser collapses runs of
    blank lines, so a chunk is often not a byte-exact substring of the source.
    Two attempts are made:

    1. an exact substring search from *start* — the common case, and fast;
    2. a regex search treating any whitespace run as equivalent to any other.

    The second pass is what handles ``re.sub(r"\\n{3,}", "\\n\\n", text)``, which
    the parser applies: the chunk carries the normalised form while offsets are
    into the normalised text, so without it *later* chunks would systematically
    be misplaced (a far worse failure than not locating them at all).

    Returns the index of the match, or -1.
    """
    if not needle:
        return -1

    found = haystack.find(needle, start)
    if found != -1:
        return found

    # Whitespace-flexible pass: build a pattern where every whitespace run in
    # the needle may match any non-empty whitespace run in the source.
    #
    # Note the filter runs on the *raw* split pieces, before escaping: splitting
    # on a capturing group keeps the separators, and escaping a whitespace piece
    # turns it into a non-empty string (e.g. ``"\\ "``) that a naive ``if p``
    # check would keep, producing ``alpha\s+\ \s+beta`` which matches nothing.
    tokens = [piece for piece in re.split(r"(\s+)", needle) if piece]
    if not tokens:
        return -1
    pattern = r"\s+".join(
        re.escape(piece) for piece in tokens if not piece.isspace()
    )
    if not pattern:
        return -1
    match = re.compile(pattern).search(haystack, start)
    return match.start() if match else -1


def _locate_span(source: str, chunk: str, cursor: int) -> tuple[int, int] | None:
    """Locate *chunk* in *source*, returning (start, end) or None.

    Handles the case that broke the naive implementation: **overlapping
    chunkers prepend the previous chunk's tail to the current chunk**.
    ``RecursiveChunker._apply_overlap`` does exactly ``prev[-64:] + chunk``, so
    the resulting string is not a contiguous substring of the source at all —
    on a real 13k-character Chinese document only 1 of 30 chunks was.

    Strategy:
      1. Try the whole chunk as a contiguous substring (the common case for
         non-overlapping strategies, and always the cheapest).
      2. Otherwise split the chunk at its largest section and locate each part
         independently, returning the span from the first part's start to the
         last part's end. The gap between them is the overlap seam, which is a
         legitimate part of the cited region.

    Returns None when no part can be located, so the caller can record an
    explicit "unknown" rather than a wrong offset.
    """
    whole = _search_from(source, chunk, cursor)
    if whole != -1:
        return whole, whole + len(chunk)

    # Split on blank lines: chunkers concatenate with the separator removed, so
    # the seam is where a paragraph break used to be.
    parts = [p for p in re.split(r"\n\s*\n", chunk) if p.strip()]
    if len(parts) < 2:
        # No internal structure to split on — try the first/last paragraph-ish
        # slice as a last resort before giving up.
        parts = [chunk]

    located: list[tuple[int, int]] = []
    search_from = cursor
    for part in parts:
        part = part.strip()
        if len(part) < 8:
            # Too short to identify reliably; skipping avoids matching a
            # coincidental fragment somewhere else in the document.
            continue
        found = _search_from(source, part, search_from)
        if found == -1 and cursor:
            found = _search_from(source, part, 0)
        if found == -1:
            continue
        located.append((found, found + len(part)))
        search_from = found + len(part)

    if not located:
        return None
    # Span from the earliest located part to the latest: everything between is
    # part of this chunk's region (including the overlap seam).
    return min(s for s, _ in located), max(e for _, e in located)


def locate_chunks(source: str, chunks: list[str]) -> list[ChunkSpan]:
    """Pair each chunk with its character span in *source*.

    Searches forward from the end of the previous match so that chunks are
    located in document order and an overlap between consecutive chunks does
    not cause the earlier one to be found twice.

    Args:
        source: the full text that was chunked.
        chunks: the chunk strings, in order.

    Returns:
        One :class:`ChunkSpan` per chunk.  A chunk that cannot be found gets
        ``char_start=None`` — the caller should treat that as "no span
        available" rather than guessing.
    """
    spans: list[ChunkSpan] = []
    cursor = 0

    for index, chunk in enumerate(chunks):
        stripped = chunk.strip()
        if not stripped:
            spans.append(ChunkSpan(text=chunk, char_start=None, char_end=None, index=index))
            continue

        located = _locate_span(source, stripped, cursor)
        if located is None and cursor > 0:
            # Retry from the beginning. A chunk that only appears before the
            # cursor means the sequence is not in document order (dedup,
            # re-sorting, or a chunker that re-emits text).
            #
            # Correctness beats ordering here: an offset that points at the
            # chunk's real text is useful even if it breaks monotonicity,
            # whereas refusing to locate it would lose information that is
            # genuinely available. Ordering is therefore a best-effort
            # property, and callers that need it should rely on their own
            # document-order iteration rather than on these offsets.
            located = _locate_span(source, stripped, 0)

        if located is None:
            spans.append(ChunkSpan(text=chunk, char_start=None, char_end=None, index=index))
            continue

        start, end = located
        spans.append(ChunkSpan(text=chunk, char_start=start, char_end=end, index=index))
        cursor = end

    return spans


def spans_to_metadata(spans: list[ChunkSpan]) -> list[dict]:
    """Convert spans into per-chunk metadata dicts for the vector store.

    Only located spans get offsets; an unlocated chunk stores ``None`` so the
    absence is explicit in the index rather than silently defaulting to 0
    (which would point every such citation at the start of the document).
    """
    return [
        {
            "chunk_index": span.index,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "char_count": span.char_count,
        }
        for span in spans
    ]


def page_for_offset(offset: int, page_starts: Sequence[int]) -> int | None:
    """Which 1-based page a character offset falls on, or None if unknown.

    ``page_starts[i]`` is the character offset at which page ``i + 1`` begins
    (see ``DocumentParser._extract_with_unstructured``). A page therefore runs
    from its own start up to the next page's start.

    Returns None when ``page_starts`` is empty — plain-text formats have no
    page concept, and reporting page 1 for everything would be a fabrication.
    """
    if not page_starts:
        return None
    if offset < page_starts[0]:
        # Before the first recorded boundary. The first element may have had no
        # page number, so attributing to page 1 is the best available answer.
        return 1
    page = 1
    for i, start in enumerate(page_starts):
        if offset >= start:
            page = i + 1
        else:
            break
    return page


def pages_for_span(
    char_start: int | None,
    char_end: int | None,
    page_starts: Sequence[int],
) -> int | None:
    """The page to attribute a chunk to, or None.

    Uses the chunk's **start** offset: a chunk that straddles a page break is
    attributed to the page it begins on, which matches how a reader would
    describe "where this passage is".
    """
    if char_start is None or not page_starts:
        return None
    return page_for_offset(char_start, page_starts)
