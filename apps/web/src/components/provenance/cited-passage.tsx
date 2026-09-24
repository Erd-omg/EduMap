'use client';

import { useMemo } from 'react';

/**
 * Splits `text` into the parts before / inside / after a character span, so
 * the cited passage can be highlighted.
 *
 * Kept as a pure function (rather than inline JSX logic) because the edge
 * cases here are the whole point and need to be testable directly:
 *
 * - **`null`/`undefined` span** → no highlight. This is deliberate and is the
 *   most important case: an unknown span must never be coerced to 0, which
 *   would highlight the first characters of the document and *look* like it
 *   worked.
 * - **Out-of-range or inverted span** → no highlight, rather than clamping.
 *   A clamped highlight would be a plausible-looking but wrong passage, which
 *   is worse than clearly showing nothing.
 * - **Empty or whitespace-only span** → no highlight; there is nothing to
 *   point at.
 */
export interface HighlightParts {
  before: string;
  match: string;
  after: string;
  /** False when no span was applied — callers can use this to show a hint. */
  highlighted: boolean;
}

export function splitBySpan(
  text: string,
  charStart?: number | null,
  charEnd?: number | null,
): HighlightParts {
  const noHighlight: HighlightParts = { before: text, match: '', after: '', highlighted: false };

  if (typeof charStart !== 'number' || typeof charEnd !== 'number') return noHighlight;
  if (charStart < 0 || charEnd <= charStart || charEnd > text.length) return noHighlight;

  const match = text.slice(charStart, charEnd);
  if (!match.trim()) return noHighlight;

  return {
    before: text.slice(0, charStart),
    match,
    after: text.slice(charEnd),
    highlighted: true,
  };
}

interface CitedPassageProps {
  /** The full chunk text (the passage sits inside it). */
  text: string;
  charStart?: number | null;
  charEnd?: number | null;
  className?: string;
}

/**
 * Renders a chunk with the cited span highlighted.
 *
 * `charStart`/`charEnd` are offsets into the *source document*, while `text`
 * here is the chunk itself. The offsets are therefore translated by the chunk's
 * own start when the caller has it; callers that pass a chunk-relative span
 * (the usual case, since `text_preview` is the chunk) can pass it directly.
 */
export function CitedPassage({ text, charStart, charEnd, className }: CitedPassageProps) {
  const parts = useMemo(
    () => splitBySpan(text, charStart, charEnd),
    [text, charStart, charEnd],
  );

  if (!parts.highlighted) {
    return <span className={className}>{text}</span>;
  }

  return (
    <span className={className}>
      {parts.before}
      <mark
        data-testid="cited-passage"
        className="rounded bg-yellow-200/70 px-0.5 text-text-primary"
      >
        {parts.match}
      </mark>
      {parts.after}
    </span>
  );
}
