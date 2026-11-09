"""Fixed-size character chunking with overlap, snapped to word boundaries.

The baseline chunker. Week 4 replaces it with smarter variants and measures the difference.
"""
from dataclasses import dataclass

DEFAULT_SIZE = 1500  # characters, roughly 350 tokens
DEFAULT_OVERLAP = 500  # must be >= the longest evidence quote so no quote is split


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # character offset in the source text
    text: str


def _snap_forward(text: str, pos: int) -> int:
    """Move pos forward to the start of the next word (no-op if already at one)."""
    while pos < len(text) and not text[pos].isspace() and pos > 0 and not text[pos - 1].isspace():
        pos += 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _snap_back(text: str, pos: int) -> int:
    """Move pos back so a chunk ending at pos does not cut a word in half."""
    if pos >= len(text) or text[pos].isspace():
        return pos
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def chunk_text(text: str, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = text.strip()
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = _snap_back(text, min(start + size, len(text)))
        if end <= start:  # a single word longer than size: cut it hard
            end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(index=len(chunks), start=start, text=piece))
        if end >= len(text):
            break
        start = _snap_forward(text, max(end - overlap, start + 1))
    return chunks
