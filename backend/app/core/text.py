"""Passage cleaning and sentence-aware chunking."""

import math
import re

_WHITESPACE = re.compile(r"\s+")
# Split after . ! ? when followed by whitespace and an uppercase letter, digit or bracket.
# Lowercase continuations ("e.g. the", "et al. reported") are therefore not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
# A bare journal reference with no abstract, e.g. "1. Am J Med. 1971 Jul;51(1):97-108. doi: ..."
_CITATION_ONLY = re.compile(r"^1\. .{0,150}\bdoi: \S+(?: Epub [^.]+\.)?$")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def is_usable(text: str | None, min_chars: int) -> bool:
    """False for missing abstracts (literal "nan"), empty text, stubs like "1." and
    citation-only entries."""
    if text is None:
        return False
    stripped = normalize_whitespace(text)
    return (
        stripped.lower() != "nan"
        and len(stripped) >= min_chars
        and not _CITATION_ONLY.match(stripped)
    )


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def _hard_split(sentence: str, limit: int) -> list[str]:
    """Split a sentence longer than `limit` on word boundaries."""
    parts: list[str] = []
    current = ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > limit:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, max_whole_chars: int, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Return the passage whole if short, else sentence-aligned chunks with overlap.

    Each chunk after the first starts with the trailing sentences of the previous
    chunk (up to ~`overlap_chars`). Chunks are balanced so that, including overlap,
    they are about equal and at most ~`chunk_chars`: with chunk_chars=1200 and
    overlap_chars=200, a 2,100-char passage becomes two ~1,150-char chunks rather
    than 1,200 + 1,100.
    """
    if len(text) <= max_whole_chars:
        return [text]

    new_per_chunk = chunk_chars - overlap_chars
    n_chunks = math.ceil((len(text) - overlap_chars) / new_per_chunk)
    limit = min(chunk_chars, overlap_chars + math.ceil((len(text) - overlap_chars) / n_chunks))
    sentences = [piece for s in split_sentences(text) for piece in _hard_split(s, chunk_chars)]

    chunks: list[str] = []
    current: list[str] = []
    new_in_current = 0  # sentences in `current` that are not overlap

    for sentence in sentences:
        size = sum(len(s) + 1 for s in current) + len(sentence)
        if new_in_current and size > limit:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars, chunk_chars)
            if sum(len(s) + 1 for s in current) + len(sentence) > chunk_chars:
                current = []
            new_in_current = 0
        current.append(sentence)
        new_in_current += 1

    if new_in_current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: list[str], overlap_chars: int, chunk_chars: int) -> list[str]:
    """Trailing sentences totalling at most `overlap_chars`.

    If even the last sentence is longer than that, it is still used as overlap
    unless it exceeds half a chunk, so most boundaries keep some shared context.
    """
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap_chars:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
    if not tail and sentences and len(sentences[-1]) <= chunk_chars // 2:
        tail = [sentences[-1]]
    return tail
