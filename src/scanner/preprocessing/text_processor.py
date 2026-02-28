"""Scanner ULTRA — Text preprocessing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TextFeatures:
    original: str = ""
    cleaned: str = ""
    chunks: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    vocabulary_richness: float = 0.0
    metadata: dict = field(default_factory=dict)


class TextProcessor:
    def __init__(self, max_length: int = 8192, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process(self, content: bytes | str, filename: str | None = None) -> TextFeatures:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        text = text[: self.max_length]
        cleaned = re.sub(r"\s+", " ", text).strip()
        chunks = self._chunk(cleaned)
        words = cleaned.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        unique_words = {w.lower() for w in words}
        return TextFeatures(
            original=text,
            cleaned=cleaned,
            chunks=chunks,
            word_count=len(words),
            sentence_count=len(sentences),
            avg_word_length=sum(len(w) for w in words) / max(len(words), 1),
            vocabulary_richness=len(unique_words) / max(len(words), 1),
            metadata={"source_file": filename, "truncated": len(text) >= self.max_length},
        )

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if text else []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks
