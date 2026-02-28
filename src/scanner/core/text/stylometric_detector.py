"""Scanner ULTRA — Stylometric analysis detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)

FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "but",
    "and",
    "or",
    "nor",
    "not",
    "so",
    "yet",
    "both",
    "either",
    "that",
    "which",
    "who",
    "this",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
}


class StylometricDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "stylometric"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.TEXT

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.TEXT_CONTENT}

    async def load_model(self) -> None:
        self.model = "stylometric_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.text or len(inp.text.strip()) < 100:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="stylo_skip",
                status=DetectorStatus.SKIPPED,
            )
        text = inp.text.strip()
        feat = self._analyze(text)
        score = self._score(feat)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=score,
            confidence=min(0.65, 0.2 + len(text) / 3000),
            method="stylometric_analysis",
            status=DetectorStatus.PASS,
            details={k: round(v, 4) for k, v in feat.items()},
        )

    @staticmethod
    def _analyze(text: str) -> dict[str, float]:
        words = text.lower().split()
        wc = max(len(words), 1)
        fw = sum(1 for w in words if w.strip(".,!?;:\"'()") in FUNCTION_WORDS)
        punct = sum(1 for c in text if c in ".,!?;:\"'()-")
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        pl = [len(p.split()) for p in paras] if paras else [wc]
        conjs = {"and", "but", "or", "because", "although", "however", "moreover"}
        cj = sum(1 for w in words if w.strip(".,!?;:") in conjs)
        prons = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "my", "your", "his"}
        pr = sum(1 for w in words if w.strip(".,!?;:") in prons)
        return {
            "fw_ratio": fw / wc,
            "punct_density": punct / wc,
            "para_std": float(np.std(pl)) if len(pl) > 1 else 0.0,
            "conj_rate": cj / wc,
            "pronoun_ratio": pr / wc,
        }

    @staticmethod
    def _score(f: dict[str, float]) -> float:
        s = 0.5
        if 0.42 <= f["fw_ratio"] <= 0.48:
            s += 0.1
        if f["para_std"] < 15:
            s += 0.1
        elif f["para_std"] > 40:
            s -= 0.05
        if f["pronoun_ratio"] < 0.03:
            s += 0.05
        return max(0.0, min(1.0, s))

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Stylometric Analyzer", "type": "statistical"}
