"""Scanner ULTRA — AI-generated text detector."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class AITextDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "ai_text_detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.TEXT

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.TEXT_CONTENT}

    async def load_model(self) -> None:
        self.model = "ai_text_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.text or len(inp.text.strip()) < 50:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="aitext_skip",
                status=DetectorStatus.SKIPPED,
            )
        text = inp.text.strip()
        feat = self._features(text)
        score = self._score(feat)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=score,
            confidence=min(0.7, 0.3 + len(text) / 5000),
            method="statistical_analysis",
            status=DetectorStatus.PASS,
            details={
                "word_count": feat["wc"],
                "vocab_richness": round(feat["ttr"], 4),
                "burstiness": round(feat["burst"], 4),
            },
        )

    @staticmethod
    def _features(text: str) -> dict[str, float]:
        words = text.split()
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        wc = len(words)
        unique = {w.lower() for w in words}
        ttr = len(unique) / max(wc, 1)
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        freq = Counter(w.lower() for w in words)
        fv = list(freq.values())
        burst = (np.std(fv) - np.mean(fv)) / (np.std(fv) + np.mean(fv) + 1e-8) if len(fv) > 1 else 0.0
        return {
            "wc": wc,
            "ttr": ttr,
            "sent_std": float(np.std(sent_lens)),
            "avg_wl": float(np.mean([len(w) for w in words])) if words else 0.0,
            "burst": float(burst),
        }

    @staticmethod
    def _score(f: dict[str, float]) -> float:
        s = 0.5
        if f["sent_std"] < 3.0:
            s += 0.15
        elif f["sent_std"] > 8.0:
            s -= 0.1
        if f["ttr"] > 0.8:
            s += 0.1
        elif f["ttr"] < 0.5:
            s -= 0.1
        if f["burst"] < -0.3:
            s += 0.1
        return max(0.0, min(1.0, s))

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "AI Text Detector", "type": "statistical"}
