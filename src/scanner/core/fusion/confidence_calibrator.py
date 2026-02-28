"""Scanner ULTRA — Confidence calibration (Platt / temperature scaling)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    def __init__(self, temperature: float = 1.5) -> None:
        self.temperature = temperature
        self._params: dict[str, dict[str, float]] = {}

    def calibrate(self, score: float, confidence: float, detector_name: str) -> tuple[float, float]:
        params = self._params.get(detector_name)
        if params:
            cal_score = self._sigmoid(params.get("a", 1.0) * score + params.get("b", 0.0))
        else:
            cal_score = self._temperature_scale(score)
        cal_conf = self._calibrate_confidence(confidence, detector_name)
        return cal_score, cal_conf

    def calibrate_batch(self, results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out = {}
        for name, r in results.items():
            cs, cc = self.calibrate(r.get("score", 0.5), r.get("confidence", 0.0), name)
            out[name] = {**r, "score": round(cs, 4), "confidence": round(cc, 4), "calibrated": True}
        return out

    def set_calibration(self, detector_name: str, a: float, b: float) -> None:
        self._params[detector_name] = {"a": a, "b": b}

    def _temperature_scale(self, score: float) -> float:
        eps = 1e-8
        s = max(eps, min(1.0 - eps, score))
        logit = np.log(s / (1.0 - s))
        return float(self._sigmoid(logit / self.temperature))

    def _calibrate_confidence(self, conf: float, name: str) -> float:
        rel = self._params.get(name, {}).get("reliability", 0.8)
        return min(0.95, conf * rel)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return float(ex / (1.0 + ex))
