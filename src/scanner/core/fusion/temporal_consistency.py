"""Scanner ULTRA — Temporal consistency analyzer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TemporalConsistency:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

    def analyze(self, frame_scores: list[float]) -> dict[str, Any]:
        if not frame_scores or len(frame_scores) < 2:
            return {"consistency": 0.5, "trend": "unknown", "flickering": False, "confidence_modifier": 1.0}

        arr = np.array(frame_scores)
        diffs = np.abs(np.diff(arr))
        smoothness = float(1.0 - np.mean(diffs))

        sign_changes = np.sum(np.abs(np.diff(np.sign(arr - 0.5))) > 0)
        flicker_rate = sign_changes / len(arr)
        is_flickering = flicker_rate > 0.3

        if len(arr) >= 3:
            slope = np.polyfit(range(len(arr)), arr, 1)[0]
            trend = "increasing" if slope > 0.02 else ("decreasing" if slope < -0.02 else "stable")
        else:
            trend = "unknown"

        if is_flickering:
            cm = 0.5
        elif smoothness > 0.8:
            cm = 1.2
        else:
            cm = 0.8 + 0.4 * smoothness

        return {
            "consistency": round(smoothness, 4),
            "trend": trend,
            "flickering": is_flickering,
            "flicker_rate": round(flicker_rate, 4),
            "score_std": round(float(np.std(arr)), 4),
            "confidence_modifier": round(min(1.5, cm), 4),
        }

    def smooth_scores(self, scores: list[float]) -> list[float]:
        if len(scores) <= self.window_size:
            return scores
        kernel = np.ones(self.window_size) / self.window_size
        return np.convolve(np.array(scores), kernel, mode="same").tolist()
