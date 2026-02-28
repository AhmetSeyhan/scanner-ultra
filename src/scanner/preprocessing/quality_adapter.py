"""Scanner ULTRA — Quality adaptation layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class QualityReport:
    level: QualityLevel = QualityLevel.MEDIUM
    resolution_score: float = 0.5
    sharpness_score: float = 0.5
    noise_score: float = 0.5
    overall_score: float = 0.5
    recommendations: list[str] | None = None

    def __post_init__(self) -> None:
        if self.recommendations is None:
            self.recommendations = []


class QualityAdapter:
    def assess_image(self, image: np.ndarray) -> QualityReport:
        h, w = image.shape[:2]
        res = self._score_resolution(w, h)
        sharp = self._score_sharpness(image)
        noise = self._score_noise(image)
        overall = 0.4 * res + 0.35 * sharp + 0.25 * noise
        recs: list[str] = []
        if res < 0.3:
            recs.append("Very low resolution")
        if sharp < 0.3:
            recs.append("Image is blurry")
        return QualityReport(
            level=self._to_level(overall),
            resolution_score=res,
            sharpness_score=sharp,
            noise_score=noise,
            overall_score=overall,
            recommendations=recs,
        )

    def assess_frames(self, frames: list[np.ndarray]) -> QualityReport:
        if not frames:
            return QualityReport(level=QualityLevel.VERY_LOW, overall_score=0.0)
        indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
        reports = [self.assess_image(frames[i]) for i in indices]
        overall = float(np.mean([r.overall_score for r in reports]))
        return QualityReport(level=self._to_level(overall), overall_score=overall)

    def assess_audio(self, waveform: np.ndarray | None, sr: int) -> QualityReport:
        if waveform is None or len(waveform) == 0:
            return QualityReport(level=QualityLevel.VERY_LOW, overall_score=0.0)
        rms = float(np.sqrt(np.mean(waveform**2)))
        duration = len(waveform) / sr if sr > 0 else 0.0
        overall = 0.6 * min(1.0, rms / 0.1) + 0.4 * min(1.0, duration / 3.0)
        return QualityReport(level=self._to_level(overall), overall_score=overall)

    def get_confidence_weight(self, quality: QualityReport) -> float:
        return {
            QualityLevel.HIGH: 1.0,
            QualityLevel.MEDIUM: 0.8,
            QualityLevel.LOW: 0.5,
            QualityLevel.VERY_LOW: 0.3,
        }.get(quality.level, 0.5)

    @staticmethod
    def _score_resolution(w: int, h: int) -> float:
        px = w * h
        if px >= 1920 * 1080:
            return 1.0
        if px >= 1280 * 720:
            return 0.7
        if px >= 640 * 480:
            return 0.4
        return 0.2

    @staticmethod
    def _score_sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))

    @staticmethod
    def _score_noise(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        median = cv2.medianBlur(gray, 5)
        diff = np.abs(gray.astype(np.float32) - median.astype(np.float32))
        return float(max(0.0, 1.0 - diff.mean() / 30.0))

    @staticmethod
    def _to_level(score: float) -> QualityLevel:
        if score >= 0.7:
            return QualityLevel.HIGH
        if score >= 0.4:
            return QualityLevel.MEDIUM
        if score >= 0.2:
            return QualityLevel.LOW
        return QualityLevel.VERY_LOW
