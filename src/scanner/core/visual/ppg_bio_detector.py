"""Scanner ULTRA — PPG biosignal-based detector.

Extracts photoplethysmography signals from face regions.
Real faces have consistent blood flow patterns; deepfakes don't.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class PPGBioDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "ppg_biosignal"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.BIOLOGICAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.BIOLOGICAL_SIGNAL}

    async def load_model(self) -> None:
        self.model = "ppg_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.frames or len(inp.frames) < 8:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="ppg_skip",
                status=DetectorStatus.SKIPPED,
            )
        ppg = self._extract_ppg(inp.frames)
        if ppg is None or len(ppg) < 8:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="ppg_insufficient",
                status=DetectorStatus.WARN,
            )
        periodicity = self._periodicity(ppg, inp.fps or 30.0)
        snr = self._snr(ppg)
        consistency = self._spatial_consistency(inp.frames)
        score = 0.4 * (1.0 - periodicity) + 0.3 * (1.0 - snr) + 0.3 * (1.0 - consistency)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=score,
            confidence=min(0.8, 0.3 + 0.5 * len(inp.frames) / 32.0),
            method="ppg_analysis",
            status=DetectorStatus.PASS,
            details={
                "periodicity": round(periodicity, 4),
                "snr": round(snr, 4),
                "spatial_consistency": round(consistency, 4),
            },
        )

    @staticmethod
    def _extract_ppg(frames: list[np.ndarray]) -> np.ndarray | None:
        signals = []
        for f in frames:
            h, w = f.shape[:2]
            roi = f[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            if roi.ndim == 3 and roi.shape[2] >= 3:
                signals.append(float(roi[:, :, 1].mean()))
        if len(signals) < 8:
            return None
        s = np.array(signals, dtype=np.float64)
        return s - np.linspace(s[0], s[-1], len(s))

    @staticmethod
    def _periodicity(signal: np.ndarray, fps: float) -> float:
        mag = np.abs(np.fft.rfft(signal))
        if mag.sum() == 0:
            return 0.0
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
        hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
        return float(mag[hr_mask].sum() / mag.sum())

    @staticmethod
    def _snr(signal: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(signal))
        if len(mag) < 2:
            return 0.0
        return float(min(1.0, mag.max() / (np.median(mag) + 1e-8) / 20.0))

    @staticmethod
    def _spatial_consistency(frames: list[np.ndarray]) -> float:
        sample = frames[:16]
        left_s, right_s = [], []
        for f in sample:
            h, w = f.shape[:2]
            mid = w // 2
            if f.ndim == 3 and f.shape[2] >= 3:
                left_s.append(float(f[h // 4 : 3 * h // 4, w // 4 : mid, 1].mean()))
                right_s.append(float(f[h // 4 : 3 * h // 4, mid : 3 * w // 4, 1].mean()))
        if len(left_s) < 4:
            return 0.5
        c = np.corrcoef(left_s, right_s)[0, 1]
        return float(max(0.0, c)) if not np.isnan(c) else 0.5

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "PPG Biosignal Analyzer", "type": "signal_processing"}
