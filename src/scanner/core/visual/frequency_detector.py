"""Scanner ULTRA — Frequency domain artifact detector.

Analyzes DCT/DFT spectral signatures for GAN fingerprints.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class FrequencyDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "frequency_analysis"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE, DetectorCapability.FREQUENCY_ANALYSIS}

    async def load_model(self) -> None:
        self.model = "frequency_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames or ([inp.image] if inp.image is not None else [])
        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="freq_skip",
                status=DetectorStatus.SKIPPED,
            )
        analyses = [self._analyze_frame(f) for f in frames[:8]]
        avg = float(np.mean([a["fake_score"] for a in analyses]))
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=avg,
            confidence=min(0.85, 0.3 + 0.1 * len(analyses)),
            method="dct_dft_spectral",
            status=DetectorStatus.PASS,
            details={
                "n_frames": len(analyses),
                "avg_flatness": round(float(np.mean([a["flatness"] for a in analyses])), 4),
            },
        )

    def _analyze_frame(self, frame: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        dft = np.fft.fft2(gray.astype(np.float32))
        mag = np.log1p(np.abs(np.fft.fftshift(dft)))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r = min(cy, cx)
        y, x = np.ogrid[:h, :w]
        low_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (r * 0.3) ** 2
        hf_ratio = 1.0 - (mag[low_mask].sum() / (mag.sum() + 1e-8))
        geo = np.exp(np.mean(np.log(mag + 1e-8)))
        flatness = float(geo / (np.mean(mag) + 1e-8))
        peak = self._periodic_peaks(mag)
        fake = 0.3 * hf_ratio + 0.3 * (1 - flatness) + 0.4 * peak
        return {"hf_ratio": float(hf_ratio), "flatness": float(flatness), "fake_score": float(max(0.0, min(1.0, fake)))}

    @staticmethod
    def _periodic_peaks(mag: np.ndarray) -> float:
        h, w = mag.shape
        h_prof = mag[h // 2, :]
        v_prof = mag[:, w // 2]

        def acf(p: np.ndarray) -> float:
            if len(p) < 10:
                return 0.0
            n = p - p.mean()
            a = np.correlate(n, n, mode="full")
            a = a[len(a) // 2 :]
            if a[0] == 0:
                return 0.0
            a = a / a[0]
            return min(1.0, len(np.where(a[2:] > 0.3)[0]) / 5.0)

        return (acf(h_prof) + acf(v_prof)) / 2.0

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Frequency Domain Analyzer", "type": "signal_processing"}
