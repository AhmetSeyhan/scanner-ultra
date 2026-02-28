"""Scanner ULTRA — GAN artifact detector.

Detects checkerboard patterns, color bleeding, boundary inconsistencies.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class GANArtifactDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "gan_artifact"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {
            DetectorCapability.VIDEO_FRAMES,
            DetectorCapability.SINGLE_IMAGE,
            DetectorCapability.GENERATOR_FINGERPRINT,
        }

    async def load_model(self) -> None:
        self.model = "artifact_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames or ([inp.image] if inp.image is not None else [])
        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="gan_skip",
                status=DetectorStatus.SKIPPED,
            )
        scores = [self._analyze(f) for f in frames[:8]]
        avg = float(np.mean(scores))
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=avg,
            confidence=min(0.8, 0.3 + 0.1 * len(scores)),
            method="gan_artifact_analysis",
            status=DetectorStatus.PASS,
            details={"n_frames": len(scores)},
        )

    def _analyze(self, frame: np.ndarray) -> float:
        return 0.4 * self._checkerboard(frame) + 0.3 * self._color_bleed(frame) + 0.3 * self._boundary(frame)

    @staticmethod
    def _checkerboard(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        hp = cv2.subtract(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        mag = np.abs(np.fft.fftshift(np.fft.fft2(hp.astype(np.float32))))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        corner = (
            mag[: h // 8, : w // 8].mean()
            + mag[: h // 8, -w // 8 :].mean()
            + mag[-h // 8 :, : w // 8].mean()
            + mag[-h // 8 :, -w // 8 :].mean()
        ) / 4
        center = mag[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8].mean()
        return float(min(1.0, corner / (center + 1e-8) / 0.5))

    @staticmethod
    def _color_bleed(frame: np.ndarray) -> float:
        if frame.ndim != 3 or frame.shape[2] < 3:
            return 0.0
        edges = [cv2.Canny(frame[:, :, i], 50, 150) for i in range(3)]
        diff = (
            np.abs(edges[0].astype(float) - edges[1].astype(float)).mean()
            + np.abs(edges[0].astype(float) - edges[2].astype(float)).mean()
        ) / 2
        return float(min(1.0, diff / 10.0))

    @staticmethod
    def _boundary(frame: np.ndarray) -> float:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
        )
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(sx**2 + sy**2)
        thresh = gm.mean() + 2 * gm.std()
        return float(min(1.0, (gm > thresh).sum() / gm.size * 50))

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "GAN Artifact Detector", "type": "signal_processing"}
