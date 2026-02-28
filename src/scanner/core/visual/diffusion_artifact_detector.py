"""Scanner ULTRA — Diffusion model artifact detector."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class DiffusionArtifactDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "diffusion_artifact"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.SINGLE_IMAGE, DetectorCapability.GENERATOR_FINGERPRINT}

    async def load_model(self) -> None:
        self.model = "diffusion_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames or ([inp.image] if inp.image is not None else [])
        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="diffusion_skip",
                status=DetectorStatus.SKIPPED,
            )
        scores = [self._analyze(f) for f in frames[:8]]
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=float(np.mean(scores)),
            confidence=min(0.7, 0.2 + 0.1 * len(scores)),
            method="diffusion_artifact_analysis",
            status=DetectorStatus.PASS,
            details={"n_frames": len(scores)},
        )

    def _analyze(self, frame: np.ndarray) -> float:
        return 0.5 * self._texture(frame) + 0.5 * self._noise(frame)

    @staticmethod
    def _texture(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        if v < 50:
            return 0.8
        if v > 500:
            return 0.2
        return 0.5

    @staticmethod
    def _noise(frame: np.ndarray) -> float:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
        )
        residual = gray - cv2.GaussianBlur(gray, (5, 5), 1.5)
        bs = 32
        h, w = residual.shape
        local_vars = [
            residual[y : y + bs, x : x + bs].var() for y in range(0, h - bs, bs) for x in range(0, w - bs, bs)
        ]
        if not local_vars:
            return 0.5
        return 0.7 if np.var(local_vars) < 1.0 else 0.3

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Diffusion Artifact Detector", "type": "signal_processing"}
