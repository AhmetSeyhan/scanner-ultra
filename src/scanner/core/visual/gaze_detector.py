"""Scanner ULTRA — Gaze consistency detector."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class GazeDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "gaze_consistency"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.BIOLOGICAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.BIOLOGICAL_SIGNAL}

    async def load_model(self) -> None:
        self.model = "gaze_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.frames or len(inp.frames) < 4:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="gaze_skip",
                status=DetectorStatus.SKIPPED,
            )
        gaze_points = [p for f in inp.frames[:32] if (p := self._estimate_gaze(f)) is not None]
        if len(gaze_points) < 4:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="gaze_insufficient",
                status=DetectorStatus.WARN,
            )
        smoothness = self._smoothness(gaze_points)
        saccade = self._saccade_pattern(gaze_points)
        score = 0.5 * (1.0 - smoothness) + 0.5 * (1.0 - saccade)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=score,
            confidence=min(0.7, 0.2 + 0.02 * len(gaze_points)),
            method="gaze_analysis",
            status=DetectorStatus.PASS,
            details={"n_points": len(gaze_points), "smoothness": round(smoothness, 4)},
        )

    @staticmethod
    def _estimate_gaze(frame: np.ndarray) -> tuple[float, float] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = cascade.detectMultiScale(gray, 1.1, 5)
        if len(eyes) < 1:
            return None
        ex, ey, ew, eh = eyes[0]
        roi = gray[ey : ey + eh, ex : ex + ew]
        if roi.size == 0:
            return None
        _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        m = cv2.moments(thresh)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"] / ew, m["m01"] / m["m00"] / eh)

    @staticmethod
    def _smoothness(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.5
        vels = [
            np.sqrt((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2)
            for i in range(1, len(points))
        ]
        return float(min(1.0, max(0.0, 1.0 - np.mean(vels) * 5)))

    @staticmethod
    def _saccade_pattern(points: list[tuple[float, float]]) -> float:
        if len(points) < 4:
            return 0.5
        vels = np.array(
            [
                np.sqrt((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2)
                for i in range(1, len(points))
            ]
        )
        thresh = np.mean(vels) + np.std(vels)
        sr = (vels > thresh).sum() / len(vels)
        fr = (vels < np.mean(vels) * 0.5).sum() / len(vels)
        n = (0.5 if 0.05 <= sr <= 0.3 else 0.0) + (0.5 if 0.3 <= fr <= 0.8 else 0.0)
        return n

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Gaze Consistency Analyzer", "type": "signal_processing"}
