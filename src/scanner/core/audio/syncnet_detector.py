"""Scanner ULTRA — Audio-visual sync detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class SyncNetDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "syncnet_av"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.MULTIMODAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.AV_SYNC, DetectorCapability.AUDIO_TRACK, DetectorCapability.VIDEO_FRAMES}

    async def load_model(self) -> None:
        self.model = "syncnet_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.audio_waveform is None or not inp.frames or len(inp.frames) < 4:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="sync_skip",
                status=DetectorStatus.SKIPPED,
            )
        vis = self._visual_energy(inp.frames)
        aud = self._audio_energy(inp.audio_waveform, inp.audio_sr or 16000, len(inp.frames), inp.fps or 30.0)
        if len(vis) < 4 or len(aud) < 4:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="sync_short",
                status=DetectorStatus.WARN,
            )
        sync, offset = self._compute_sync(vis, aud)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=1.0 - sync,
            confidence=min(0.75, 0.3 + 0.05 * len(inp.frames)),
            method="av_sync_correlation",
            status=DetectorStatus.PASS,
            details={"sync_correlation": round(sync, 4), "offset_frames": offset},
        )

    @staticmethod
    def _visual_energy(frames: list[np.ndarray]) -> np.ndarray:
        e = []
        for f in frames:
            h, w = f.shape[:2]
            roi = f[2 * h // 3 :, w // 4 : 3 * w // 4]
            e.append(float(roi.mean()) if roi.size > 0 else 0.0)
        a = np.array(e)
        return (a - a.mean()) / (a.std() + 1e-8)

    @staticmethod
    def _audio_energy(wav: np.ndarray, sr: int, n: int, fps: float) -> np.ndarray:
        spf = int(sr / fps)
        e = [float(np.sqrt(np.mean(wav[i * spf : (i + 1) * spf] ** 2))) for i in range(n) if (i + 1) * spf <= len(wav)]
        a = np.array(e) if e else np.zeros(1)
        return (a - a.mean()) / (a.std() + 1e-8)

    @staticmethod
    def _compute_sync(vis: np.ndarray, aud: np.ndarray) -> tuple[float, int]:
        n = min(len(vis), len(aud))
        vis, aud = vis[:n], aud[:n]
        if n < 4:
            return 0.5, 0
        corr = np.correlate(vis, aud, mode="full")
        idx = np.argmax(np.abs(corr))
        norm = np.sqrt(np.sum(vis**2) * np.sum(aud**2))
        return min(1.0, float(np.abs(corr[idx]) / (norm + 1e-8))), int(idx - (n - 1))

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "SyncNet AV", "type": "cross-modal correlation"}
