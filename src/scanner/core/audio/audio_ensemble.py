"""Scanner ULTRA — Audio ensemble detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "wavlm_forensic": 0.25,
    "cqt_spectral": 0.25,
    "ecapa_tdnn": 0.15,
    "voice_clone": 0.15,
    "syncnet_av": 0.20,
}


class AudioEnsemble(BaseDetector):
    def __init__(
        self, model_path: str | None = None, device: str = "auto", weights: dict[str, float] | None = None
    ) -> None:
        super().__init__(model_path, device)
        self.weights = weights or DEFAULT_WEIGHTS

    @property
    def name(self) -> str:
        return "audio_ensemble"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.AUDIO

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.AUDIO_TRACK}

    async def load_model(self) -> None:
        self.model = "audio_ensemble_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        sub: dict[str, dict] = inp.metadata.get("audio_results", {})
        if not sub:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="audio_ensemble_skip",
                status=DetectorStatus.SKIPPED,
            )
        ws, tw, confs = 0.0, 0.0, []
        for name, r in sub.items():
            w = self.weights.get(name, 0.05)
            c = r.get("confidence", 0.0)
            if c > 0:
                ws += w * r.get("score", 0.5) * c
                tw += w * c
                confs.append(c)
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=ws / tw if tw > 0 else 0.5,
            confidence=min(0.9, float(np.mean(confs)) * 1.1) if confs else 0.0,
            method="audio_weighted_ensemble",
            status=DetectorStatus.PASS,
            details={"n_detectors": len(sub), "active": len(confs)},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Audio Ensemble", "weights": self.weights}
