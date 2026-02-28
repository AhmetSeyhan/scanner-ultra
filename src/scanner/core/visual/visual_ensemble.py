"""Scanner ULTRA — Visual ensemble detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "clip_forensic": 0.25,
    "efficientnet_b0": 0.15,
    "xception": 0.12,
    "vit_forgery": 0.10,
    "frequency_analysis": 0.10,
    "gan_artifact": 0.08,
    "diffusion_artifact": 0.08,
    "ppg_biosignal": 0.06,
    "gaze_consistency": 0.06,
}


class VisualEnsemble(BaseDetector):
    def __init__(
        self, model_path: str | None = None, device: str = "auto", weights: dict[str, float] | None = None
    ) -> None:
        super().__init__(model_path, device)
        self.weights = weights or DEFAULT_WEIGHTS

    @property
    def name(self) -> str:
        return "visual_ensemble"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    async def load_model(self) -> None:
        self.model = "ensemble_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        sub_results: dict[str, dict] = inp.metadata.get("visual_results", {})
        if not sub_results:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="ensemble_skip",
                status=DetectorStatus.SKIPPED,
            )
        ws, tw, confs = 0.0, 0.0, []
        for name, r in sub_results.items():
            w = self.weights.get(name, 0.05)
            c = r.get("confidence", 0.0)
            if c > 0:
                ws += w * r.get("score", 0.5) * c
                tw += w * c
                confs.append(c)
        final = ws / tw if tw > 0 else 0.5
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=final,
            confidence=min(0.95, float(np.mean(confs)) * 1.1) if confs else 0.0,
            method="weighted_ensemble",
            status=DetectorStatus.PASS,
            details={"n_detectors": len(sub_results), "active": len(confs)},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Visual Ensemble", "type": "weighted_voting", "weights": self.weights}
