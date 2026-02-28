"""Scanner ULTRA — WavLM-based audio deepfake detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class WavLMDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "wavlm_forensic"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.AUDIO

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.AUDIO_TRACK}

    async def load_model(self) -> None:
        try:
            from transformers import WavLMModel

            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            self.model.to(self.device).eval()
        except ImportError:
            self.model = None

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.audio_waveform is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="wavlm_skip",
                status=DetectorStatus.SKIPPED,
            )
        if self.model is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="wavlm_stub",
                status=DetectorStatus.PASS,
                details={"mode": "stub"},
            )
        try:
            import torch

            wav = inp.audio_waveform[: 16000 * 10]
            tensor = torch.FloatTensor(wav).unsqueeze(0).to(self.device)
            with torch.no_grad():
                hidden = self.model(tensor).last_hidden_state
            std_feat = hidden.std(dim=1).cpu().numpy()[0]
            uniformity = 1.0 - min(1.0, float(np.mean(std_feat)) / 0.5)
            score = 0.3 + 0.4 * uniformity
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=score,
                confidence=0.6,
                method="wavlm_features",
                status=DetectorStatus.PASS,
                details={"hidden_dim": hidden.shape[-1]},
            )
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="wavlm_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
            )

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "WavLM Base Plus", "params": "94.7M", "input": "16kHz mono"}
