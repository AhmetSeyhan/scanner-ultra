"""Scanner ULTRA — Voice cloning artifact detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class VoiceCloneDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "voice_clone"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.AUDIO

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.AUDIO_TRACK}

    async def load_model(self) -> None:
        self.model = "voice_clone_ready"

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.audio_waveform is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="vc_skip",
                status=DetectorStatus.SKIPPED,
            )
        sr = inp.audio_sr or 16000
        pitch = self._pitch(inp.audio_waveform, sr)
        prosody = self._prosody(inp.audio_waveform, sr)
        vocoder = self._vocoder(inp.audio_waveform, sr)
        score = 0.35 * pitch + 0.35 * prosody + 0.3 * vocoder
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=score,
            confidence=0.55,
            method="voice_clone_analysis",
            status=DetectorStatus.PASS,
            details={"pitch_monotony": round(pitch, 4), "prosody": round(prosody, 4), "vocoder": round(vocoder, 4)},
        )

    @staticmethod
    def _pitch(wav: np.ndarray, sr: int) -> float:
        frame_len = int(0.03 * sr)
        hop = frame_len // 2
        pitches = []
        for i in range(0, len(wav) - frame_len, hop):
            zc = np.sum(np.abs(np.diff(np.sign(wav[i : i + frame_len]))) > 0)
            f0 = zc * sr / (2 * frame_len)
            if 50 < f0 < 500:
                pitches.append(f0)
        if len(pitches) < 5:
            return 0.5
        std = np.std(pitches)
        if std < 10:
            return 0.8
        if std > 40:
            return 0.2
        return 0.5

    @staticmethod
    def _prosody(wav: np.ndarray, sr: int) -> float:
        fl = int(0.025 * sr)
        energies = [float(np.sqrt(np.mean(wav[i : i + fl] ** 2))) for i in range(0, len(wav) - fl, fl // 2)]
        if len(energies) < 10:
            return 0.5
        diff = np.abs(np.diff(energies))
        s = float(np.std(diff) / (np.mean(diff) + 1e-8))
        if s < 0.5:
            return 0.7
        if s > 3.0:
            return 0.3
        return 0.5

    @staticmethod
    def _vocoder(wav: np.ndarray, sr: int) -> float:
        fft = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
        hf = np.mean(fft[freqs > 7000]) if (freqs > 7000).any() else 0
        lf = np.mean(fft[(freqs > 100) & (freqs <= 7000)]) if ((freqs > 100) & (freqs <= 7000)).any() else 1
        ratio = hf / (lf + 1e-8)
        if ratio > 0.3:
            return 0.7
        if ratio < 0.05:
            return 0.6
        return 0.3

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Voice Clone Detector", "type": "signal_processing"}
