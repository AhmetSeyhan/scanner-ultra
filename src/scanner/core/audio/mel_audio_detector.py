"""Scanner ULTRA — Mel-spectrogram CNN audio deepfake detector.

Architecture matches Colab training (pentashieldv2model.ipynb):
  PentaShieldAudioModel: Conv2d(1→32→64→128→256) + AdaptiveAvgPool2d(4,4) + classifier
Trained on: 3004lakshu/Deepfake-Audio dataset
Results: 100% accuracy, 100% AUC (604 samples)
Weights: weights/best_audio_model.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parents[4] / "weights"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "best_audio_model.pt"

# Colab training parameters
_SAMPLE_RATE = 16000
_DURATION_SEC = 4
_N_MELS = 128


class MelAudioDetector(BaseDetector):
    """Mel-spectrogram CNN trained for audio deepfake detection."""

    @property
    def name(self) -> str:
        return "mel_audio_cnn"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.AUDIO

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.AUDIO_TRACK}

    async def load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn

            # Colab eğitimiyle birebir aynı mimari
            class PentaShieldAudioModel(nn.Module):
                def __init__(self, n_mels: int = 128, n_classes: int = 2):
                    super().__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4)),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(256 * 4 * 4, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, n_classes),
                    )

                def forward(self, x):
                    return self.classifier(self.cnn(x))

            model = PentaShieldAudioModel(n_mels=_N_MELS, n_classes=2)

            weights_path = Path(self.model_path) if self.model_path else _DEFAULT_WEIGHTS

            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                logger.info("MelAudio weights yüklendi: %s", weights_path)
                self._weights_loaded = True
            else:
                logger.warning(
                    "Audio weights bulunamadı (%s) — stub mode. "
                    "Google Drive'dan indirin: scanner-ultra-weights/best_audio_model.pt",
                    weights_path,
                )
                self._weights_loaded = False

            self.model = model.to(self.device).eval()

        except ImportError:
            logger.warning("torch/librosa mevcut değil — stub mode")
            self.model = None
            self._weights_loaded = False

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.audio_waveform is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="mel_audio_skip",
                status=DetectorStatus.SKIPPED,
            )

        if self.model is None or not getattr(self, "_weights_loaded", False):
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="mel_audio_stub",
                status=DetectorStatus.WARN,
                details={"note": "Weights yok — best_audio_model.pt'yi weights/ klasörüne kopyalayın"},
            )

        try:
            import torch
            import torch.nn.functional as F

            mel = self._compute_mel(inp.audio_waveform, inp.audio_sr or _SAMPLE_RATE)
            if mel is None:
                return DetectorResult(
                    detector_name=self.name,
                    detector_type=self.detector_type,
                    score=0.5,
                    confidence=0.1,
                    method="mel_audio_fallback",
                    status=DetectorStatus.WARN,
                )

            tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                prob = F.softmax(logits, dim=-1)[0, 1].item()

            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=prob,
                confidence=0.85,
                method="mel_audio_cnn",
                status=DetectorStatus.PASS,
                details={"fake_prob": round(prob, 4), "n_mels": _N_MELS},
            )

        except Exception as exc:
            logger.error("MelAudioDetector hata: %s", exc)
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="mel_audio_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
            )

    def _compute_mel(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        try:
            import librosa

            max_len = _SAMPLE_RATE * _DURATION_SEC
            if sr != _SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=_SAMPLE_RATE)
            if len(wav) > max_len:
                wav = wav[:max_len]
            else:
                wav = np.pad(wav, (0, max_len - len(wav)))
            mel = librosa.feature.melspectrogram(y=wav, sr=_SAMPLE_RATE, n_mels=_N_MELS, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            return mel_db
        except Exception as exc:
            logger.warning("Mel hesaplama hatası: %s", exc)
            return None

    def get_model_info(self) -> dict[str, Any]:
        weights_loaded = _DEFAULT_WEIGHTS.exists()
        return {
            "name": "PentaShieldAudioModel (MelCNN)",
            "params": "1.44M",
            "input": f"{_N_MELS} mel bands, {_DURATION_SEC}s audio",
            "training": "3004lakshu/Deepfake-Audio dataset",
            "accuracy": "100%",
            "auc": "100%",
            "weights_file": str(_DEFAULT_WEIGHTS),
            "weights_loaded": weights_loaded,
        }
