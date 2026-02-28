"""Scanner ULTRA — Audio preprocessing.

Extracts waveform features: CQT, LogSpec (NOT MelSpec — 37% improvement).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    waveform: np.ndarray | None = None
    sr: int = 16000
    cqt: np.ndarray | None = None
    log_spec: np.ndarray | None = None
    duration_sec: float = 0.0
    rms_energy: float = 0.0
    metadata: dict = field(default_factory=dict)


class AudioProcessor:
    def __init__(
        self, target_sr: int = 16000, max_duration: float = 10.0, n_bins: int = 84, hop_length: int = 512
    ) -> None:
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.n_bins = n_bins
        self.hop_length = hop_length

    async def process(self, content: bytes, filename: str) -> AudioFeatures:
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            waveform, sr = self._load_audio(tmp_path)
            cqt = self._compute_cqt(waveform, sr)
            log_spec = self._compute_log_spectrogram(waveform, sr)
            rms = float(np.sqrt(np.mean(waveform**2))) if waveform is not None else 0.0
            return AudioFeatures(
                waveform=waveform,
                sr=sr,
                cqt=cqt,
                log_spec=log_spec,
                duration_sec=len(waveform) / sr if waveform is not None and sr > 0 else 0.0,
                rms_energy=rms,
                metadata={"source_file": filename},
            )
        except Exception:
            logger.exception("Audio processing failed for %s", filename)
            return AudioFeatures(metadata={"error": "processing_failed"})

    def _load_audio(self, path: str) -> tuple[np.ndarray | None, int]:
        try:
            import librosa

            waveform, sr = librosa.load(path, sr=self.target_sr, mono=True, duration=self.max_duration)
            return waveform, sr
        except ImportError:
            return self._load_with_scipy(path)

    def _load_with_scipy(self, path: str) -> tuple[np.ndarray | None, int]:
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(path)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data[: int(self.max_duration * sr)], sr
        except Exception:
            return None, 0

    def _compute_cqt(self, waveform: np.ndarray | None, sr: int) -> np.ndarray | None:
        if waveform is None:
            return None
        try:
            import librosa

            cqt = np.abs(
                librosa.cqt(waveform, sr=sr, n_bins=self.n_bins, bins_per_octave=12, hop_length=self.hop_length)
            )
            return librosa.amplitude_to_db(cqt, ref=np.max)
        except ImportError:
            return None

    @staticmethod
    def _compute_log_spectrogram(waveform: np.ndarray | None, sr: int) -> np.ndarray | None:
        if waveform is None:
            return None
        try:
            from scipy.signal import stft

            _, _, zxx = stft(waveform, fs=sr, nperseg=1024, noverlap=768)
            return 10 * np.log10(np.abs(zxx) ** 2 + 1e-10)
        except ImportError:
            return None

    async def process_waveform(self, waveform: np.ndarray, sr: int) -> AudioFeatures:
        waveform = waveform[: int(self.max_duration * sr)]
        return AudioFeatures(
            waveform=waveform,
            sr=sr,
            cqt=self._compute_cqt(waveform, sr),
            log_spec=self._compute_log_spectrogram(waveform, sr),
            duration_sec=len(waveform) / sr if sr > 0 else 0.0,
            rms_energy=float(np.sqrt(np.mean(waveform**2))),
        )
