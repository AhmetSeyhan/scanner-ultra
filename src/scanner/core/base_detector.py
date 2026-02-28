"""Scanner ULTRA — Abstract BaseDetector.

Every detection engine (visual, audio, text) inherits from BaseDetector.
Provides async lifecycle, lazy model loading, timing, and error handling.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from scanner.models.enums import (
    DetectorCapability,
    DetectorStatus,
    DetectorType,
)

logger = logging.getLogger(__name__)


class DetectorResult:
    """Standardised output every detector must return."""

    __slots__ = (
        "detector_name",
        "detector_type",
        "score",
        "confidence",
        "method",
        "status",
        "details",
        "heatmap",
        "processing_time_ms",
    )

    def __init__(
        self,
        *,
        detector_name: str,
        detector_type: DetectorType,
        score: float,
        confidence: float,
        method: str = "",
        status: DetectorStatus = DetectorStatus.PASS,
        details: dict[str, Any] | None = None,
        heatmap: bytes | None = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.detector_name = detector_name
        self.detector_type = detector_type
        self.score = max(0.0, min(1.0, score))
        self.confidence = max(0.0, min(1.0, confidence))
        self.method = method
        self.status = status
        self.details = details or {}
        self.heatmap = heatmap
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_type": self.detector_type.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "status": self.status.value,
            "details": self.details,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class DetectorInput:
    """Unified input payload for all detectors.

    Each detector picks the fields it needs and ignores the rest.
    """

    __slots__ = (
        "frames",
        "fps",
        "video_path",
        "audio_waveform",
        "audio_sr",
        "image",
        "text",
        "metadata",
    )

    def __init__(
        self,
        *,
        frames: list[np.ndarray] | None = None,
        fps: float = 0.0,
        video_path: str | None = None,
        audio_waveform: np.ndarray | None = None,
        audio_sr: int = 0,
        image: np.ndarray | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.frames = frames
        self.fps = fps
        self.video_path = video_path
        self.audio_waveform = audio_waveform
        self.audio_sr = audio_sr
        self.image = image
        self.text = text
        self.metadata = metadata or {}


class BaseDetector(ABC):
    """Abstract base class for every Scanner detection engine.

    Subclasses MUST implement:
      - name, detector_type, capabilities (properties)
      - load_model()  — async model loading
      - detect()      — async detection
      - get_model_info()
    """

    def __init__(self, model_path: str | None = None, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model: Any = None
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    # --- abstract properties ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector identifier."""
        ...

    @property
    @abstractmethod
    def detector_type(self) -> DetectorType: ...

    @property
    @abstractmethod
    def capabilities(self) -> set[DetectorCapability]: ...

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def enabled(self) -> bool:
        return True

    # --- lifecycle ---

    async def ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if not self._loaded:
            logger.info("Loading model for %s on %s", self.name, self.device)
            await self.load_model()
            self._loaded = True

    @abstractmethod
    async def load_model(self) -> None:
        """Load model weights. Called once via ensure_loaded()."""
        ...

    async def shutdown(self) -> None:
        """Optional cleanup on application shutdown."""
        self.model = None
        self._loaded = False

    def health_check(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.detector_type.value,
            "loaded": self._loaded,
            "enabled": self.enabled,
            "device": self.device,
        }

    # --- detection ---

    @abstractmethod
    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        """Core detection logic. Subclasses implement this."""
        ...

    async def detect(self, inp: DetectorInput) -> DetectorResult:
        """Public entry-point: ensures model loaded, wraps with timing and error handling."""
        await self.ensure_loaded()
        start = time.perf_counter()
        try:
            result = await self._run_detection(inp)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Detector %s failed: %s", self.name, exc, exc_info=True)
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method=f"{self.name}_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
                processing_time_ms=elapsed,
            )
        result.processing_time_ms = (time.perf_counter() - start) * 1000
        return result

    # --- info ---

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata (name, params, size, training status)."""
        ...

    def __repr__(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities)
        return f"<{self.__class__.__name__} name={self.name!r} type={self.detector_type.value} caps=[{caps}]>"
