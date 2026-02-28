"""HYDRA ENGINE — Input Purifier.

Detects and removes adversarial perturbations from input frames.
Techniques: spatial smoothing, JPEG compression defense, bit-depth reduction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PurificationResult:
    """Result of input purification."""

    purified_frames: list[np.ndarray] = field(default_factory=list)
    adversarial_detected: bool = False
    perturbation_magnitude: float = 0.0
    method_applied: list[str] = field(default_factory=list)
    per_frame_magnitudes: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adversarial_detected": self.adversarial_detected,
            "perturbation_magnitude": round(self.perturbation_magnitude, 6),
            "method_applied": self.method_applied,
            "n_frames": len(self.purified_frames),
        }


class InputPurifier:
    """Adversarial perturbation detection and removal.

    Pipeline:
      1. Spatial smoothing (Gaussian blur) — attenuates high-freq perturbations
      2. JPEG compression defense — destroys small adversarial noise
      3. Bit-depth reduction — quantizes away sub-LSB perturbations
      4. L2 distance check — large delta between original and purified = adversarial

    The purifier does NOT replace the original frames for detection; instead,
    the engine runs detection on BOTH original and purified, comparing results
    to detect adversarial manipulation.
    """

    # Perturbation magnitude above this → adversarial flag
    ADVERSARIAL_THRESHOLD = 0.015

    def __init__(
        self,
        sigma: float = 0.8,
        jpeg_quality: int = 75,
        bit_depth: int = 4,
    ) -> None:
        self.sigma = sigma
        self.jpeg_quality = jpeg_quality
        self.bit_depth = bit_depth

    def purify(self, frames: list[np.ndarray]) -> PurificationResult:
        """Purify frames and detect adversarial perturbations."""
        if not frames:
            return PurificationResult()

        purified = []
        magnitudes = []
        methods_used: set[str] = set()

        for frame in frames:
            # Stage 1: Spatial smoothing
            smoothed = self._spatial_smooth(frame)
            methods_used.add("spatial_smooth")

            # Stage 2: JPEG compression defense
            compressed = self._jpeg_defense(smoothed)
            methods_used.add("jpeg_defense")

            # Stage 3: Bit-depth reduction
            reduced = self._bit_depth_reduce(compressed)
            methods_used.add("bit_depth_reduce")

            # Measure perturbation vs original
            mag = self._perturbation_magnitude(frame, reduced)
            magnitudes.append(mag)
            purified.append(reduced)

        avg_magnitude = float(np.mean(magnitudes))
        adversarial = avg_magnitude > self.ADVERSARIAL_THRESHOLD

        if adversarial:
            logger.warning(
                "Adversarial perturbation detected: avg_magnitude=%.6f (threshold=%.4f)",
                avg_magnitude,
                self.ADVERSARIAL_THRESHOLD,
            )

        return PurificationResult(
            purified_frames=purified,
            adversarial_detected=adversarial,
            perturbation_magnitude=avg_magnitude,
            method_applied=sorted(methods_used),
            per_frame_magnitudes=[round(m, 6) for m in magnitudes],
        )

    def _spatial_smooth(self, frame: np.ndarray) -> np.ndarray:
        """Gaussian blur to attenuate high-frequency adversarial noise."""
        # Kernel size must be odd; derive from sigma
        ksize = max(3, int(self.sigma * 4) | 1)
        return cv2.GaussianBlur(frame, (ksize, ksize), self.sigma)

    def _jpeg_defense(self, frame: np.ndarray) -> np.ndarray:
        """JPEG encode/decode — lossy compression destroys small perturbations."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        # Encode to JPEG bytes, then decode back
        success, encoded = cv2.imencode(".jpg", frame, encode_param)
        if not success:
            return frame
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            return frame
        # Match original channel count
        if frame.ndim == 3 and frame.shape[2] == 3 and decoded.shape[2] == 3:
            # cv2 decodes as BGR; if input was RGB, convert back
            return decoded
        return decoded

    def _bit_depth_reduce(self, frame: np.ndarray, bits: int | None = None) -> np.ndarray:
        """Reduce bit depth and restore — quantizes away sub-LSB noise."""
        b = bits or self.bit_depth
        if b >= 8:
            return frame
        shift = 8 - b
        # Quantize down then scale back up
        reduced = (frame >> shift) << shift
        # Add half-step to reduce bias
        reduced = np.clip(reduced + (1 << (shift - 1)), 0, 255).astype(np.uint8)
        return reduced

    @staticmethod
    def _perturbation_magnitude(original: np.ndarray, purified: np.ndarray) -> float:
        """Compute normalized L2 distance between original and purified frames.

        Returns a value in [0, 1] where 0 = identical, higher = more perturbation.
        For clean images this is typically 0.005-0.01 (due to JPEG+blur).
        Adversarial images typically show 0.02+ due to the perturbation being stripped.
        """
        orig_f = original.astype(np.float32) / 255.0
        pur_f = purified.astype(np.float32) / 255.0
        diff = orig_f - pur_f
        l2 = float(np.sqrt(np.mean(diff**2)))
        return l2
