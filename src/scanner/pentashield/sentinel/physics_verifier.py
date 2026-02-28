"""ZERO-DAY SENTINEL — Physics Verifier.

Verifies physical consistency in visual content.  Deepfakes often violate
physical laws that are hard to model: lighting direction, shadow consistency,
specular reflections, color temperature, and boundary sharpness.

All checks are signal-processing based (OpenCV) — no ML models required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsResult:
    """Result from physics verification."""

    physics_score: float = 1.0  # 1.0 = consistent, 0.0 = many anomalies
    check_scores: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics_score": round(self.physics_score, 4),
            "check_scores": {k: round(v, 4) for k, v in self.check_scores.items()},
            "anomalies": self.anomalies,
            "n_anomalies": len(self.anomalies),
        }


class PhysicsVerifier:
    """Verifies physical consistency of visual frames.

    Runs 5 independent checks, each producing a consistency score [0, 1]:
      1. Lighting direction — L/R brightness ratio stability
      2. Shadow consistency — gradient direction coherence
      3. Specular reflection — eye catchlight symmetry
      4. Color temperature — face vs background color match
      5. Edge sharpness — boundary discontinuity detection

    Each check returns (name, score, anomaly_string_or_None).
    """

    # Score below this triggers anomaly text
    ANOMALY_THRESHOLD = 0.6

    def verify(self, frames: list[np.ndarray]) -> PhysicsResult:
        """Verify physical consistency across frames."""
        if not frames:
            return PhysicsResult(physics_score=1.0)

        # Sample frames for efficiency (max 8)
        sample = frames[:: max(1, len(frames) // 8)][:8]

        checks = [
            self._lighting,
            self._shadow,
            self._specular,
            self._color_temperature,
            self._edge_gradient,
        ]

        scores: dict[str, float] = {}
        anomalies: list[str] = []

        for check_fn in checks:
            try:
                name, score, anomaly = check_fn(sample)
                scores[name] = score
                if anomaly:
                    anomalies.append(anomaly)
            except Exception as exc:
                logger.debug("Physics check failed: %s", exc)

        overall = float(np.mean(list(scores.values()))) if scores else 1.0

        return PhysicsResult(
            physics_score=overall,
            check_scores=scores,
            anomalies=anomalies,
        )

    def _lighting(self, frames: list[np.ndarray]) -> tuple[str, float, str | None]:
        """Check lighting direction consistency across frames.

        Compares brightness ratio of left vs right face halves.
        In real video, this ratio is stable across frames.
        """
        ratios = []
        for frame in frames:
            gray = self._to_gray(frame)
            h, w = gray.shape
            # Center face region (middle 50%)
            y1, y2 = h // 4, 3 * h // 4
            x_mid = w // 2

            left_mean = float(gray[y1:y2, w // 4 : x_mid].mean()) + 1e-8
            right_mean = float(gray[y1:y2, x_mid : 3 * w // 4].mean()) + 1e-8
            ratio = left_mean / right_mean
            ratios.append(ratio)

        if len(ratios) < 2:
            return ("lighting", 1.0, None)

        # Consistency = low std in ratios → stable lighting direction
        std = float(np.std(ratios))
        score = max(0.0, min(1.0, 1.0 - std * 5.0))

        anomaly = None
        if score < self.ANOMALY_THRESHOLD:
            anomaly = f"Lighting direction inconsistency: L/R ratio std={std:.3f} across frames"

        return ("lighting", score, anomaly)

    def _shadow(self, frames: list[np.ndarray]) -> tuple[str, float, str | None]:
        """Check shadow direction consistency via gradient analysis.

        Computes dominant gradient direction (Sobel) in the nose region.
        Consistent shadow direction across frames = consistent score.
        """
        angles = []
        for frame in frames:
            gray = self._to_gray(frame)
            h, w = gray.shape
            # Nose region (center of face)
            roi = gray[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
            if roi.size == 0:
                continue

            gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            # Dominant gradient angle
            angle = float(np.arctan2(gy.mean(), gx.mean()))
            angles.append(angle)

        if len(angles) < 2:
            return ("shadow", 1.0, None)

        # Angular consistency via circular std
        angles_arr = np.array(angles)
        mean_sin = float(np.mean(np.sin(angles_arr)))
        mean_cos = float(np.mean(np.cos(angles_arr)))
        r = np.sqrt(mean_sin**2 + mean_cos**2)
        score = float(np.clip(r, 0.0, 1.0))

        anomaly = None
        if score < self.ANOMALY_THRESHOLD:
            anomaly = f"Shadow direction inconsistency: circular R={r:.3f}"

        return ("shadow", score, anomaly)

    def _specular(self, frames: list[np.ndarray]) -> tuple[str, float, str | None]:
        """Check corneal reflection (catchlight) symmetry.

        In real faces, both eyes reflect the same light source(s) in
        similar positions.  We detect bright spots in the eye regions
        and check L/R symmetry.
        """
        symmetries = []
        for frame in frames:
            gray = self._to_gray(frame)
            h, w = gray.shape

            # Approximate eye regions (top-center)
            eye_y1, eye_y2 = h // 4, h // 2
            left_eye = gray[eye_y1:eye_y2, w // 6 : w // 2 - w // 12]
            right_eye = gray[eye_y1:eye_y2, w // 2 + w // 12 : 5 * w // 6]

            if left_eye.size == 0 or right_eye.size == 0:
                continue

            # Find brightest spot (catchlight) in each eye
            left_max = float(left_eye.max())
            right_max = float(right_eye.max())
            left_pos = np.unravel_index(left_eye.argmax(), left_eye.shape)
            right_pos = np.unravel_index(right_eye.argmax(), right_eye.shape)

            # Brightness similarity
            bright_sim = 1.0 - abs(left_max - right_max) / 255.0

            # Position similarity (normalized by eye region size)
            if left_eye.shape[0] > 0 and left_eye.shape[1] > 0:
                y_sim = 1.0 - abs(left_pos[0] / left_eye.shape[0] - right_pos[0] / right_eye.shape[0])
                x_sim = 1.0 - abs(left_pos[1] / left_eye.shape[1] - right_pos[1] / right_eye.shape[1])
            else:
                y_sim = x_sim = 1.0

            sym = (bright_sim + y_sim + x_sim) / 3.0
            symmetries.append(sym)

        if not symmetries:
            return ("specular", 1.0, None)

        score = float(np.mean(symmetries))

        anomaly = None
        if score < self.ANOMALY_THRESHOLD:
            anomaly = f"Eye reflection asymmetry detected: symmetry={score:.3f}"

        return ("specular", score, anomaly)

    def _color_temperature(self, frames: list[np.ndarray]) -> tuple[str, float, str | None]:
        """Check face vs background color temperature consistency.

        Real images have consistent white balance. Composited faces may
        have different color temperature than the background.
        """
        diffs = []
        for frame in frames:
            if frame.ndim != 3 or frame.shape[2] < 3:
                continue

            h, w = frame.shape[:2]

            # Face region (center)
            face = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            # Background (corners)
            bg_tl = frame[: h // 6, : w // 6]
            bg_tr = frame[: h // 6, 5 * w // 6 :]
            bg_bl = frame[5 * h // 6 :, : w // 6]
            bg_br = frame[5 * h // 6 :, 5 * w // 6 :]
            bg = (
                np.concatenate(
                    [r.reshape(-1, 3) for r in [bg_tl, bg_tr, bg_bl, bg_br] if r.size > 0],
                    axis=0,
                )
                if any(r.size > 0 for r in [bg_tl, bg_tr, bg_bl, bg_br])
                else None
            )

            if bg is None or bg.shape[0] == 0 or face.size == 0:
                continue

            face_flat = face.reshape(-1, 3).astype(np.float32)

            # Color temperature proxy: R/B ratio
            face_rb = float(np.mean(face_flat[:, 0]) / (np.mean(face_flat[:, 2]) + 1e-8))
            bg_rb = float(np.mean(bg[:, 0].astype(np.float32)) / (np.mean(bg[:, 2].astype(np.float32)) + 1e-8))

            diff = abs(face_rb - bg_rb)
            diffs.append(diff)

        if not diffs:
            return ("color_temperature", 1.0, None)

        avg_diff = float(np.mean(diffs))
        # Normalize: diff > 0.5 is highly suspicious
        score = max(0.0, min(1.0, 1.0 - avg_diff * 2.0))

        anomaly = None
        if score < self.ANOMALY_THRESHOLD:
            anomaly = f"Color temperature mismatch between face and background: diff={avg_diff:.3f}"

        return ("color_temperature", score, anomaly)

    def _edge_gradient(self, frames: list[np.ndarray]) -> tuple[str, float, str | None]:
        """Check face boundary sharpness for pasting artifacts.

        Composited faces often have abnormal edge sharpness at the paste
        boundary — either too sharp (no blending) or too smooth (heavy feathering).
        Compare face boundary edge magnitude to interior edge magnitude.
        """
        ratios = []
        for frame in frames:
            gray = self._to_gray(frame)
            h, w = gray.shape

            # Compute edge magnitude
            edges = cv2.Canny(gray, 50, 150)

            # Face boundary ring (between 20% and 30% from center)
            mask_outer = np.zeros_like(edges, dtype=bool)
            mask_inner = np.zeros_like(edges, dtype=bool)

            cy, cx = h // 2, w // 2
            r_outer = min(cy, cx) * 0.45
            r_inner = min(cy, cx) * 0.35

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

            mask_outer[(dist >= r_inner) & (dist <= r_outer)] = True
            mask_inner[dist < r_inner] = True

            boundary_density = float(edges[mask_outer].mean()) if mask_outer.any() else 0
            interior_density = float(edges[mask_inner].mean()) if mask_inner.any() else 0

            if interior_density > 0:
                ratio = boundary_density / (interior_density + 1e-8)
                ratios.append(ratio)

        if not ratios:
            return ("edge_gradient", 1.0, None)

        avg_ratio = float(np.mean(ratios))
        # Ideal ratio is ~1.0. Deviation indicates artifact.
        # Score decreases as ratio deviates from 1.0
        deviation = abs(avg_ratio - 1.0)
        score = max(0.0, min(1.0, 1.0 - deviation))

        anomaly = None
        if score < self.ANOMALY_THRESHOLD:
            direction = "sharper" if avg_ratio > 1.0 else "smoother"
            anomaly = (
                f"Face boundary is {direction} than interior (ratio={avg_ratio:.3f}), possible compositing artifact"
            )

        return ("edge_gradient", score, anomaly)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        return frame[:, :, 0]
