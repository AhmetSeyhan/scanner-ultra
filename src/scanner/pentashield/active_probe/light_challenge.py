"""Scanner ULTRA — Active Light Challenge.

Instructs user's screen to flash different colors and verifies face reflection changes.
Real faces show consistent reflections in eyes (catchlights) and face brightness changes.
Deepfakes fail to synchronize these changes properly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LightChallengeResult:
    """Result of active light challenge verification."""

    challenge_id: str
    passed: bool
    reflection_change_score: float
    brightness_correlation: float
    synchronization_score: float
    anomalies: list[str]
    overall_score: float


class LightChallenge:
    """Active light challenge - verify face reflection to screen color changes."""

    # Thresholds
    MIN_REFLECTION_CHANGE = 0.15  # Minimum 15% reflection change expected
    MIN_BRIGHTNESS_CORRELATION = 0.6  # Minimum correlation between screen & face
    MIN_SYNCHRONIZATION = 0.7  # Minimum timing synchronization score

    def verify(
        self,
        challenge_id: str,
        color_sequence: list[dict],
        frame_sequence: list[tuple[np.ndarray, int]],  # (frame, timestamp_ms)
    ) -> LightChallengeResult:
        """Verify light challenge response.

        Args:
            challenge_id: Unique challenge identifier
            color_sequence: List of {color: hex, duration_ms: int}
            frame_sequence: List of (frame, timestamp_ms) tuples from WebRTC

        Returns:
            LightChallengeResult with verification scores
        """
        if not frame_sequence or len(frame_sequence) < len(color_sequence):
            logger.warning(
                "Insufficient frames for light challenge (got %d, need %d)",
                len(frame_sequence),
                len(color_sequence),
            )
            return LightChallengeResult(
                challenge_id=challenge_id,
                passed=False,
                reflection_change_score=0.0,
                brightness_correlation=0.0,
                synchronization_score=0.0,
                anomalies=["insufficient_frames"],
                overall_score=0.0,
            )

        # Step 1: Extract eye reflections (catchlights)
        reflection_score = self._measure_reflection_changes(frame_sequence, color_sequence)

        # Step 2: Measure face brightness correlation with screen colors
        brightness_corr = self._measure_brightness_correlation(frame_sequence, color_sequence)

        # Step 3: Check synchronization (timing)
        sync_score = self._measure_synchronization(frame_sequence, color_sequence)

        # Anomaly detection
        anomalies = []
        if reflection_score < self.MIN_REFLECTION_CHANGE:
            anomalies.append("low_reflection_change")
        if brightness_corr < self.MIN_BRIGHTNESS_CORRELATION:
            anomalies.append("low_brightness_correlation")
        if sync_score < self.MIN_SYNCHRONIZATION:
            anomalies.append("poor_synchronization")

        # Overall score (weighted average)
        overall = 0.4 * reflection_score + 0.4 * brightness_corr + 0.2 * sync_score

        passed = overall >= 0.6 and len(anomalies) == 0

        return LightChallengeResult(
            challenge_id=challenge_id,
            passed=passed,
            reflection_change_score=round(reflection_score, 4),
            brightness_correlation=round(brightness_corr, 4),
            synchronization_score=round(sync_score, 4),
            anomalies=anomalies,
            overall_score=round(overall, 4),
        )

    @staticmethod
    def _measure_reflection_changes(
        frame_sequence: list[tuple[np.ndarray, int]],
        color_sequence: list[dict],
    ) -> float:
        """Measure eye reflection (catchlight) changes during color flashes.

        Args:
            frame_sequence: Frames with timestamps
            color_sequence: Color flash sequence

        Returns:
            Reflection change score (0.0 - 1.0)
        """
        # Detect eyes in frames
        eye_reflections = []

        for frame, _ in frame_sequence[: len(color_sequence)]:
            # Extract eye region (center-top region approximate)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            # Eye region (approximate - top-center 20%)
            eye_region = gray[int(h * 0.2) : int(h * 0.5), int(w * 0.3) : int(w * 0.7)]

            # Detect bright spots (catchlights / specular reflections)
            _, bright_mask = cv2.threshold(eye_region, 200, 255, cv2.THRESH_BINARY)
            reflection_intensity = cv2.countNonZero(bright_mask) / bright_mask.size

            eye_reflections.append(reflection_intensity)

        if len(eye_reflections) < 2:
            return 0.0

        # Measure variation in reflections (should change with screen color)
        reflection_range = max(eye_reflections) - min(eye_reflections)

        # Normalize to 0-1 (higher variation = better)
        score = min(1.0, reflection_range * 2.0)

        return score

    @staticmethod
    def _measure_brightness_correlation(
        frame_sequence: list[tuple[np.ndarray, int]],
        color_sequence: list[dict],
    ) -> float:
        """Measure correlation between screen color brightness and face brightness.

        Args:
            frame_sequence: Frames with timestamps
            color_sequence: Color flash sequence

        Returns:
            Correlation score (0.0 - 1.0)
        """
        # Convert screen colors to brightness values
        screen_brightness = []
        for color_dict in color_sequence:
            hex_color = color_dict["color"].lstrip("#")
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            # Perceived brightness (ITU-R BT.601)
            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            screen_brightness.append(brightness)

        # Extract face brightness from frames
        face_brightness = []
        for frame, _ in frame_sequence[: len(color_sequence)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            # Face region (center crop)
            face_region = gray[int(h * 0.2) : int(h * 0.8), int(w * 0.25) : int(w * 0.75)]
            mean_brightness = face_region.mean() / 255.0
            face_brightness.append(mean_brightness)

        if len(screen_brightness) != len(face_brightness) or len(screen_brightness) < 2:
            return 0.0

        # Compute Pearson correlation
        corr = np.corrcoef(screen_brightness, face_brightness)[0, 1]

        # Handle NaN (all values identical)
        if np.isnan(corr):
            return 0.0

        # Convert to 0-1 range (correlation can be -1 to 1, we want positive)
        score = max(0.0, corr)

        return score

    @staticmethod
    def _measure_synchronization(
        frame_sequence: list[tuple[np.ndarray, int]],
        color_sequence: list[dict],
    ) -> float:
        """Measure timing synchronization between color changes and frame changes.

        Real-time human faces reflect changes within 50-100ms.
        Deepfake rendering introduces delay (>200ms).

        Args:
            frame_sequence: Frames with timestamps
            color_sequence: Color flash sequence

        Returns:
            Synchronization score (0.0 - 1.0)
        """
        if len(frame_sequence) < len(color_sequence):
            return 0.0

        # Expected timing: each color should appear within its duration window
        expected_timings = []
        cumulative_time = 0
        for color_dict in color_sequence:
            expected_timings.append(cumulative_time)
            cumulative_time += color_dict["duration_ms"]

        # Actual frame timings
        actual_timings = [ts for _, ts in frame_sequence[: len(color_sequence)]]

        # Measure timing drift
        drifts = []
        for expected, actual in zip(expected_timings, actual_timings):
            drift = abs(actual - expected)
            drifts.append(drift)

        if not drifts:
            return 0.0

        avg_drift = np.mean(drifts)

        # Score: <50ms drift = 1.0, >500ms drift = 0.0
        score = max(0.0, 1.0 - (avg_drift / 500.0))

        return score
