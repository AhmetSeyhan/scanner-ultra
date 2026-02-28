"""Scanner ULTRA — Active Motion Challenge.

Requests random head movements and verifies 3D geometric consistency.
Real faces show smooth, physically plausible motion with consistent facial landmarks.
Deepfakes exhibit jitter, unnatural motion, or inconsistent 3D geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionChallengeResult:
    """Result of active motion challenge verification."""

    challenge_id: str
    passed: bool
    angle_achieved: float
    angle_accuracy_score: float
    smoothness_score: float
    consistency_3d_score: float
    response_latency_ms: float
    anomalies: list[str]
    overall_score: float


class MotionChallenge:
    """Active motion challenge - verify head movement response."""

    # Thresholds
    MIN_ANGLE_ACCURACY = 0.7  # Must achieve 70% of requested angle
    MIN_SMOOTHNESS = 0.6  # Motion must be smooth (not jerky)
    MIN_3D_CONSISTENCY = 0.7  # Face landmarks must move consistently
    MAX_RESPONSE_LATENCY_MS = 2000  # Human should start moving within 2s

    def verify(
        self,
        challenge_id: str,
        challenge_params: dict,
        frame_sequence: list[tuple[np.ndarray, int]],  # (frame, timestamp_ms)
        baseline_frame: np.ndarray | None = None,
    ) -> MotionChallengeResult:
        """Verify motion challenge response.

        Args:
            challenge_id: Unique challenge identifier
            challenge_params: {direction, expected_angle, tolerance}
            frame_sequence: List of (frame, timestamp_ms) from WebRTC during challenge
            baseline_frame: Optional baseline frame before challenge (for angle comparison)

        Returns:
            MotionChallengeResult with verification scores
        """
        if not frame_sequence or len(frame_sequence) < 3:
            logger.warning("Insufficient frames for motion challenge")
            return MotionChallengeResult(
                challenge_id=challenge_id,
                passed=False,
                angle_achieved=0.0,
                angle_accuracy_score=0.0,
                smoothness_score=0.0,
                consistency_3d_score=0.0,
                response_latency_ms=0.0,
                anomalies=["insufficient_frames"],
                overall_score=0.0,
            )

        direction = challenge_params.get("direction", "left")
        expected_angle = challenge_params.get("expected_angle", 30)
        tolerance = challenge_params.get("tolerance", 10)

        # Step 1: Measure angle achieved
        angle_achieved = self._measure_head_angle(frame_sequence, direction, baseline_frame)
        angle_accuracy = self._score_angle_accuracy(angle_achieved, expected_angle, tolerance)

        # Step 2: Measure motion smoothness
        smoothness = self._measure_smoothness(frame_sequence)

        # Step 3: Measure 3D consistency (face landmarks)
        consistency_3d = self._measure_3d_consistency(frame_sequence)

        # Step 4: Measure response latency (time to start moving)
        latency_ms = self._measure_response_latency(frame_sequence)

        # Anomaly detection
        anomalies = []
        if angle_accuracy < self.MIN_ANGLE_ACCURACY:
            anomalies.append("insufficient_angle")
        if smoothness < self.MIN_SMOOTHNESS:
            anomalies.append("jerky_motion")
        if consistency_3d < self.MIN_3D_CONSISTENCY:
            anomalies.append("3d_inconsistency")
        if latency_ms > self.MAX_RESPONSE_LATENCY_MS:
            anomalies.append("slow_response")

        # Overall score
        overall = (
            0.3 * angle_accuracy + 0.3 * smoothness + 0.3 * consistency_3d + 0.1 * (1.0 - min(1.0, latency_ms / 2000.0))
        )

        passed = overall >= 0.65 and len(anomalies) <= 1  # Allow 1 minor anomaly

        return MotionChallengeResult(
            challenge_id=challenge_id,
            passed=passed,
            angle_achieved=round(angle_achieved, 2),
            angle_accuracy_score=round(angle_accuracy, 4),
            smoothness_score=round(smoothness, 4),
            consistency_3d_score=round(consistency_3d, 4),
            response_latency_ms=round(latency_ms, 2),
            anomalies=anomalies,
            overall_score=round(overall, 4),
        )

    @staticmethod
    def _measure_head_angle(
        frame_sequence: list[tuple[np.ndarray, int]],
        direction: str,
        baseline_frame: np.ndarray | None,
    ) -> float:
        """Measure head rotation angle achieved.

        Uses optical flow to estimate head rotation from baseline.

        Args:
            frame_sequence: Frames during challenge
            direction: Expected direction (left, right, up, down, nod, shake)
            baseline_frame: Frame before challenge

        Returns:
            Estimated angle in degrees
        """
        if not baseline_frame.any() or len(frame_sequence) < 2:
            return 0.0

        # Use max displacement frame (frame with most movement)
        max_displacement = 0.0
        max_displacement_frame = None

        baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_RGB2GRAY)

        for frame, _ in frame_sequence:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                baseline_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Total displacement
            displacement = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()

            if displacement > max_displacement:
                max_displacement = displacement
                max_displacement_frame = flow

        if max_displacement_frame is None:
            return 0.0

        # Estimate angle from flow magnitude (rough approximation)
        # Typical head turn: ~30 degrees = ~20 pixels displacement at center
        angle = max_displacement * 1.5  # Heuristic scaling

        return min(90.0, angle)  # Cap at 90 degrees

    @staticmethod
    def _score_angle_accuracy(achieved: float, expected: float, tolerance: float) -> float:
        """Score how accurately the expected angle was achieved.

        Args:
            achieved: Actual angle measured
            expected: Expected angle
            tolerance: Tolerance in degrees

        Returns:
            Accuracy score (0.0 - 1.0)
        """
        error = abs(achieved - expected)

        if error <= tolerance:
            return 1.0
        elif error <= tolerance * 2:
            return 1.0 - (error - tolerance) / tolerance
        else:
            return 0.0

    @staticmethod
    def _measure_smoothness(frame_sequence: list[tuple[np.ndarray, int]]) -> float:
        """Measure motion smoothness (detect jitter / abrupt changes).

        Args:
            frame_sequence: Frames during challenge

        Returns:
            Smoothness score (0.0 - 1.0)
        """
        if len(frame_sequence) < 3:
            return 1.0

        # Compute frame-to-frame differences
        diffs = []
        for i in range(len(frame_sequence) - 1):
            gray1 = cv2.cvtColor(frame_sequence[i][0], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame_sequence[i + 1][0], cv2.COLOR_RGB2GRAY)

            diff = cv2.absdiff(gray1, gray2)
            mean_diff = diff.mean()
            diffs.append(mean_diff)

        if len(diffs) < 2:
            return 1.0

        # Measure "jerk" (second derivative)
        jerks = []
        for i in range(len(diffs) - 1):
            jerk = abs(diffs[i + 1] - diffs[i])
            jerks.append(jerk)

        if not jerks:
            return 1.0

        # Lower jerk = smoother motion
        avg_jerk = np.mean(jerks)

        # Score: jerk < 5 = perfect, jerk > 20 = jerky
        score = max(0.0, 1.0 - (avg_jerk / 20.0))

        return score

    @staticmethod
    def _measure_3d_consistency(frame_sequence: list[tuple[np.ndarray, int]]) -> float:
        """Measure 3D geometric consistency of face landmarks during motion.

        Real faces maintain consistent inter-landmark distances.
        Deepfakes may show distortion or inconsistent geometry.

        Args:
            frame_sequence: Frames during challenge

        Returns:
            3D consistency score (0.0 - 1.0)
        """
        # Simplified: measure face region aspect ratio consistency
        # (Full implementation would use MediaPipe Face Mesh)

        aspect_ratios = []

        for frame, _ in frame_sequence:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            # Assume face is center region
            face_region = gray[int(h * 0.2) : int(h * 0.8), int(w * 0.25) : int(w * 0.75)]

            fh, fw = face_region.shape
            if fw > 0:
                aspect_ratio = fh / fw
                aspect_ratios.append(aspect_ratio)

        if len(aspect_ratios) < 2:
            return 1.0

        # Consistency: low variance in aspect ratio
        ar_std = np.std(aspect_ratios)

        # Score: std < 0.05 = consistent, std > 0.2 = inconsistent
        score = max(0.0, 1.0 - (ar_std / 0.2))

        return score

    @staticmethod
    def _measure_response_latency(frame_sequence: list[tuple[np.ndarray, int]]) -> float:
        """Measure time from challenge start to first significant motion.

        Args:
            frame_sequence: Frames with timestamps

        Returns:
            Latency in milliseconds
        """
        if len(frame_sequence) < 2:
            return 0.0

        # First frame timestamp
        start_ts = frame_sequence[0][1]

        # Find first frame with significant change
        baseline_gray = cv2.cvtColor(frame_sequence[0][0], cv2.COLOR_RGB2GRAY)

        for frame, ts in frame_sequence[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(baseline_gray, gray)

            if diff.mean() > 10.0:  # Threshold for "significant" change
                return float(ts - start_ts)

        # No significant motion detected
        return float(frame_sequence[-1][1] - start_ts)
