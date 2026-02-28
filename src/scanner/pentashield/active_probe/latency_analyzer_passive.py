"""Scanner ULTRA — Latency analyzer for playback detection.

Analyzes frame timing characteristics to distinguish live video from pre-recorded/playback.
Real-time human responses show natural variability (200-600ms), while deepfake pipelines
introduce extra latency (>800ms) and often have uniform frame intervals (pre-rendered).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Result of latency analysis."""

    avg_frame_interval_ms: float = 0.0
    frame_jitter_ms: float = 0.0
    estimated_pipeline_latency_ms: float = 0.0
    is_realtime: bool = True
    latency_score: float = 1.0


class LatencyAnalyzer:
    """Analyzes frame timing for playback detection."""

    # Normal human response: 150-600ms
    HUMAN_RESPONSE_MIN_MS = 150
    HUMAN_RESPONSE_MAX_MS = 600

    # Deepfake pipeline typically adds >500ms latency
    SUSPICIOUS_LATENCY_MS = 800

    # Expected jitter range for live video (ms)
    MIN_JITTER_MS = 2.0  # Too uniform → suspicious
    MAX_JITTER_MS = 50.0  # Too variable → network issues or suspicious

    def analyze(
        self,
        frames: list[np.ndarray] | None,
        fps: float = 30.0,
        timestamps: list[float] | None = None,
    ) -> LatencyResult:
        """Analyze frame timing for playback detection.

        Args:
            frames: List of frames (used for count if timestamps not provided)
            fps: Frames per second (used to compute expected interval)
            timestamps: Optional list of frame timestamps (seconds). If None, assumes uniform.

        Returns:
            LatencyResult with timing analysis
        """
        if not frames or len(frames) < 2:
            logger.debug("Insufficient frames for latency analysis, returning default")
            return LatencyResult()

        # Frame intervals (ms)
        intervals_ms = self._compute_intervals(frames, fps, timestamps)

        if not intervals_ms or len(intervals_ms) < 2:
            return LatencyResult()

        # Average interval
        avg_interval = float(np.mean(intervals_ms))

        # Jitter (standard deviation of intervals)
        jitter = float(np.std(intervals_ms))

        # Estimate pipeline latency (deviation from expected 1/fps)
        expected_interval = 1000.0 / fps
        pipeline_latency = max(0.0, avg_interval - expected_interval)

        # Check if realtime
        is_realtime = self._is_realtime(avg_interval, jitter, fps)

        # Latency score (1.0 = normal, 0.0 = suspicious)
        latency_score = self._compute_score(avg_interval, jitter, pipeline_latency)

        return LatencyResult(
            avg_frame_interval_ms=round(avg_interval, 2),
            frame_jitter_ms=round(jitter, 2),
            estimated_pipeline_latency_ms=round(pipeline_latency, 2),
            is_realtime=is_realtime,
            latency_score=round(latency_score, 4),
        )

    @staticmethod
    def _compute_intervals(
        frames: list[np.ndarray],
        fps: float,
        timestamps: list[float] | None,
    ) -> list[float]:
        """Compute frame intervals in milliseconds.

        Args:
            frames: List of frames
            fps: Frames per second
            timestamps: Optional frame timestamps (seconds)

        Returns:
            List of intervals (ms) between consecutive frames
        """
        if timestamps and len(timestamps) >= 2:
            # Use provided timestamps
            intervals = []
            for i in range(len(timestamps) - 1):
                interval_ms = (timestamps[i + 1] - timestamps[i]) * 1000.0
                intervals.append(interval_ms)
            return intervals

        # Fallback: assume uniform fps
        expected_interval_ms = 1000.0 / fps
        num_intervals = len(frames) - 1
        return [expected_interval_ms] * num_intervals

    def _is_realtime(self, avg_interval: float, jitter: float, fps: float) -> bool:
        """Determine if video appears to be real-time.

        Args:
            avg_interval: Average frame interval (ms)
            jitter: Frame interval jitter (ms)
            fps: Frames per second

        Returns:
            True if appears real-time, False if likely playback
        """
        expected_interval = 1000.0 / fps

        # Check 1: Interval close to expected
        interval_deviation = abs(avg_interval - expected_interval)
        if interval_deviation > 100.0:  # >100ms deviation → suspicious
            return False

        # Check 2: Jitter in acceptable range
        if jitter < self.MIN_JITTER_MS:  # Too uniform → pre-rendered
            return False

        if jitter > self.MAX_JITTER_MS:  # Too variable → network/processing issues
            return False

        return True

    def _compute_score(self, avg_interval: float, jitter: float, pipeline_latency: float) -> float:
        """Compute overall latency score (1.0 = normal, 0.0 = suspicious).

        Args:
            avg_interval: Average frame interval (ms)
            jitter: Frame interval jitter (ms)
            pipeline_latency: Estimated pipeline latency (ms)

        Returns:
            Latency score (0.0 - 1.0)
        """
        score = 1.0

        # Penalty 1: High pipeline latency
        if pipeline_latency > self.SUSPICIOUS_LATENCY_MS:
            penalty = min(0.5, (pipeline_latency - self.SUSPICIOUS_LATENCY_MS) / 1000.0)
            score -= penalty

        # Penalty 2: Too uniform (low jitter)
        if jitter < self.MIN_JITTER_MS:
            penalty = min(0.3, (self.MIN_JITTER_MS - jitter) / self.MIN_JITTER_MS * 0.3)
            score -= penalty

        # Penalty 3: Too variable (high jitter)
        if jitter > self.MAX_JITTER_MS:
            penalty = min(0.2, (jitter - self.MAX_JITTER_MS) / 100.0)
            score -= penalty

        return max(0.0, score)
