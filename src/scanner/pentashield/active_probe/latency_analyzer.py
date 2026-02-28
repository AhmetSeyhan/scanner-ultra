"""Scanner ULTRA — Active Latency Analyzer.

Measures challenge-to-response latency in real-time streams.
Real humans respond within 50-200ms (webcam + network + reaction).
Deepfake pipelines add rendering latency (>800ms for model inference + encoding).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LatencyAnalysis:
    """Result of latency analysis for a challenge."""

    challenge_id: str
    challenge_sent_ms: int
    first_response_ms: int
    latency_ms: float
    is_realtime: bool
    confidence: float


class LatencyAnalyzer:
    """Analyzes challenge-response latency for liveness detection."""

    # Thresholds (milliseconds)
    REAL_HUMAN_MIN = 50  # Webcam lag + network
    REAL_HUMAN_MAX = 600  # Human reaction time + network
    DEEPFAKE_THRESHOLD = 800  # Deepfake rendering adds significant delay

    def analyze(
        self,
        challenge_id: str,
        challenge_sent_timestamp: int,
        response_timestamps: list[int],
    ) -> LatencyAnalysis:
        """Analyze latency from challenge sent to first response.

        Args:
            challenge_id: Unique challenge identifier
            challenge_sent_timestamp: Unix timestamp (ms) when challenge was sent
            response_timestamps: List of frame timestamps (ms) received after challenge

        Returns:
            LatencyAnalysis with timing verdict
        """
        if not response_timestamps:
            logger.warning("No response frames received for challenge %s", challenge_id)
            return LatencyAnalysis(
                challenge_id=challenge_id,
                challenge_sent_ms=challenge_sent_timestamp,
                first_response_ms=0,
                latency_ms=9999.0,
                is_realtime=False,
                confidence=0.0,
            )

        # First response after challenge
        first_response_ms = min(response_timestamps)

        # Latency calculation
        latency_ms = float(first_response_ms - challenge_sent_timestamp)

        # Realtime verdict
        is_realtime = self._is_realtime(latency_ms)

        # Confidence score (0.0 - 1.0)
        confidence = self._compute_confidence(latency_ms)

        return LatencyAnalysis(
            challenge_id=challenge_id,
            challenge_sent_ms=challenge_sent_timestamp,
            first_response_ms=first_response_ms,
            latency_ms=round(latency_ms, 2),
            is_realtime=is_realtime,
            confidence=round(confidence, 4),
        )

    def _is_realtime(self, latency_ms: float) -> bool:
        """Determine if latency indicates real-time human response.

        Args:
            latency_ms: Measured latency

        Returns:
            True if appears real-time, False if likely deepfake
        """
        # Too fast: suspicious (deepfake pre-rendered or bot)
        if latency_ms < self.REAL_HUMAN_MIN:
            return False

        # Normal range: real human
        if latency_ms <= self.REAL_HUMAN_MAX:
            return True

        # High latency: likely deepfake rendering
        if latency_ms > self.DEEPFAKE_THRESHOLD:
            return False

        # Borderline (600-800ms): uncertain, lean toward fake
        return False

    def _compute_confidence(self, latency_ms: float) -> float:
        """Compute confidence score for realtime verdict.

        Args:
            latency_ms: Measured latency

        Returns:
            Confidence score (0.0 - 1.0), 1.0 = definitely real
        """
        # Optimal range: 100-300ms (typical human + network)
        if 100 <= latency_ms <= 300:
            return 1.0

        # Acceptable range: 50-600ms
        if self.REAL_HUMAN_MIN <= latency_ms <= self.REAL_HUMAN_MAX:
            # Linear decay from center (200ms = 1.0)
            distance = abs(latency_ms - 200)
            return max(0.5, 1.0 - (distance / 400.0))

        # Too fast (<50ms): likely bot/pre-rendered
        if latency_ms < self.REAL_HUMAN_MIN:
            return 0.2

        # Too slow (>600ms): increasingly likely deepfake
        if latency_ms > self.REAL_HUMAN_MAX:
            # >800ms = 0.0 confidence
            excess = latency_ms - self.REAL_HUMAN_MAX
            return max(0.0, 0.5 - (excess / 400.0))

        return 0.5


class BatchLatencyAnalyzer:
    """Analyzes latency patterns across multiple challenges in a session."""

    def __init__(self) -> None:
        """Initialize batch analyzer."""
        self.analyzer = LatencyAnalyzer()

    def analyze_session(self, challenge_response_pairs: list[tuple[str, int, list[int]]]) -> dict:
        """Analyze latency across all challenges in a session.

        Args:
            challenge_response_pairs: List of (challenge_id, sent_ts, response_ts_list)

        Returns:
            Session-level latency analysis with statistics
        """
        analyses = []

        for challenge_id, sent_ts, response_ts in challenge_response_pairs:
            analysis = self.analyzer.analyze(challenge_id, sent_ts, response_ts)
            analyses.append(analysis)

        if not analyses:
            return {
                "num_challenges": 0,
                "avg_latency_ms": 0.0,
                "latency_std_ms": 0.0,
                "realtime_count": 0,
                "avg_confidence": 0.0,
                "session_verdict": "insufficient_data",
            }

        # Statistics
        latencies = [a.latency_ms for a in analyses]
        confidences = [a.confidence for a in analyses]
        realtime_count = sum(1 for a in analyses if a.is_realtime)

        avg_latency = sum(latencies) / len(latencies)
        import statistics

        latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        avg_confidence = sum(confidences) / len(confidences)

        # Session verdict (majority vote)
        if realtime_count >= len(analyses) / 2:
            session_verdict = "realtime"
        else:
            session_verdict = "deepfake_suspected"

        return {
            "num_challenges": len(analyses),
            "avg_latency_ms": round(avg_latency, 2),
            "latency_std_ms": round(latency_std, 2),
            "realtime_count": realtime_count,
            "avg_confidence": round(avg_confidence, 4),
            "session_verdict": session_verdict,
            "individual_analyses": [
                {
                    "challenge_id": a.challenge_id,
                    "latency_ms": a.latency_ms,
                    "is_realtime": a.is_realtime,
                    "confidence": a.confidence,
                }
                for a in analyses
            ],
        }
