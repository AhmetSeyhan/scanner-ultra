"""Scanner ULTRA — Probe session coordinator for ACTIVE PROBE.

Coordinates light challenge, motion challenge, and latency analysis based on media type.
Produces liveness verdict (live, suspicious, playback, or not_applicable).

NOTE: This uses PASSIVE probe modules for offline video/image analysis.
For real-time ACTIVE challenge-response, use SessionManager with active modules.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.models.enums import MediaType
from scanner.models.schemas import ActiveProbeResult

# Import PASSIVE probe modules (for offline analysis)
from scanner.pentashield.active_probe.latency_analyzer_passive import LatencyAnalyzer
from scanner.pentashield.active_probe.light_challenge_passive import LightChallenge
from scanner.pentashield.active_probe.motion_challenge_passive import MotionChallenge

logger = logging.getLogger(__name__)


class ProbeSession:
    """Coordinates passive probe analysis for offline video/image."""

    def __init__(self) -> None:
        """Initialize passive challenge modules."""
        self.light = LightChallenge()
        self.motion = MotionChallenge()
        self.latency = LatencyAnalyzer()

    def run(
        self,
        media_type: MediaType,
        frames: list[np.ndarray] | None = None,
        fps: float = 30.0,
        timestamps: list[float] | None = None,
    ) -> ActiveProbeResult:
        """Run passive probe analysis based on media type.

        Challenge strategy by media type:
          - STREAM/VIDEO: All challenges (light, motion, latency)
          - IMAGE: Limited light challenge only (single frame)
          - AUDIO/TEXT: Not applicable

        Args:
            media_type: Type of media being analyzed
            frames: Optional video/image frames
            fps: Frames per second for video
            timestamps: Optional frame timestamps for latency analysis

        Returns:
            ActiveProbeResult with liveness verdict and challenge details
        """
        # Check if probe is applicable
        if media_type in (MediaType.AUDIO, MediaType.TEXT):
            logger.debug(f"Active probe not applicable for {media_type}")
            return ActiveProbeResult(probe_available=False)

        # Run challenges based on media type
        challenge_results = []
        challenges_run = 0

        if media_type in (MediaType.VIDEO, MediaType.STREAM):
            # Full probe: all challenges
            light_result = self.light.evaluate(frames)
            motion_result = self.motion.evaluate(frames)
            latency_result = self.latency.analyze(frames, fps, timestamps)

            challenge_results.extend(
                [
                    {"challenge": "light", "result": light_result.__dict__},
                    {"challenge": "motion", "result": motion_result.__dict__},
                    {"challenge": "latency", "result": latency_result.__dict__},
                ]
            )
            challenges_run = 3

        elif media_type == MediaType.IMAGE:
            # Limited probe: light challenge only (single frame analysis)
            light_result = self.light.evaluate(frames)
            challenge_results.append({"challenge": "light", "result": light_result.__dict__})
            challenges_run = 1

        # Evaluate liveness
        liveness_score, verdict = self._evaluate_liveness(challenge_results, media_type)

        # Latency analysis dict (for API)
        latency_analysis: dict[str, Any] = {}
        if media_type in (MediaType.VIDEO, MediaType.STREAM):
            latency_analysis = latency_result.__dict__ if "latency_result" in locals() else {}

        return ActiveProbeResult(
            probe_available=True,
            challenges_run=challenges_run,
            challenge_results=challenge_results,
            latency_analysis=latency_analysis,
            liveness_score=round(liveness_score, 4),
            probe_verdict=verdict,
        )

    @staticmethod
    def _evaluate_liveness(challenge_results: list[dict[str, Any]], media_type: MediaType) -> tuple[float, str]:
        """Evaluate liveness from challenge results.

        Args:
            challenge_results: List of challenge result dicts
            media_type: Type of media

        Returns:
            (liveness_score, verdict)
            - liveness_score: 0.0 (playback/fake) to 1.0 (live/real)
            - verdict: "live", "suspicious", "playback", or "not_applicable"
        """
        if not challenge_results:
            return (1.0, "not_applicable")

        # Collect scores
        scores = []
        for result_dict in challenge_results:
            result = result_dict.get("result", {})
            challenge_type = result_dict.get("challenge")

            if challenge_type == "light":
                scores.append(result.get("overall_score", 1.0))
            elif challenge_type == "motion":
                scores.append(result.get("overall_score", 1.0))
            elif challenge_type == "latency":
                scores.append(result.get("latency_score", 1.0))

        if not scores:
            return (1.0, "not_applicable")

        # Average liveness score
        avg_score = float(np.mean(scores))

        # Determine verdict
        if media_type == MediaType.IMAGE:
            # Image: only light challenge, less conclusive
            if avg_score >= 0.7:
                verdict = "live"
            elif avg_score >= 0.4:
                verdict = "suspicious"
            else:
                verdict = "playback"
        else:
            # Video/Stream: full probe
            if avg_score >= 0.8:
                verdict = "live"
            elif avg_score >= 0.5:
                verdict = "suspicious"
            else:
                verdict = "playback"

        return (avg_score, verdict)
