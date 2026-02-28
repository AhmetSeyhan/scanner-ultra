"""Scanner ULTRA — Challenge Session Manager.

Manages active probe challenge sessions:
  - Session creation (generate random challenges)
  - Session tracking (store challenges + responses)
  - Session verification (validate all challenges passed)
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from scanner.pentashield.active_probe.challenge_protocol import (
    Challenge,
    ChallengeProtocol,
)
from scanner.pentashield.active_probe.latency_analyzer import (
    BatchLatencyAnalyzer,
    LatencyAnalysis,
)
from scanner.pentashield.active_probe.light_challenge import (
    LightChallenge,
)
from scanner.pentashield.active_probe.motion_challenge import (
    MotionChallenge,
)

logger = logging.getLogger(__name__)


@dataclass
class ChallengeSession:
    """A single active probe session with multiple challenges."""

    session_id: str
    created_at: int  # Unix timestamp (ms)
    challenges: list[Challenge]
    current_challenge_index: int = 0
    challenge_results: dict[str, Any] = field(default_factory=dict)  # challenge_id → result
    latency_analyses: dict[str, LatencyAnalysis] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    overall_passed: bool = False
    expires_at: int = 0  # Unix timestamp (ms)


class SessionManager:
    """Manages active probe challenge sessions."""

    SESSION_TIMEOUT_MS = 120_000  # 2 minutes per session
    MAX_ACTIVE_SESSIONS = 1000  # Memory limit

    def __init__(self) -> None:
        """Initialize session manager."""
        self.sessions: dict[str, ChallengeSession] = {}
        self.protocol = ChallengeProtocol()
        self.light_verifier = LightChallenge()
        self.motion_verifier = MotionChallenge()
        self.latency_analyzer = BatchLatencyAnalyzer()

    def create_session(
        self,
        num_challenges: int = 3,
        challenge_types: list[str] | None = None,
    ) -> ChallengeSession:
        """Create a new challenge session.

        Args:
            num_challenges: Number of challenges in session (default 3)
            challenge_types: Types to include (default: ["light", "motion"])

        Returns:
            ChallengeSession object with generated challenges
        """
        # Clean up expired sessions
        self._cleanup_expired_sessions()

        # Generate session ID
        session_id = f"session_{secrets.token_hex(16)}"

        # Generate random challenges
        challenges = self.protocol.generate_session_challenges(num_challenges, challenge_types)

        # Create session
        now = int(time.time() * 1000)
        session = ChallengeSession(
            session_id=session_id,
            created_at=now,
            challenges=challenges,
            expires_at=now + self.SESSION_TIMEOUT_MS,
        )

        # Store session
        if len(self.sessions) >= self.MAX_ACTIVE_SESSIONS:
            # Evict oldest session
            oldest_id = min(self.sessions.keys(), key=lambda k: self.sessions[k].created_at)
            del self.sessions[oldest_id]

        self.sessions[session_id] = session

        logger.info(
            "Created challenge session %s with %d challenges",
            session_id,
            len(challenges),
        )

        return session

    def get_session(self, session_id: str) -> ChallengeSession | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChallengeSession or None if not found/expired
        """
        session = self.sessions.get(session_id)

        if session is None:
            return None

        # Check if expired
        now = int(time.time() * 1000)
        if now > session.expires_at:
            logger.warning("Session %s expired", session_id)
            del self.sessions[session_id]
            return None

        return session

    def get_current_challenge(self, session_id: str) -> Challenge | None:
        """Get the current active challenge for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current Challenge or None if session complete/not found
        """
        session = self.get_session(session_id)

        if session is None:
            return None

        if session.current_challenge_index >= len(session.challenges):
            return None

        return session.challenges[session.current_challenge_index]

    def submit_challenge_response(
        self,
        session_id: str,
        challenge_id: str,
        response_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit response for a challenge and verify it.

        Args:
            session_id: Session identifier
            challenge_id: Challenge identifier
            response_data: Response data from client {frames, timestamps, etc.}

        Returns:
            Verification result dict
        """
        session = self.get_session(session_id)

        if session is None:
            return {"error": "session_not_found", "passed": False}

        # Find challenge
        challenge = next(
            (c for c in session.challenges if c.challenge_id == challenge_id),
            None,
        )

        if challenge is None:
            return {"error": "challenge_not_found", "passed": False}

        # Update session status
        if session.status == "pending":
            session.status = "in_progress"

        # Verify challenge based on type
        if challenge.challenge_type == "light":
            result = self._verify_light_challenge(challenge, response_data)
        elif challenge.challenge_type == "motion":
            result = self._verify_motion_challenge(challenge, response_data)
        elif challenge.challenge_type == "audio":
            result = self._verify_audio_challenge(challenge, response_data)
        else:
            return {"error": "unknown_challenge_type", "passed": False}

        # Store result
        session.challenge_results[challenge_id] = result

        # Analyze latency
        if "challenge_sent_ms" in response_data and "response_timestamps" in response_data:
            latency = self.latency_analyzer.analyzer.analyze(
                challenge_id,
                response_data["challenge_sent_ms"],
                response_data["response_timestamps"],
            )
            session.latency_analyses[challenge_id] = latency

        # Move to next challenge
        session.current_challenge_index += 1

        # Check if session complete
        if session.current_challenge_index >= len(session.challenges):
            self._finalize_session(session)

        return {
            "challenge_id": challenge_id,
            "passed": result.get("passed", False),
            "result": result,
            "session_status": session.status,
            "challenges_remaining": len(session.challenges) - session.current_challenge_index,
        }

    def _verify_light_challenge(self, challenge: Challenge, response_data: dict) -> dict[str, Any]:
        """Verify light challenge response."""
        frame_sequence = response_data.get("frame_sequence", [])
        color_sequence = challenge.parameters.get("sequence", [])

        result = self.light_verifier.verify(challenge.challenge_id, color_sequence, frame_sequence)

        return result.__dict__

    def _verify_motion_challenge(self, challenge: Challenge, response_data: dict) -> dict[str, Any]:
        """Verify motion challenge response."""
        frame_sequence = response_data.get("frame_sequence", [])
        baseline_frame = response_data.get("baseline_frame")

        result = self.motion_verifier.verify(
            challenge.challenge_id,
            challenge.parameters,
            frame_sequence,
            baseline_frame,
        )

        return result.__dict__

    @staticmethod
    def _verify_audio_challenge(challenge: Challenge, response_data: dict) -> dict[str, Any]:
        """Verify audio challenge response (placeholder)."""
        # TODO: Implement audio verification (speech-to-text + matching)
        return {
            "challenge_id": challenge.challenge_id,
            "passed": False,
            "error": "audio_verification_not_implemented",
        }

    def _finalize_session(self, session: ChallengeSession) -> None:
        """Finalize session after all challenges complete."""
        # Count passed challenges
        passed_count = sum(1 for result in session.challenge_results.values() if result.get("passed", False))

        # Session passes if majority of challenges passed
        required_passes = len(session.challenges) // 2 + 1
        session.overall_passed = passed_count >= required_passes

        # Analyze latencies
        if session.latency_analyses:
            pairs = [
                (
                    la.challenge_id,
                    la.challenge_sent_ms,
                    [la.first_response_ms],
                )
                for la in session.latency_analyses.values()
            ]
            latency_summary = self.latency_analyzer.analyze_session(pairs)

            # Override if latency indicates deepfake
            if latency_summary["session_verdict"] == "deepfake_suspected":
                session.overall_passed = False

        session.status = "completed" if session.overall_passed else "failed"

        logger.info(
            "Session %s finalized: %s (passed %d/%d challenges)",
            session.session_id,
            session.status,
            passed_count,
            len(session.challenges),
        )

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from memory."""
        now = int(time.time() * 1000)
        expired = [sid for sid, s in self.sessions.items() if now > s.expires_at]

        for sid in expired:
            del self.sessions[sid]

        if expired:
            logger.debug("Cleaned up %d expired sessions", len(expired))
