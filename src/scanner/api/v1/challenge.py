"""Scanner ULTRA — Active Challenge API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scanner.pentashield.active_probe.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/challenge", tags=["Active Probe"])

# Global session manager (in production, use Redis/database)
session_manager = SessionManager()


class StartChallengeRequest(BaseModel):
    """Request to start a new challenge session."""

    num_challenges: int = 3
    challenge_types: list[str] | None = None


class ChallengeResponse(BaseModel):
    """Response for a challenge."""

    session_id: str
    challenge_id: str | None = None
    challenge_type: str | None = None
    instruction: str | None = None
    parameters: dict[str, Any] | None = None
    expected_duration_ms: int | None = None
    status: str
    challenges_total: int
    challenges_completed: int


class VerifyChallengeRequest(BaseModel):
    """Request to verify a challenge response."""

    challenge_id: str
    challenge_sent_ms: int
    response_timestamps: list[int]
    frame_sequence: list[tuple[list, int]] | None = None  # (frame_data, timestamp)
    baseline_frame: list | None = None


class VerifyChallengeResponse(BaseModel):
    """Response for challenge verification."""

    challenge_id: str
    passed: bool
    result: dict[str, Any]
    session_status: str
    challenges_remaining: int
    overall_passed: bool | None = None


@router.post("/start", response_model=ChallengeResponse)
async def start_challenge_session(request: StartChallengeRequest) -> ChallengeResponse:
    """Start a new active challenge session.

    Returns the first challenge to be executed.
    """
    try:
        session = session_manager.create_session(
            num_challenges=request.num_challenges,
            challenge_types=request.challenge_types or ["light", "motion"],
        )

        current_challenge = session_manager.get_current_challenge(session.session_id)

        if current_challenge is None:
            raise HTTPException(status_code=500, detail="Failed to generate challenges")

        return ChallengeResponse(
            session_id=session.session_id,
            challenge_id=current_challenge.challenge_id,
            challenge_type=current_challenge.challenge_type,
            instruction=current_challenge.instruction,
            parameters=current_challenge.parameters,
            expected_duration_ms=current_challenge.expected_duration_ms,
            status=session.status,
            challenges_total=len(session.challenges),
            challenges_completed=0,
        )

    except Exception as e:
        logger.exception("Failed to start challenge session")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{session_id}", response_model=ChallengeResponse)
async def get_challenge_session(session_id: str) -> ChallengeResponse:
    """Get current challenge for a session."""
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    current_challenge = session_manager.get_current_challenge(session_id)

    if current_challenge is None:
        # Session complete
        return ChallengeResponse(
            session_id=session_id,
            status=session.status,
            challenges_total=len(session.challenges),
            challenges_completed=len(session.challenge_results),
        )

    return ChallengeResponse(
        session_id=session_id,
        challenge_id=current_challenge.challenge_id,
        challenge_type=current_challenge.challenge_type,
        instruction=current_challenge.instruction,
        parameters=current_challenge.parameters,
        expected_duration_ms=current_challenge.expected_duration_ms,
        status=session.status,
        challenges_total=len(session.challenges),
        challenges_completed=len(session.challenge_results),
    )


@router.post("/{session_id}/verify", response_model=VerifyChallengeResponse)
async def verify_challenge_response(session_id: str, request: VerifyChallengeRequest) -> VerifyChallengeResponse:
    """Verify a challenge response and get next challenge or final result."""
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    try:
        # Submit response
        result = session_manager.submit_challenge_response(
            session_id=session_id,
            challenge_id=request.challenge_id,
            response_data={
                "challenge_sent_ms": request.challenge_sent_ms,
                "response_timestamps": request.response_timestamps,
                "frame_sequence": request.frame_sequence or [],
                "baseline_frame": request.baseline_frame,
            },
        )

        # Get updated session
        updated_session = session_manager.get_session(session_id)

        overall_passed = None
        if updated_session and updated_session.status == "completed":
            overall_passed = updated_session.overall_passed

        return VerifyChallengeResponse(
            challenge_id=request.challenge_id,
            passed=result.get("passed", False),
            result=result.get("result", {}),
            session_status=result.get("session_status", "unknown"),
            challenges_remaining=result.get("challenges_remaining", 0),
            overall_passed=overall_passed,
        )

    except Exception as e:
        logger.exception("Failed to verify challenge response")
        raise HTTPException(status_code=500, detail=str(e)) from e
