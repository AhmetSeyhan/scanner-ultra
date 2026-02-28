"""Tests for PentaShieldEngine — integration of HYDRA + SENTINEL."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from scanner.models.enums import MediaType
from scanner.models.schemas import PentaShieldResult
from scanner.pentashield.engine import PentaShieldEngine


@pytest.fixture
def engine():
    return PentaShieldEngine()


@pytest.fixture
def clean_detector_results():
    """Detector results for clearly authentic content."""
    return {
        "clip_deepfake": {"score": 0.1, "confidence": 0.85, "details": {}},
        "frequency_analysis": {"score": 0.15, "confidence": 0.7, "details": {}},
        "efficientnet_b0": {"score": 0.12, "confidence": 0.8, "details": {}},
        "ppg_biosignal": {
            "score": 0.2,
            "confidence": 0.6,
            "details": {"periodicity": 0.6, "snr": 0.4, "spatial_consistency": 0.8},
        },
        "gaze_analysis": {
            "score": 0.15,
            "confidence": 0.7,
            "details": {"consistency": 0.8},
        },
    }


@pytest.fixture
def fake_detector_results():
    """Detector results for clearly fake content."""
    return {
        "clip_deepfake": {"score": 0.9, "confidence": 0.85, "details": {}},
        "frequency_analysis": {"score": 0.85, "confidence": 0.7, "details": {}},
        "gan_artifact": {"score": 0.88, "confidence": 0.75, "details": {}},
    }


class TestPentaShieldEngine:
    def test_analyze_clean_no_overrides(self, engine, clean_detector_results):
        """Clean authentic content without frames → no overrides."""
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=clean_detector_results,
                fused_score=0.15,
                fused_confidence=0.8,
                media_type=MediaType.VIDEO,
                frames=None,  # No frames → purifier skipped
            )
        )
        assert result.override_verdict is None
        assert result.override_reason is None
        assert not result.hydra.adversarial_detected

    def test_analyze_fake_no_override(self, engine, fake_detector_results):
        """Clearly fake content → HYDRA confirms, no override needed."""
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=fake_detector_results,
                fused_score=0.88,
                fused_confidence=0.8,
                media_type=MediaType.IMAGE,
            )
        )
        # HYDRA heads should agree it's fake
        assert result.hydra.consensus_score > 0.5
        # No override (fusion already correct)
        assert result.override_verdict is None

    def test_result_schema_match(self, engine, clean_detector_results):
        """Result should match PentaShieldResult schema."""
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=clean_detector_results,
                fused_score=0.15,
                fused_confidence=0.8,
                media_type=MediaType.IMAGE,
            )
        )
        assert isinstance(result, PentaShieldResult)
        # Should be serializable
        data = result.model_dump()
        assert "hydra" in data
        assert "sentinel" in data
        assert "processing_time_ms" in data

    def test_processing_time_tracked(self, engine, clean_detector_results):
        """Processing time should be positive."""
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=clean_detector_results,
                fused_score=0.5,
                fused_confidence=0.5,
                media_type=MediaType.IMAGE,
            )
        )
        assert result.processing_time_ms > 0

    def test_hydra_multi_head_populated(self, engine, clean_detector_results):
        """HYDRA should produce head verdicts."""
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=clean_detector_results,
                fused_score=0.15,
                fused_confidence=0.8,
                media_type=MediaType.VIDEO,
            )
        )
        assert len(result.hydra.head_verdicts) == 3
        assert 0.0 <= result.hydra.consensus_score <= 1.0
        assert 0.0 <= result.hydra.robustness_score <= 1.0

    def test_sentinel_populated(self, engine, clean_detector_results):
        """SENTINEL should produce OOD + physics + bio scores."""
        rng = np.random.RandomState(42)
        frames = [rng.randint(80, 180, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=clean_detector_results,
                fused_score=0.15,
                fused_confidence=0.8,
                media_type=MediaType.VIDEO,
                frames=frames,
            )
        )
        assert 0.0 <= result.sentinel.ood_score <= 1.0
        assert 0.0 <= result.sentinel.physics_score <= 1.0
        assert 0.0 <= result.sentinel.bio_consistency <= 1.0
        assert result.sentinel.alert_level in ("none", "low", "medium", "high", "critical")

    def test_no_frames_still_works(self, engine):
        """Engine should work without frames (audio/text only)."""
        results = {
            "cqt_spectral": {"score": 0.3, "confidence": 0.7, "details": {}},
            "wavlm_deepfake": {"score": 0.25, "confidence": 0.6, "details": {}},
        }
        result = asyncio.get_event_loop().run_until_complete(
            engine.analyze(
                detector_results=results,
                fused_score=0.28,
                fused_confidence=0.65,
                media_type=MediaType.AUDIO,
                frames=None,
            )
        )
        assert isinstance(result, PentaShieldResult)
        assert not result.hydra.purification_applied
        # Physics should default to 1.0 (no frames to check)
        assert result.sentinel.physics_score == 1.0
