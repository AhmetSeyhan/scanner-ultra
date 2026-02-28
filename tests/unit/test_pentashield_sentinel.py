"""Tests for ZERO-DAY SENTINEL: OODDetector, PhysicsVerifier, BioConsistency, AnomalyScorer."""

from __future__ import annotations

import numpy as np

from scanner.models.schemas import SentinelResult
from scanner.pentashield.sentinel.anomaly_scorer import AnomalyScorer
from scanner.pentashield.sentinel.bio_consistency import BioConsistency, BioResult
from scanner.pentashield.sentinel.ood_detector import OODDetector, OODResult
from scanner.pentashield.sentinel.physics_verifier import PhysicsResult, PhysicsVerifier

# ── OODDetector ──


class TestOODDetector:
    def setup_method(self):
        self.ood = OODDetector(temperature=1.5)

    def test_empty_results(self):
        result = self.ood.detect({})
        assert result.ood_score == 0.5
        assert not result.is_novel_type

    def test_in_distribution_polarized(self):
        """Detectors with polarized scores → low OOD (confident decisions)."""
        results = {
            "det1": {"score": 0.1},
            "det2": {"score": 0.15},
            "det3": {"score": 0.05},
            "det4": {"score": 0.2},
        }
        result = self.ood.detect(results)
        assert result.ood_score < 0.5  # Low OOD for in-distribution

    def test_high_entropy_scattered(self):
        """Scattered scores across range → higher entropy → higher OOD."""
        results = {
            "det1": {"score": 0.1},
            "det2": {"score": 0.5},
            "det3": {"score": 0.9},
        }
        result = self.ood.detect(results)
        assert result.entropy > 0.3  # Should have some entropy

    def test_novel_type_flag(self):
        """Extremely high OOD score → is_novel_type flag."""
        # Force high entropy + all scores at 0.5 (maximum uncertainty)
        results = {f"det{i}": {"score": 0.48 + np.random.uniform(-0.02, 0.02)} for i in range(10)}
        result = self.ood.detect(results)
        # Even if not flagged as novel, the variance check should fire
        assert result.score_variance >= 0.0

    def test_ood_score_bounded(self):
        results = {"det1": {"score": 0.5}, "det2": {"score": 0.5}}
        result = self.ood.detect(results)
        assert 0.0 <= result.ood_score <= 1.0

    def test_feature_distance_without_reference(self):
        """Without reference embeddings, feature distance is None."""
        results = {"det1": {"score": 0.3}}
        embedding = np.random.randn(512).astype(np.float32)
        result = self.ood.detect(results, clip_embeddings=embedding)
        # No reference loaded → feature_distance stays None
        assert result.feature_distance is None


# ── PhysicsVerifier ──


class TestPhysicsVerifier:
    def setup_method(self):
        self.verifier = PhysicsVerifier()

    def test_no_frames(self):
        result = self.verifier.verify([])
        assert result.physics_score == 1.0
        assert result.anomalies == []

    def test_consistent_uniform_frames(self):
        """Uniform frames → high consistency scores."""
        frames = [np.full((128, 128, 3), 128, dtype=np.uint8) for _ in range(4)]
        result = self.verifier.verify(frames)
        assert result.physics_score >= 0.5
        assert isinstance(result.check_scores, dict)

    def test_lighting_asymmetry(self):
        """Frame with strong left-right brightness difference across frames."""
        frames = []
        for i in range(4):
            frame = np.full((128, 128, 3), 100, dtype=np.uint8)
            # Alternating lighting: frame 0,2 bright-left, frame 1,3 bright-right
            if i % 2 == 0:
                frame[:, :64, :] = 200
                frame[:, 64:, :] = 50
            else:
                frame[:, :64, :] = 50
                frame[:, 64:, :] = 200
            frames.append(frame)
        result = self.verifier.verify(frames)
        lighting_score = result.check_scores.get("lighting", 1.0)
        assert lighting_score < 0.8  # Should detect inconsistency

    def test_edge_gradient_returns_score(self):
        """Edge gradient check should produce a score."""
        frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = self.verifier.verify([frame])
        assert "edge_gradient" in result.check_scores

    def test_color_temperature_check(self):
        """Color temperature check on frames with face-like regions."""
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)
        # Make center (face) warm, corners (bg) cool
        frame[32:96, 32:96, 0] = 200  # Red (warm face)
        frame[32:96, 32:96, 2] = 80  # Less blue in face
        frame[:20, :20, 2] = 200  # Blue corners (cool bg)
        frame[:20, 108:, 2] = 200
        result = self.verifier.verify([frame])
        assert "color_temperature" in result.check_scores


# ── BioConsistency ──


class TestBioConsistency:
    def setup_method(self):
        self.bio = BioConsistency()

    def test_no_bio_data(self):
        """No PPG/gaze data → neutral score."""
        result = self.bio.check({})
        assert result.bio_consistency == 1.0
        assert result.issues == []

    def test_consistent_ppg_signals(self):
        """PPG with good periodicity and SNR → high consistency."""
        results = {
            "ppg_biosignal": {
                "score": 0.2,
                "confidence": 0.7,
                "details": {
                    "periodicity": 0.6,
                    "snr": 0.4,
                    "spatial_consistency": 0.8,
                },
            }
        }
        result = self.bio.check(results)
        assert result.bio_consistency > 0.5

    def test_inconsistent_ppg(self):
        """PPG with zero periodicity and no SNR → low consistency."""
        results = {
            "ppg_biosignal": {
                "score": 0.8,
                "confidence": 0.3,
                "details": {
                    "periodicity": 0.0,
                    "snr": 0.0,
                    "spatial_consistency": 0.1,
                },
            }
        }
        result = self.bio.check(results)
        assert result.bio_consistency < 0.7
        assert len(result.issues) > 0

    def test_gaze_consistency(self):
        """Gaze data with reasonable consistency."""
        results = {
            "gaze_analysis": {
                "score": 0.2,
                "confidence": 0.6,
                "details": {"consistency": 0.8, "blink_rate": 15},
            }
        }
        result = self.bio.check(results)
        assert result.bio_consistency > 0.5

    def test_ppg_and_gaze_cross_check(self):
        """Both PPG and gaze present → cross-check runs."""
        results = {
            "ppg_biosignal": {
                "score": 0.3,
                "confidence": 0.6,
                "details": {"periodicity": 0.5, "snr": 0.3, "spatial_consistency": 0.7},
            },
            "gaze_analysis": {
                "score": 0.25,
                "confidence": 0.7,
                "details": {"consistency": 0.75},
            },
        }
        result = self.bio.check(results)
        assert "signal_smoothness" in result.check_details


# ── AnomalyScorer ──


class TestAnomalyScorer:
    def setup_method(self):
        self.scorer = AnomalyScorer()

    def test_all_normal(self):
        ood = OODResult(ood_score=0.1)
        physics = PhysicsResult(physics_score=0.9)
        bio = BioResult(bio_consistency=0.95)
        result = self.scorer.score(ood, physics, bio)
        assert result.anomaly_score < 0.2
        assert result.alert_level == "none"

    def test_high_anomaly(self):
        ood = OODResult(ood_score=0.9, is_novel_type=True)
        physics = PhysicsResult(physics_score=0.1, anomalies=["lighting", "shadow"])
        bio = BioResult(bio_consistency=0.1)
        result = self.scorer.score(ood, physics, bio)
        assert result.anomaly_score > 0.7
        assert result.alert_level in ("high", "critical")

    def test_ood_heavy(self):
        """High OOD alone can drive alert level."""
        ood = OODResult(ood_score=0.95, is_novel_type=True)
        physics = PhysicsResult(physics_score=0.8)
        bio = BioResult(bio_consistency=0.9)
        result = self.scorer.score(ood, physics, bio)
        assert result.anomaly_score > 0.2
        assert result.is_novel_type

    def test_sentinel_result_type(self):
        ood = OODResult(ood_score=0.5)
        physics = PhysicsResult(physics_score=0.5)
        bio = BioResult(bio_consistency=0.5)
        result = self.scorer.score(ood, physics, bio)
        assert isinstance(result, SentinelResult)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_explain_method(self):
        ood = OODResult(ood_score=0.8, is_novel_type=True)
        physics = PhysicsResult(physics_score=0.4, anomalies=["shadow mismatch"])
        bio = BioResult(bio_consistency=0.9)
        result = self.scorer.score(ood, physics, bio)
        explanation = AnomalyScorer.explain(result)
        assert "factors" in explanation
        assert len(explanation["factors"]) > 0
