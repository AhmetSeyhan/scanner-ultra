"""Tests for HYDRA ENGINE: InputPurifier, MultiHeadEnsemble, MinorityReport, AdversarialAuditor."""

from __future__ import annotations

import numpy as np

from scanner.models.enums import MediaType
from scanner.pentashield.hydra.adversarial_auditor import AdversarialAuditor
from scanner.pentashield.hydra.input_purifier import InputPurifier
from scanner.pentashield.hydra.minority_report import MinorityReport
from scanner.pentashield.hydra.multi_head_ensemble import MultiHeadEnsemble

# ── InputPurifier ──


class TestInputPurifier:
    def setup_method(self):
        self.purifier = InputPurifier(sigma=0.8, jpeg_quality=75, bit_depth=4)

    def test_empty_frames(self):
        result = self.purifier.purify([])
        assert not result.adversarial_detected
        assert result.purified_frames == []

    def test_spatial_smoothing_preserves_shape(self):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = self.purifier.purify([frame])
        assert len(result.purified_frames) == 1
        assert result.purified_frames[0].shape == (64, 64, 3)

    def test_jpeg_defense_preserves_content(self):
        """JPEG defense should not drastically alter clean content."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = self.purifier.purify([frame])
        diff = np.abs(result.purified_frames[0].astype(float) - frame.astype(float))
        assert diff.mean() < 30  # JPEG artifacts are small

    def test_adversarial_detection_with_noise(self):
        """High-frequency adversarial noise → should be detected."""
        clean = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Add high-frequency noise (adversarial perturbation)
        noise = np.random.normal(0, 25, clean.shape).astype(np.int16)
        noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = self.purifier.purify([noisy])
        # The perturbation magnitude should be higher than clean
        assert result.perturbation_magnitude > 0.01

    def test_clean_input_low_perturbation(self):
        """Clean input → low perturbation magnitude."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = self.purifier.purify([frame])
        # Uniform content → very low perturbation after smoothing/JPEG
        assert result.perturbation_magnitude < 0.05

    def test_multiple_frames(self):
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
        result = self.purifier.purify(frames)
        assert len(result.purified_frames) == 5
        assert len(result.per_frame_magnitudes) == 5
        assert "spatial_smooth" in result.method_applied


# ── MultiHeadEnsemble ──


class TestMultiHeadEnsemble:
    def setup_method(self):
        self.ensemble = MultiHeadEnsemble()

    def test_empty_results(self):
        result = self.ensemble.evaluate({}, MediaType.IMAGE, 0.5)
        assert result.consensus_score == 0.5
        assert result.agreement == 1.0

    def test_unanimous_fake(self):
        results = {
            "clip_deepfake": {"score": 0.9, "confidence": 0.8},
            "frequency_analysis": {"score": 0.85, "confidence": 0.7},
            "gan_artifact": {"score": 0.88, "confidence": 0.75},
        }
        result = self.ensemble.evaluate(results, MediaType.IMAGE, 0.87)
        assert result.consensus_score > 0.7
        for v in result.head_verdicts:
            assert v > 0.5  # All heads agree: fake

    def test_unanimous_authentic(self):
        results = {
            "clip_deepfake": {"score": 0.1, "confidence": 0.85},
            "frequency_analysis": {"score": 0.15, "confidence": 0.7},
            "gan_artifact": {"score": 0.12, "confidence": 0.75},
        }
        result = self.ensemble.evaluate(results, MediaType.IMAGE, 0.12)
        assert result.consensus_score < 0.4
        assert result.agreement > 0.7

    def test_mixed_results(self):
        results = {
            "clip_deepfake": {"score": 0.9, "confidence": 0.8},
            "frequency_analysis": {"score": 0.2, "confidence": 0.7},
            "gan_artifact": {"score": 0.5, "confidence": 0.6},
        }
        result = self.ensemble.evaluate(results, MediaType.IMAGE, 0.5)
        assert len(result.head_verdicts) == 3
        # Agreement should be lower due to mixed results
        assert result.agreement < 1.0

    def test_specialist_uses_media_type_weights(self):
        """Specialist head should weight differently for audio vs video."""
        results = {
            "cqt_spectral": {"score": 0.8, "confidence": 0.7},
            "wavlm_deepfake": {"score": 0.3, "confidence": 0.6},
            "voice_clone": {"score": 0.7, "confidence": 0.8},
            "ecapa_tdnn": {"score": 0.5, "confidence": 0.5},
        }
        result_audio = self.ensemble.evaluate(results, MediaType.AUDIO, 0.5)
        result_video = self.ensemble.evaluate(results, MediaType.VIDEO, 0.5)
        # Different media types → different specialist scores
        assert result_audio.head_verdicts[2] != result_video.head_verdicts[2]


# ── MinorityReport ──


class TestMinorityReport:
    def setup_method(self):
        self.minority = MinorityReport()

    def test_no_dissent_all_agree(self):
        result = self.minority.analyze([0.3, 0.35, 0.32], agreement=0.95)
        assert not result.has_dissent

    def test_single_dissent(self):
        result = self.minority.analyze([0.2, 0.25, 0.8], agreement=0.3)
        assert result.has_dissent
        assert result.dissenting_head is not None
        assert result.dissent_magnitude > 0.3

    def test_strong_dissent_critical(self):
        """2/3 say authentic but 1 says fake → critical pattern."""
        result = self.minority.analyze([0.1, 0.15, 0.9], agreement=0.1)
        assert result.has_dissent
        assert "CRITICAL" in (result.recommendation or "")

    def test_dissent_magnitude(self):
        result = self.minority.analyze([0.5, 0.5, 0.9], agreement=0.5)
        assert result.has_dissent
        assert result.dissent_magnitude > 0.3


# ── AdversarialAuditor ──


class TestAdversarialAuditor:
    def setup_method(self):
        self.auditor = AdversarialAuditor()

    def test_clean_audit(self):
        results = {
            "clip_deepfake": {"score": 0.3, "confidence": 0.8},
            "frequency_analysis": {"score": 0.25, "confidence": 0.7},
        }
        audit = self.auditor.audit(results, perturbation_magnitude=0.005)
        assert not audit.adversarial_detected
        assert audit.robustness_score > 0.5

    def test_adversarial_detected_high_perturbation_and_divergence(self):
        original = {
            "clip_deepfake": {"score": 0.2, "confidence": 0.8},
            "frequency_analysis": {"score": 0.3, "confidence": 0.7},
        }
        purified = {
            "clip_deepfake": {"score": 0.7, "confidence": 0.6},
            "frequency_analysis": {"score": 0.8, "confidence": 0.65},
        }
        audit = self.auditor.audit(original, purified_results=purified, perturbation_magnitude=0.05)
        assert audit.adversarial_detected
        assert audit.robustness_score < 0.5

    def test_confidence_anomaly_detection(self):
        """High scores + low confidence → suspicious."""
        results = {
            "det1": {"score": 0.9, "confidence": 0.1},
            "det2": {"score": 0.85, "confidence": 0.15},
            "det3": {"score": 0.8, "confidence": 0.2},
        }
        audit = self.auditor.audit(results, perturbation_magnitude=0.0)
        # Should trigger confidence anomaly indicator
        conf_indicator = next((i for i in audit.indicators if i.name == "confidence_anomaly"), None)
        assert conf_indicator is not None
        assert conf_indicator.triggered

    def test_cross_detector_inconsistency(self):
        """Signal-based and NN-based detectors disagree strongly."""
        results = {
            "frequency_analysis": {"score": 0.1},
            "cqt_spectral": {"score": 0.15},
            "clip_deepfake": {"score": 0.9},
            "efficientnet_b0": {"score": 0.85},
        }
        audit = self.auditor.audit(results, perturbation_magnitude=0.0)
        cross_indicator = next((i for i in audit.indicators if i.name == "cross_detector_inconsistency"), None)
        assert cross_indicator is not None
        assert cross_indicator.triggered
