"""Tests for fusion modules."""

from scanner.models.enums import ThreatLevel, Verdict


class TestCrossModalAttention:
    def test_fuse_single_modality(self):
        from scanner.core.fusion.cross_modal_attention import CrossModalAttention

        cma = CrossModalAttention()
        result = cma.fuse(
            visual_results={"det1": {"score": 0.8, "confidence": 0.7}},
            audio_results={},
            text_results={},
        )
        assert "fused_score" in result
        assert 0.0 <= result["fused_score"] <= 1.0

    def test_fuse_multi_modality(self):
        from scanner.core.fusion.cross_modal_attention import CrossModalAttention

        cma = CrossModalAttention()
        result = cma.fuse(
            visual_results={"v1": {"score": 0.9, "confidence": 0.8}},
            audio_results={"a1": {"score": 0.85, "confidence": 0.7}},
            text_results={"t1": {"score": 0.3, "confidence": 0.5}},
        )
        assert result["fused_score"] > 0.0
        assert "attention_weights" in result

    def test_fuse_no_results(self):
        from scanner.core.fusion.cross_modal_attention import CrossModalAttention

        cma = CrossModalAttention()
        result = cma.fuse({}, {}, {})
        assert result["fused_score"] == 0.5


class TestTemporalConsistency:
    def test_stable_scores(self):
        from scanner.core.fusion.temporal_consistency import TemporalConsistency

        tc = TemporalConsistency()
        result = tc.analyze([0.8, 0.81, 0.79, 0.82, 0.78])
        assert result["consistency"] > 0.7
        assert not result["flickering"]

    def test_flickering_scores(self):
        from scanner.core.fusion.temporal_consistency import TemporalConsistency

        tc = TemporalConsistency()
        result = tc.analyze([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
        assert result["flickering"]

    def test_smooth(self):
        from scanner.core.fusion.temporal_consistency import TemporalConsistency

        tc = TemporalConsistency(window_size=3)
        smoothed = tc.smooth_scores([0.1, 0.9, 0.1, 0.9, 0.1])
        assert len(smoothed) == 5


class TestConfidenceCalibrator:
    def test_temperature_scaling(self):
        from scanner.core.fusion.confidence_calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator(temperature=1.5)
        s, c = cal.calibrate(0.9, 0.8, "unknown_det")
        assert 0.0 <= s <= 1.0
        assert c <= 0.95

    def test_platt_scaling(self):
        from scanner.core.fusion.confidence_calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()
        cal.set_calibration("det1", a=2.0, b=-1.0)
        s, c = cal.calibrate(0.7, 0.8, "det1")
        assert 0.0 <= s <= 1.0

    def test_batch(self):
        from scanner.core.fusion.confidence_calibrator import ConfidenceCalibrator

        cal = ConfidenceCalibrator()
        batch = {"d1": {"score": 0.8, "confidence": 0.7}, "d2": {"score": 0.3, "confidence": 0.5}}
        result = cal.calibrate_batch(batch)
        assert len(result) == 2
        assert all(r["calibrated"] for r in result.values())


class TestTrustScoreEngine:
    def test_fake(self):
        from scanner.core.fusion.trust_score_engine import TrustScoreEngine

        tse = TrustScoreEngine()
        result = tse.compute(fused_score=0.95, confidence=0.9)
        assert result["verdict"] == Verdict.FAKE
        assert result["threat_level"] == ThreatLevel.CRITICAL
        assert result["trust_score"] < 0.1

    def test_authentic(self):
        from scanner.core.fusion.trust_score_engine import TrustScoreEngine

        tse = TrustScoreEngine()
        result = tse.compute(fused_score=0.05, confidence=0.9)
        assert result["verdict"] == Verdict.AUTHENTIC
        assert result["trust_score"] > 0.9

    def test_uncertain(self):
        from scanner.core.fusion.trust_score_engine import TrustScoreEngine

        tse = TrustScoreEngine()
        result = tse.compute(fused_score=0.5, confidence=0.3)
        assert result["verdict"] == Verdict.UNCERTAIN
