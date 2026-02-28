"""HYDRA ENGINE — Adversarial Auditor.

Systematically collects evidence of adversarial attacks by examining:
  1. Perturbation magnitude (from InputPurifier)
  2. Result divergence (original vs purified detector scores)
  3. Confidence anomalies (high score + low confidence)
  4. Cross-detector inconsistency (signal-based vs NN-based)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Detectors grouped by analysis approach
SIGNAL_BASED = {"frequency_analysis", "cqt_spectral", "ppg_biosignal", "gaze_analysis"}
NN_BASED = {"clip_deepfake", "efficientnet_b0", "xception", "vit_deepfake", "wavlm_deepfake"}


@dataclass
class Indicator:
    """A single adversarial indicator."""

    name: str
    triggered: bool = False
    severity: float = 0.0  # 0-1
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "triggered": self.triggered,
            "severity": round(self.severity, 4),
            "detail": self.detail,
        }


@dataclass
class AuditResult:
    """Result of adversarial audit."""

    adversarial_detected: bool = False
    indicators: list[Indicator] = field(default_factory=list)
    robustness_score: float = 1.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "adversarial_detected": self.adversarial_detected,
            "robustness_score": round(self.robustness_score, 4),
            "triggered_count": sum(i.triggered for i in self.indicators),
            "total_indicators": len(self.indicators),
            "indicators": [i.to_dict() for i in self.indicators],
            "summary": self.summary,
        }


class AdversarialAuditor:
    """Collects and evaluates evidence of adversarial manipulation.

    An adversarial attack is flagged when 2+ indicators trigger simultaneously.
    The robustness score (0=vulnerable, 1=robust) reflects overall confidence
    that the result hasn't been tampered with.
    """

    # Thresholds
    PERTURBATION_THRESHOLD = 0.015
    DIVERGENCE_THRESHOLD = 0.20
    CONFIDENCE_ANOMALY_THRESHOLD = 0.3
    CROSS_DETECTOR_THRESHOLD = 0.35
    MIN_TRIGGERS_FOR_ADVERSARIAL = 2

    def audit(
        self,
        original_results: dict[str, dict],
        purified_results: dict[str, dict] | None = None,
        perturbation_magnitude: float = 0.0,
    ) -> AuditResult:
        """Run adversarial audit on detection results."""
        indicators: list[Indicator] = []

        # Indicator 1: Perturbation magnitude
        indicators.append(self._check_perturbation(perturbation_magnitude))

        # Indicator 2: Result divergence (original vs purified)
        indicators.append(self._check_divergence(original_results, purified_results))

        # Indicator 3: Confidence anomalies
        indicators.append(self._check_confidence_anomaly(original_results))

        # Indicator 4: Cross-detector inconsistency
        indicators.append(self._check_cross_detector(original_results))

        triggered = sum(i.triggered for i in indicators)
        is_adversarial = triggered >= self.MIN_TRIGGERS_FOR_ADVERSARIAL
        robustness = self._compute_robustness(indicators)

        summary = self._generate_summary(indicators, is_adversarial, robustness)

        if is_adversarial:
            logger.warning(
                "Adversarial attack detected: %d/%d indicators triggered, robustness=%.3f",
                triggered,
                len(indicators),
                robustness,
            )

        return AuditResult(
            adversarial_detected=is_adversarial,
            indicators=indicators,
            robustness_score=robustness,
            summary=summary,
        )

    def _check_perturbation(self, magnitude: float) -> Indicator:
        """Check if input-purification revealed significant perturbation."""
        triggered = magnitude > self.PERTURBATION_THRESHOLD
        severity = min(1.0, magnitude / (self.PERTURBATION_THRESHOLD * 3))
        return Indicator(
            name="perturbation_magnitude",
            triggered=triggered,
            severity=severity,
            detail=f"L2 perturbation: {magnitude:.6f} (threshold: {self.PERTURBATION_THRESHOLD})",
        )

    def _check_divergence(
        self,
        original: dict[str, dict],
        purified: dict[str, dict] | None,
    ) -> Indicator:
        """Check if purified results diverge significantly from original.

        Large divergence means the purification removed something that was
        affecting detector outputs — strong sign of adversarial noise.
        """
        if purified is None:
            return Indicator(
                name="result_divergence",
                triggered=False,
                severity=0.0,
                detail="No purified results available for comparison",
            )

        divergences = []
        for name in original:
            if name in purified:
                orig_score = original[name].get("score", 0.5)
                pur_score = purified[name].get("score", 0.5)
                divergences.append(abs(orig_score - pur_score))

        if not divergences:
            return Indicator(
                name="result_divergence",
                triggered=False,
                severity=0.0,
                detail="No overlapping detectors to compare",
            )

        avg_div = float(np.mean(divergences))
        max_div = float(np.max(divergences))
        triggered = avg_div > self.DIVERGENCE_THRESHOLD
        severity = min(1.0, avg_div / (self.DIVERGENCE_THRESHOLD * 2))

        return Indicator(
            name="result_divergence",
            triggered=triggered,
            severity=severity,
            detail=f"Avg divergence: {avg_div:.4f}, max: {max_div:.4f} (threshold: {self.DIVERGENCE_THRESHOLD})",
        )

    def _check_confidence_anomaly(self, results: dict[str, dict]) -> Indicator:
        """Detect high score + low confidence pattern.

        Adversarial attacks often push scores to extremes while leaving
        the model internally uncertain (low confidence).
        """
        anomalies = []
        for name, data in results.items():
            score = data.get("score", 0.5)
            confidence = data.get("confidence", 0.5)
            # High-certainty score (far from 0.5) but low confidence
            score_extremity = abs(score - 0.5) * 2  # 0-1 scale
            if score_extremity > 0.5 and confidence < 0.3:
                anomalies.append((name, score, confidence))

        triggered = len(anomalies) >= 2
        severity = min(1.0, len(anomalies) / 4.0)

        detail = f"{len(anomalies)} detectors show high-score/low-confidence pattern"
        if anomalies:
            examples = ", ".join(f"{n}(s={s:.2f},c={c:.2f})" for n, s, c in anomalies[:3])
            detail += f": {examples}"

        return Indicator(
            name="confidence_anomaly",
            triggered=triggered,
            severity=severity,
            detail=detail,
        )

    def _check_cross_detector(self, results: dict[str, dict]) -> Indicator:
        """Check consistency between signal-based and NN-based detectors.

        Adversarial attacks typically target NN-based detectors but leave
        signal-based detectors (FFT, CQT, PPG) unaffected, creating a
        distinctive pattern.
        """
        signal_scores = []
        nn_scores = []

        for name, data in results.items():
            score = data.get("score")
            if score is None:
                continue
            if name in SIGNAL_BASED:
                signal_scores.append(float(score))
            elif name in NN_BASED:
                nn_scores.append(float(score))

        if not signal_scores or not nn_scores:
            return Indicator(
                name="cross_detector_inconsistency",
                triggered=False,
                severity=0.0,
                detail="Insufficient detector types for cross-comparison",
            )

        signal_mean = float(np.mean(signal_scores))
        nn_mean = float(np.mean(nn_scores))
        gap = abs(signal_mean - nn_mean)

        triggered = gap > self.CROSS_DETECTOR_THRESHOLD
        severity = min(1.0, gap / (self.CROSS_DETECTOR_THRESHOLD * 2))

        return Indicator(
            name="cross_detector_inconsistency",
            triggered=triggered,
            severity=severity,
            detail=(
                f"Signal-based mean: {signal_mean:.3f}, NN-based mean: {nn_mean:.3f}, "
                f"gap: {gap:.3f} (threshold: {self.CROSS_DETECTOR_THRESHOLD})"
            ),
        )

    @staticmethod
    def _compute_robustness(indicators: list[Indicator]) -> float:
        """Compute overall robustness score from indicators.

        1.0 = fully robust (no adversarial signs)
        0.0 = likely under attack
        """
        if not indicators:
            return 1.0
        severities = [i.severity for i in indicators]
        # Robustness is the complement of the weighted severity
        # Triggered indicators contribute more
        weighted = sum(s * (2.0 if i.triggered else 0.5) for i, s in zip(indicators, severities))
        max_possible = len(indicators) * 2.0
        return max(0.0, 1.0 - weighted / max_possible)

    @staticmethod
    def _generate_summary(
        indicators: list[Indicator],
        is_adversarial: bool,
        robustness: float,
    ) -> str:
        """Generate human-readable audit summary."""
        triggered = [i for i in indicators if i.triggered]
        if not triggered:
            return "No adversarial indicators detected. Results appear robust."

        names = ", ".join(i.name for i in triggered)
        if is_adversarial:
            return (
                f"ADVERSARIAL ATTACK SUSPECTED: {len(triggered)} indicators triggered "
                f"({names}). Robustness score: {robustness:.2f}. "
                f"Results may be unreliable — manual review recommended."
            )

        return (
            f"Minor anomaly: {len(triggered)} indicator(s) triggered ({names}). "
            f"Robustness score: {robustness:.2f}. "
            f"Below adversarial threshold but worth monitoring."
        )
