"""ZERO-DAY SENTINEL — Anomaly Scorer.

Combines OOD, Physics, and Bio sub-scores into a final anomaly assessment.
Produces a unified anomaly_score and alert_level for the sentinel report.
"""

from __future__ import annotations

import logging
from typing import Any

from scanner.models.schemas import SentinelResult
from scanner.pentashield.sentinel.bio_consistency import BioResult
from scanner.pentashield.sentinel.ood_detector import OODResult
from scanner.pentashield.sentinel.physics_verifier import PhysicsResult

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Combine sentinel sub-scores into final anomaly assessment.

    Sub-score weights (sum = 1.0):
      - OOD:     0.35 — out-of-distribution novelty
      - Physics: 0.35 — physical consistency violations
      - Bio:     0.30 — biological signal inconsistency

    Alert levels:
      - none:     anomaly < 0.2
      - low:      0.2 ≤ anomaly < 0.4
      - medium:   0.4 ≤ anomaly < 0.6
      - high:     0.6 ≤ anomaly < 0.8
      - critical: anomaly ≥ 0.8
    """

    WEIGHTS: dict[str, float] = {
        "ood": 0.35,
        "physics": 0.35,
        "bio": 0.30,
    }

    ALERT_THRESHOLDS: list[tuple[float, str]] = [
        (0.8, "critical"),
        (0.6, "high"),
        (0.4, "medium"),
        (0.2, "low"),
        (0.0, "none"),
    ]

    def score(
        self,
        ood_result: OODResult,
        physics_result: PhysicsResult,
        bio_result: BioResult,
    ) -> SentinelResult:
        """Combine all sentinel sub-scores into final assessment."""
        # OOD score is already in [0, 1] where 1 = anomalous
        ood_val = ood_result.ood_score

        # Physics and bio are consistency scores (1 = good).
        # Flip so that 1 = anomalous for weighting.
        physics_val = 1.0 - physics_result.physics_score
        bio_val = 1.0 - bio_result.bio_consistency

        anomaly = self.WEIGHTS["ood"] * ood_val + self.WEIGHTS["physics"] * physics_val + self.WEIGHTS["bio"] * bio_val
        anomaly = max(0.0, min(1.0, anomaly))

        alert_level = self._to_alert(anomaly)

        if alert_level in ("high", "critical"):
            logger.warning(
                "Sentinel alert=%s: anomaly=%.3f (ood=%.3f, physics=%.3f, bio=%.3f)",
                alert_level,
                anomaly,
                ood_val,
                physics_val,
                bio_val,
            )

        return SentinelResult(
            ood_score=round(ood_result.ood_score, 4),
            is_novel_type=ood_result.is_novel_type,
            physics_score=round(physics_result.physics_score, 4),
            physics_anomalies=physics_result.anomalies,
            bio_consistency=round(bio_result.bio_consistency, 4),
            anomaly_score=round(anomaly, 4),
            alert_level=alert_level,
        )

    def _to_alert(self, anomaly: float) -> str:
        """Map anomaly score to alert level string."""
        for threshold, level in self.ALERT_THRESHOLDS:
            if anomaly >= threshold:
                return level
        return "none"

    @staticmethod
    def explain(result: SentinelResult) -> dict[str, Any]:
        """Generate human-readable explanation of sentinel findings."""
        factors: list[str] = []

        if result.is_novel_type:
            factors.append(f"Content appears to be a novel/unseen type (OOD score: {result.ood_score:.2f})")

        if result.physics_score < 0.6:
            factors.append(f"Physical inconsistencies detected (score: {result.physics_score:.2f})")
            if result.physics_anomalies:
                factors.extend(f"  - {a}" for a in result.physics_anomalies)

        if result.bio_consistency < 0.6:
            factors.append(f"Biological signal inconsistencies (score: {result.bio_consistency:.2f})")

        if not factors:
            factors.append("No significant anomalies detected by sentinel analysis.")

        return {
            "alert_level": result.alert_level,
            "anomaly_score": result.anomaly_score,
            "factors": factors,
        }
