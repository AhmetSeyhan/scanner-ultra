"""HYDRA ENGINE — Minority Report.

Tracks dissenting opinions among multi-head verdicts.
When one head strongly disagrees with the majority, it may indicate:
  - Targeted adversarial attack (fooling some detectors but not all)
  - Edge case where different analysis strategies diverge
  - Genuine ambiguity requiring human review

The majority verdict is applied, but dissenting views are recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Head names matching MultiHeadEnsemble order
HEAD_NAMES = ["conservative", "statistical", "specialist"]


@dataclass
class MinorityReportResult:
    """Result of minority report analysis."""

    has_dissent: bool = False
    dissenting_head: str | None = None
    dissenting_head_index: int | None = None
    dissent_magnitude: float = 0.0
    majority_score: float = 0.5
    minority_score: float = 0.5
    alert_text: str | None = None
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if not self.has_dissent:
            return {"has_dissent": False}
        return {
            "has_dissent": True,
            "dissenting_head": self.dissenting_head,
            "dissent_magnitude": round(self.dissent_magnitude, 4),
            "majority_score": round(self.majority_score, 4),
            "minority_score": round(self.minority_score, 4),
            "alert_text": self.alert_text,
            "recommendation": self.recommendation,
        }


class MinorityReport:
    """Detects and reports dissenting opinions among heads.

    A dissent is flagged when one head's score differs from the other two
    by more than DISSENT_THRESHOLD. This is especially critical when
    2/3 heads say "authentic" but 1 says "fake" — possible targeted attack.
    """

    DISSENT_THRESHOLD = 0.3

    def analyze(
        self,
        head_verdicts: list[float],
        agreement: float,
    ) -> MinorityReportResult:
        """Detect dissenting opinions among heads."""
        if len(head_verdicts) < 2:
            return MinorityReportResult()

        # Find the head that deviates most from the others
        verdicts = np.array(head_verdicts)
        dissent_idx, magnitude = self._find_outlier(verdicts)

        if magnitude < self.DISSENT_THRESHOLD:
            return MinorityReportResult()

        # Calculate majority vs minority
        majority_indices = [i for i in range(len(verdicts)) if i != dissent_idx]
        majority_score = float(np.mean(verdicts[majority_indices]))
        minority_score = float(verdicts[dissent_idx])

        head_name = HEAD_NAMES[dissent_idx] if dissent_idx < len(HEAD_NAMES) else f"head_{dissent_idx}"

        alert_text = self._generate_alert(head_name, minority_score, majority_score, magnitude)
        recommendation = self._generate_recommendation(minority_score, majority_score, magnitude)

        logger.warning(
            "Minority report: %s dissents (score=%.3f vs majority=%.3f, magnitude=%.3f)",
            head_name,
            minority_score,
            majority_score,
            magnitude,
        )

        return MinorityReportResult(
            has_dissent=True,
            dissenting_head=head_name,
            dissenting_head_index=dissent_idx,
            dissent_magnitude=magnitude,
            majority_score=majority_score,
            minority_score=minority_score,
            alert_text=alert_text,
            recommendation=recommendation,
        )

    @staticmethod
    def _find_outlier(verdicts: np.ndarray) -> tuple[int, float]:
        """Find the index and magnitude of the most outlying verdict.

        For each head, compute abs distance to median of others.
        Return the head with the largest distance.
        """
        n = len(verdicts)
        max_dist = 0.0
        max_idx = 0

        for i in range(n):
            others = np.concatenate([verdicts[:i], verdicts[i + 1 :]])
            dist = abs(float(verdicts[i]) - float(np.median(others)))
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        return max_idx, max_dist

    @staticmethod
    def _generate_alert(
        head_name: str,
        minority_score: float,
        majority_score: float,
        magnitude: float,
    ) -> str:
        """Generate human-readable alert text."""
        minority_label = "fake" if minority_score > 0.5 else "authentic"
        majority_label = "fake" if majority_score > 0.5 else "authentic"

        if minority_label != majority_label:
            return (
                f"DISSENT: {head_name} head rates content as {minority_label} "
                f"(score={minority_score:.2f}) while majority rates it as "
                f"{majority_label} (score={majority_score:.2f}). "
                f"Divergence magnitude: {magnitude:.2f}"
            )

        return (
            f"Minor disagreement: {head_name} head score ({minority_score:.2f}) "
            f"differs from majority ({majority_score:.2f}) by {magnitude:.2f}"
        )

    @staticmethod
    def _generate_recommendation(
        minority_score: float,
        majority_score: float,
        magnitude: float,
    ) -> str:
        """Generate recommendation based on dissent pattern."""
        # Most critical: majority says authentic but minority says fake
        # This could indicate a targeted adversarial attack
        if majority_score < 0.4 and minority_score > 0.6:
            return (
                "CRITICAL: One analysis head detects manipulation that others miss. "
                "This pattern is consistent with a targeted adversarial attack. "
                "Manual review strongly recommended."
            )

        # Majority says fake but minority says authentic
        if majority_score > 0.6 and minority_score < 0.4:
            return (
                "One analysis head considers the content authentic despite "
                "majority detecting manipulation. This may indicate partial "
                "manipulation or high-quality deepfake."
            )

        # High magnitude disagreement in general
        if magnitude > 0.5:
            return (
                "Significant disagreement between analysis strategies. "
                "Content may be at the boundary of detection capability. "
                "Human review recommended."
            )

        return "Minor divergence in analysis. Majority verdict applied."
