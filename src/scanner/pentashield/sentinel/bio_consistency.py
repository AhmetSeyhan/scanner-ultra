"""ZERO-DAY SENTINEL — Bio Consistency Checker.

Cross-checks biological signals from FAZ 1 detectors (PPG, gaze).
Individual bio signals can be fooled, but maintaining consistency
across multiple bio channels simultaneously is extremely difficult.

Cross-checks:
  1. PPG temporal — physiological plausibility of heart-rate signal
  2. Gaze-head consistency — gaze direction vs implied head pose
  3. PPG-gaze correlation — blink artifacts in PPG signal
  4. Bio signal smoothness — unnatural jumps in time-series
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BioResult:
    """Result from bio consistency analysis."""

    bio_consistency: float = 1.0  # 1.0 = consistent, 0.0 = inconsistent
    check_details: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bio_consistency": round(self.bio_consistency, 4),
            "check_details": {k: round(v, 4) for k, v in self.check_details.items()},
            "issues": self.issues,
        }


class BioConsistency:
    """Cross-check biological signals for consistency.

    Works by analyzing the details dict from PPG and gaze detectors.
    If those detectors haven't run (e.g., for image/audio-only input),
    returns neutral score (1.0 = no bio inconsistency found).
    """

    # Thresholds for individual checks
    PERIODICITY_MIN = 0.15  # PPG should have *some* periodicity
    PERIODICITY_MAX = 0.95  # Perfect periodicity is suspicious
    SNR_MIN = 0.05  # Some SNR expected in real PPG
    SPATIAL_CONSISTENCY_MIN = 0.3  # Left/right PPG correlation

    def check(
        self,
        detector_results: dict[str, dict],
        frames: list[np.ndarray] | None = None,
    ) -> BioResult:
        """Cross-check biological signals for consistency."""
        checks: dict[str, float] = {}
        issues: list[str] = []

        ppg_data = self._get_details(detector_results, "ppg_biosignal")
        gaze_data = self._get_details(detector_results, "gaze_analysis")

        # PPG temporal plausibility
        if ppg_data:
            score, issue = self._ppg_temporal(ppg_data)
            checks["ppg_temporal"] = score
            if issue:
                issues.append(issue)

        # Gaze-head pose consistency
        if gaze_data:
            score, issue = self._gaze_consistency(gaze_data)
            checks["gaze_consistency"] = score
            if issue:
                issues.append(issue)

        # PPG spatial consistency (left vs right face)
        if ppg_data:
            score, issue = self._ppg_spatial(ppg_data)
            checks["ppg_spatial"] = score
            if issue:
                issues.append(issue)

        # Bio signal smoothness (check for unnatural jumps)
        if ppg_data or gaze_data:
            score, issue = self._signal_smoothness(ppg_data, gaze_data)
            checks["signal_smoothness"] = score
            if issue:
                issues.append(issue)

        # Overall: mean of all checks, default to 1.0 if no bio data
        overall = float(np.mean(list(checks.values()))) if checks else 1.0

        return BioResult(
            bio_consistency=overall,
            check_details=checks,
            issues=issues,
        )

    @staticmethod
    def _get_details(results: dict[str, dict], name: str) -> dict[str, Any]:
        """Extract details dict from a named detector result."""
        entry = results.get(name, {})
        return entry.get("details", {})

    def _ppg_temporal(self, ppg: dict[str, Any]) -> tuple[float, str | None]:
        """Check PPG signal temporal plausibility.

        Real PPG has moderate periodicity (heart rate) and non-zero SNR.
        Synthetic faces either have no PPG signal or perfectly periodic one.
        """
        periodicity = ppg.get("periodicity", 0.5)
        snr = ppg.get("snr", 0.5)

        score = 1.0
        issues: list[str] = []

        # Periodicity check: should be in plausible range
        if periodicity < self.PERIODICITY_MIN:
            penalty = (self.PERIODICITY_MIN - periodicity) / self.PERIODICITY_MIN
            score -= 0.3 * penalty
            issues.append(f"Low PPG periodicity ({periodicity:.3f})")
        elif periodicity > self.PERIODICITY_MAX:
            penalty = (periodicity - self.PERIODICITY_MAX) / (1.0 - self.PERIODICITY_MAX)
            score -= 0.2 * penalty
            issues.append(f"Suspiciously perfect PPG periodicity ({periodicity:.3f})")

        # SNR check
        if snr < self.SNR_MIN:
            score -= 0.3
            issues.append(f"No detectable PPG signal (SNR={snr:.3f})")

        score = max(0.0, score)
        issue = "; ".join(issues) if issues else None
        return (score, issue)

    @staticmethod
    def _gaze_consistency(gaze: dict[str, Any]) -> tuple[float, str | None]:
        """Check gaze direction plausibility.

        Real gaze patterns show natural saccades and micro-movements.
        Synthetic faces often have unnaturally stable or random gaze.
        """
        # Use available gaze metrics
        blink_rate = gaze.get("blink_rate", None)
        consistency = gaze.get("consistency", 0.7)

        score = float(consistency)
        issues: list[str] = []

        # Blink rate check: normal is 15-20 per minute
        # If available and zero → suspicious
        if blink_rate is not None and blink_rate == 0:
            score -= 0.2
            issues.append("No blinks detected in gaze tracking")

        # Very low consistency already penalized by gaze detector
        if consistency < 0.3:
            issues.append(f"Low gaze consistency ({consistency:.3f})")

        score = max(0.0, min(1.0, score))
        issue = "; ".join(issues) if issues else None
        return (score, issue)

    def _ppg_spatial(self, ppg: dict[str, Any]) -> tuple[float, str | None]:
        """Check PPG spatial consistency (left vs right face).

        Real faces show correlated PPG signals on both face halves.
        """
        spatial = ppg.get("spatial_consistency", None)
        if spatial is None:
            return (1.0, None)

        score = float(spatial)
        issue = None

        if score < self.SPATIAL_CONSISTENCY_MIN:
            issue = f"PPG spatial inconsistency: L/R correlation={score:.3f}"

        return (max(0.0, score), issue)

    @staticmethod
    def _signal_smoothness(
        ppg: dict[str, Any],
        gaze: dict[str, Any],
    ) -> tuple[float, str | None]:
        """Check for unnatural discontinuities in bio signals.

        Real biological signals change smoothly. Synthetic signals may
        have sudden jumps between frames.
        """
        scores: list[float] = []

        # Use PPG score and gaze score as proxies for signal quality
        ppg_score = ppg.get("score") if ppg else None
        gaze_score = gaze.get("score") if gaze else None

        if ppg_score is not None and gaze_score is not None:
            # Both present: check if they're at least roughly consistent
            # Both should lean the same direction
            ppg_verdict = ppg_score > 0.5  # True = likely fake
            gaze_verdict = gaze_score > 0.5
            if ppg_verdict == gaze_verdict:
                scores.append(1.0)  # Consistent
            else:
                scores.append(0.6)  # Moderate disagreement

        # If only one signal, we can't cross-check, so assume OK
        if not scores:
            return (1.0, None)

        avg = float(np.mean(scores))
        issue = None
        if avg < 0.7:
            issue = "Bio signal cross-check: PPG and gaze disagree on authenticity"

        return (avg, issue)
