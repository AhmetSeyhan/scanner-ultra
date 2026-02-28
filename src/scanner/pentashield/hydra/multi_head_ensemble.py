"""HYDRA ENGINE — Multi-Head Ensemble.

Three independent decision heads evaluate detector results from different angles.
Each head uses a distinct strategy; consensus is reached via majority vote.

Heads:
  1. ConservativeHead — worst-case (max fake score)
  2. StatisticalHead — weighted median, outlier-resistant
  3. SpecialistHead — media-type-aware weighted combination
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scanner.models.enums import MediaType

logger = logging.getLogger(__name__)


# --- Specialist weights per media type ---
# Maps detector_name → weight.  Weights within a group sum to 1.0.
SPECIALIST_WEIGHTS: dict[MediaType, dict[str, float]] = {
    MediaType.VIDEO: {
        "ppg_biosignal": 0.30,
        "clip_deepfake": 0.25,
        "frequency_analysis": 0.20,
        "gaze_analysis": 0.15,
        "efficientnet_b0": 0.10,
    },
    MediaType.IMAGE: {
        "clip_deepfake": 0.30,
        "frequency_analysis": 0.25,
        "gan_artifact": 0.20,
        "diffusion_artifact": 0.15,
        "efficientnet_b0": 0.10,
    },
    MediaType.AUDIO: {
        "cqt_spectral": 0.30,
        "wavlm_deepfake": 0.25,
        "voice_clone": 0.25,
        "ecapa_tdnn": 0.20,
    },
    MediaType.TEXT: {
        "ai_text": 0.40,
        "perplexity": 0.35,
        "stylometric": 0.25,
    },
}


@dataclass
class MultiHeadResult:
    """Result from multi-head ensemble evaluation."""

    head_verdicts: list[float] = field(default_factory=list)
    consensus_score: float = 0.5
    agreement: float = 1.0
    head_names: list[str] = field(default_factory=lambda: ["conservative", "statistical", "specialist"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_verdicts": [round(v, 4) for v in self.head_verdicts],
            "consensus_score": round(self.consensus_score, 4),
            "agreement": round(self.agreement, 4),
            "head_names": self.head_names,
        }


class MultiHeadEnsemble:
    """Three independent heads evaluate detector results.

    - ConservativeHead: takes the maximum fake score (worst-case assumption)
    - StatisticalHead: computes weighted median, discarding outliers
    - SpecialistHead: uses media-type-specific expert weights
    """

    def evaluate(
        self,
        detector_results: dict[str, dict],
        media_type: MediaType,
        fused_score: float,
    ) -> MultiHeadResult:
        """Run 3 independent heads on detector results."""
        scores = self._extract_scores(detector_results)
        if not scores:
            return MultiHeadResult(
                head_verdicts=[fused_score, fused_score, fused_score],
                consensus_score=fused_score,
                agreement=1.0,
            )

        h1 = self._conservative_head(scores)
        h2 = self._statistical_head(scores)
        h3 = self._specialist_head(scores, detector_results, media_type)
        verdicts = [h1, h2, h3]

        consensus = self._consensus(verdicts)
        agreement = self._agreement(verdicts)

        return MultiHeadResult(
            head_verdicts=verdicts,
            consensus_score=consensus,
            agreement=agreement,
        )

    @staticmethod
    def _extract_scores(results: dict[str, dict]) -> dict[str, float]:
        """Extract detector_name → score mapping from results."""
        out: dict[str, float] = {}
        for name, data in results.items():
            score = data.get("score")
            if score is not None:
                out[name] = float(score)
        return out

    @staticmethod
    def _conservative_head(scores: dict[str, float]) -> float:
        """Worst-case: take the highest fake score among all detectors."""
        if not scores:
            return 0.5
        max_score = max(scores.values())
        # Soften slightly to avoid single-detector dominance
        # Blend: 70% max + 30% mean
        mean_score = float(np.mean(list(scores.values())))
        return 0.7 * max_score + 0.3 * mean_score

    @staticmethod
    def _statistical_head(scores: dict[str, float]) -> float:
        """Robust central tendency: weighted median with outlier rejection.

        1. Sort scores
        2. Reject scores outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        3. Return median of remaining
        """
        if not scores:
            return 0.5

        vals = np.array(sorted(scores.values()))
        if len(vals) < 3:
            return float(np.median(vals))

        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtered = vals[(vals >= lower) & (vals <= upper)]
        if len(filtered) == 0:
            return float(np.median(vals))

        return float(np.median(filtered))

    @staticmethod
    def _specialist_head(
        scores: dict[str, float],
        results: dict[str, dict],
        media_type: MediaType,
    ) -> float:
        """Media-type-aware weighted combination using expert weights."""
        weights = SPECIALIST_WEIGHTS.get(media_type, {})
        if not weights:
            # Fallback: equal weight for all available scores
            return float(np.mean(list(scores.values()))) if scores else 0.5

        weighted_sum = 0.0
        weight_sum = 0.0

        for det_name, weight in weights.items():
            if det_name in scores:
                # Also factor in confidence if available
                confidence = results.get(det_name, {}).get("confidence", 0.5)
                effective_weight = weight * max(0.1, confidence)
                weighted_sum += scores[det_name] * effective_weight
                weight_sum += effective_weight

        if weight_sum == 0:
            return float(np.mean(list(scores.values()))) if scores else 0.5

        return weighted_sum / weight_sum

    @staticmethod
    def _consensus(verdicts: list[float]) -> float:
        """Compute consensus score from head verdicts.

        Uses median to be robust against a single outlier head.
        """
        return float(np.median(verdicts))

    @staticmethod
    def _agreement(verdicts: list[float]) -> float:
        """Measure agreement between heads.

        Returns 1.0 when all heads agree perfectly, decreasing with disagreement.
        Uses: 1.0 - 3*std (scaled so std=0.33 gives agreement=0).
        """
        if len(verdicts) < 2:
            return 1.0
        std = float(np.std(verdicts))
        return max(0.0, min(1.0, 1.0 - 3.0 * std))
