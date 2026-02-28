"""ZERO-DAY SENTINEL — Out-of-Distribution Detector.

Detects inputs that fall outside the training distribution —
novel deepfake types that existing detectors haven't been trained on.

Techniques:
  1. Energy Score — log-sum-exp of detector scores (pseudo-logits)
  2. Detector Entropy — high entropy = uncertain = possible OOD
  3. Feature Distance — kNN on CLIP embeddings (when reference available)
  4. Score Variance — abnormal variance pattern across detectors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """Result from out-of-distribution detection."""

    ood_score: float = 0.0  # 0=in-distribution, 1=novel
    is_novel_type: bool = False
    energy: float = 0.0
    entropy: float = 0.0
    score_variance: float = 0.0
    feature_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ood_score": round(self.ood_score, 4),
            "is_novel_type": self.is_novel_type,
            "energy": round(self.energy, 4),
            "entropy": round(self.entropy, 4),
            "score_variance": round(self.score_variance, 4),
        }
        if self.feature_distance is not None:
            result["feature_distance"] = round(self.feature_distance, 4)
        return result


class OODDetector:
    """Detect out-of-distribution inputs using detector score analysis.

    Smart stub mode: Uses detector score distribution heuristics.
    With reference embeddings: Also uses CLIP embedding kNN distance.

    OOD score is normalized to [0, 1]:
      0.0 = clearly in-distribution (familiar content)
      1.0 = clearly out-of-distribution (novel content)
    """

    NOVEL_THRESHOLD = 0.7

    def __init__(
        self,
        reference_path: str | None = None,
        temperature: float = 1.5,
    ) -> None:
        self.temperature = temperature
        self.reference_path = reference_path
        self._reference_embeddings: np.ndarray | None = None

    def detect(
        self,
        detector_results: dict[str, dict],
        clip_embeddings: np.ndarray | None = None,
    ) -> OODResult:
        """Detect out-of-distribution inputs."""
        scores = self._extract_scores(detector_results)

        if not scores:
            return OODResult(ood_score=0.5, is_novel_type=False)

        energy = self._energy_score(scores)
        entropy = self._detector_entropy(scores)
        variance = self._score_variance(scores)

        # Feature distance (optional — requires reference embeddings)
        feat_dist = None
        if clip_embeddings is not None:
            self._load_reference()
            if self._reference_embeddings is not None:
                feat_dist = self._feature_distance(clip_embeddings)

        ood_score = self._combine(energy, entropy, variance, feat_dist)
        is_novel = ood_score > self.NOVEL_THRESHOLD

        if is_novel:
            logger.warning(
                "Novel input type detected: OOD score=%.3f (energy=%.3f, entropy=%.3f, var=%.3f)",
                ood_score,
                energy,
                entropy,
                variance,
            )

        return OODResult(
            ood_score=ood_score,
            is_novel_type=is_novel,
            energy=energy,
            entropy=entropy,
            score_variance=variance,
            feature_distance=feat_dist,
        )

    @staticmethod
    def _extract_scores(results: dict[str, dict]) -> list[float]:
        """Extract raw scores from detector results."""
        return [float(data["score"]) for data in results.values() if "score" in data]

    def _energy_score(self, scores: list[float]) -> float:
        """Energy-based OOD score.

        E(x) = -T * log( sum(exp(s_i / T)) )

        In-distribution inputs have low energy (confident, polarized scores).
        OOD inputs have high energy (scores clustered around 0.5).

        We normalize to [0, 1] via sigmoid mapping.
        """
        if not scores:
            return 0.5

        arr = np.array(scores)
        # Treat scores as pseudo-logits, scale to [-2, 2] range
        logits = (arr - 0.5) * 4.0
        energy = -self.temperature * np.log(np.sum(np.exp(logits / self.temperature)) + 1e-8)
        # Normalize: high energy → high OOD
        # Empirical range: energy in [-10, 0] → map to [0, 1]
        normalized = 1.0 / (1.0 + np.exp(-0.5 * (energy + 5.0)))
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _detector_entropy(scores: list[float]) -> float:
        """Entropy of the detector score distribution.

        High entropy = detectors are confused = possible OOD.
        Low entropy = detectors agree (either clearly fake or clearly real).

        Uses histogram binning of scores into [0, 0.33), [0.33, 0.66), [0.66, 1.0].
        """
        if len(scores) < 2:
            return 0.5

        # Bin scores into 3 categories: authentic, uncertain, fake
        bins = np.array([0.0, 0.33, 0.66, 1.01])
        hist, _ = np.histogram(scores, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]

        if len(probs) <= 1:
            return 0.0  # All in one bin → low entropy

        entropy = -float(np.sum(probs * np.log2(probs)))
        # Normalize by max entropy (log2(3) ≈ 1.585)
        max_entropy = np.log2(len(bins) - 1)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    @staticmethod
    def _score_variance(scores: list[float]) -> float:
        """Normalized variance of detector scores.

        Very low variance with scores clustered at 0.5 → OOD indicator.
        High variance with polarized scores → normal disagreement.
        """
        if len(scores) < 2:
            return 0.0

        arr = np.array(scores)
        mean = float(np.mean(arr))
        var = float(np.var(arr))

        # If mean is close to 0.5 AND variance is low → suspicious
        mean_centrality = 1.0 - abs(mean - 0.5) * 2  # 1 when mean=0.5
        low_var_indicator = max(0.0, 1.0 - var * 20.0)  # 1 when var→0

        return float(np.clip(mean_centrality * low_var_indicator, 0.0, 1.0))

    def _load_reference(self) -> None:
        """Lazy-load reference embeddings for kNN distance."""
        if self._reference_embeddings is not None or self.reference_path is None:
            return
        try:
            data = np.load(self.reference_path)
            self._reference_embeddings = data.get("embeddings", data.get("arr_0"))
            logger.info(
                "Loaded %d reference embeddings from %s",
                len(self._reference_embeddings) if self._reference_embeddings is not None else 0,
                self.reference_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("Reference embeddings not available: %s", exc)
            self._reference_embeddings = None

    def _feature_distance(self, embedding: np.ndarray) -> float:
        """kNN distance to reference embeddings.

        Compute L2 distance to the k nearest neighbors in the reference set.
        Large distance → OOD.
        """
        if self._reference_embeddings is None:
            return 0.0

        ref = self._reference_embeddings
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Ensure compatible dimensions
        if embedding.shape[-1] != ref.shape[-1]:
            return 0.0

        # Compute L2 distances to all reference points
        dists = np.linalg.norm(ref - embedding, axis=1)
        k = min(5, len(dists))
        knn_dist = float(np.mean(np.sort(dists)[:k]))

        # Normalize: empirical range [0, 2] for L2-normalized embeddings
        return float(np.clip(knn_dist / 2.0, 0.0, 1.0))

    @staticmethod
    def _combine(
        energy: float,
        entropy: float,
        variance: float,
        feat_dist: float | None,
    ) -> float:
        """Combine OOD sub-scores into final score."""
        if feat_dist is not None:
            # With feature distance: weight it heavily
            combined = 0.25 * energy + 0.25 * entropy + 0.15 * variance + 0.35 * feat_dist
        else:
            # Without feature distance: rely on score-based heuristics
            combined = 0.35 * energy + 0.40 * entropy + 0.25 * variance
        return float(np.clip(combined, 0.0, 1.0))
