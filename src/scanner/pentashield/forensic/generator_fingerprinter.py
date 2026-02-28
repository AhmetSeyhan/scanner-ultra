"""Scanner ULTRA — Generator fingerprinting.

Identifies the specific AI generator (StyleGAN2, Stable Diffusion, etc.) by combining:
  - Spectral fingerprint matches
  - GAN/diffusion artifact scores
  - Metadata evidence (AI software signatures)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Generator families and their indicators
GENERATOR_FAMILIES = {
    # GAN-based generators
    "stylegan2": {
        "family": "gan",
        "indicators": ["periodic_peaks", "checkerboard", "upsampling_artifacts"],
    },
    "stylegan3": {
        "family": "gan",
        "indicators": ["periodic_peaks", "rotation_equivariance"],
    },
    "progan": {
        "family": "gan",
        "indicators": ["periodic_peaks", "progressive_artifacts"],
    },
    # Face swap tools
    "faceswap": {
        "family": "faceswap",
        "indicators": ["boundary_artifacts", "color_mismatch", "resolution_gap"],
    },
    "deepfacelab": {
        "family": "faceswap",
        "indicators": ["boundary_artifacts", "color_bleed", "mask_artifacts"],
    },
    "roop": {
        "family": "faceswap",
        "indicators": ["boundary_artifacts", "temporal_flicker"],
    },
    # Diffusion-based generators
    "stable_diffusion": {
        "family": "diffusion",
        "indicators": ["uniform_noise", "low_texture_var", "guidance_artifacts"],
    },
    "dalle": {
        "family": "diffusion",
        "indicators": ["uniform_noise", "high_detail", "inpainting_marks"],
    },
    "midjourney": {
        "family": "diffusion",
        "indicators": ["uniform_noise", "artistic_style", "high_detail"],
    },
}

# Evidence weights for final score
EVIDENCE_WEIGHTS = {
    "spectral_match": 0.40,  # Spectral fingerprint similarity
    "gan_artifact": 0.20,  # GAN artifact detector score
    "diffusion_artifact": 0.20,  # Diffusion artifact detector score
    "metadata": 0.20,  # AI software detected in metadata
}


@dataclass
class FingerprintResult:
    """Result of generator fingerprinting."""

    top_match: str | None
    top_confidence: float
    candidates: list[dict[str, Any]]
    generator_family: str | None
    evidence_summary: dict[str, Any]


class GeneratorFingerprinter:
    """Identifies the specific generator from combined evidence."""

    def __init__(self, generator_db: dict[str, dict] | None = None) -> None:
        """Initialize with optional custom generator database.

        Args:
            generator_db: Custom generator families. If None, uses GENERATOR_FAMILIES.
        """
        self.generator_db = generator_db or GENERATOR_FAMILIES

    def fingerprint(
        self,
        spectral_result: Any,  # SpectralResult
        detector_results: dict[str, dict],
        defense_results: dict[str, Any] | None = None,
    ) -> FingerprintResult:
        """Identify the most likely generator from combined evidence.

        Args:
            spectral_result: Output from SpectralAnalyzer
            detector_results: Dict of detector outputs (gan_artifact, diffusion_artifact, etc.)
            defense_results: Optional metadata forensics results

        Returns:
            FingerprintResult with top match, confidence, and ranked candidates
        """
        # Collect evidence
        evidence = self._collect_evidence(spectral_result, detector_results, defense_results)

        # Score each candidate generator
        scored = self._score_candidates(evidence)

        # Rank candidates
        ranked = self._rank_candidates(scored)

        # Top match
        top = ranked[0] if ranked else {"name": None, "confidence": 0.0, "evidence": {}}

        # Determine family
        family = None
        if top["name"]:
            family = self.generator_db.get(top["name"], {}).get("family")

        return FingerprintResult(
            top_match=top["name"],
            top_confidence=top["confidence"],
            candidates=ranked,
            generator_family=family,
            evidence_summary=evidence,
        )

    @staticmethod
    def _collect_evidence(
        spectral_result: Any,
        detector_results: dict[str, dict],
        defense_results: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Collect all available evidence for fingerprinting.

        Args:
            spectral_result: SpectralResult object
            detector_results: Detector outputs
            defense_results: Metadata forensics results

        Returns:
            Evidence dict with spectral, gan, diffusion, metadata scores
        """
        evidence = {
            "spectral_match": None,
            "spectral_best": None,
            "gan_artifact_score": 0.0,
            "diffusion_artifact_score": 0.0,
            "ai_software": None,
        }

        # Spectral evidence
        if spectral_result and spectral_result.best_match:
            evidence["spectral_match"] = spectral_result.best_match
            evidence["spectral_best"] = spectral_result.best_match_score

        # GAN artifact score
        gan_result = detector_results.get("gan_artifact", {})
        evidence["gan_artifact_score"] = gan_result.get("score", 0.0)

        # Diffusion artifact score
        diff_result = detector_results.get("diffusion_artifact", {})
        evidence["diffusion_artifact_score"] = diff_result.get("score", 0.0)

        # Metadata evidence
        if defense_results:
            metadata = defense_results.get("metadata_forensics", {})
            if metadata.get("ai_software_detected"):
                evidence["ai_software"] = metadata.get("ai_software_name")

        return evidence

    def _score_candidates(self, evidence: dict[str, Any]) -> dict[str, float]:
        """Score each candidate generator based on evidence.

        Args:
            evidence: Evidence dict from _collect_evidence

        Returns:
            Dict of {generator_name: confidence_score}
        """
        scored = {}

        for gen_name in self.generator_db:
            score = self._score_single_candidate(gen_name, evidence)
            scored[gen_name] = score

        return scored

    def _score_single_candidate(self, gen_name: str, evidence: dict[str, Any]) -> float:
        """Score a single generator candidate.

        Args:
            gen_name: Generator name (e.g., "stylegan2")
            evidence: Evidence dict

        Returns:
            Confidence score (0.0 - 1.0)
        """
        score = 0.0
        gen_family = self.generator_db.get(gen_name, {}).get("family", "unknown")

        # 1. Spectral match (40% weight)
        if evidence["spectral_match"] == gen_name and evidence["spectral_best"] is not None:
            score += EVIDENCE_WEIGHTS["spectral_match"] * evidence["spectral_best"]

        # 2. GAN artifact (20% weight)
        if gen_family == "gan" or gen_family == "faceswap":
            score += EVIDENCE_WEIGHTS["gan_artifact"] * evidence["gan_artifact_score"]

        # 3. Diffusion artifact (20% weight)
        if gen_family == "diffusion":
            score += EVIDENCE_WEIGHTS["diffusion_artifact"] * evidence["diffusion_artifact_score"]

        # 4. Metadata (20% weight)
        if evidence["ai_software"]:
            software = evidence["ai_software"].lower()
            if gen_name.lower() in software or any(
                ind in software for ind in self.generator_db.get(gen_name, {}).get("indicators", [])
            ):
                score += EVIDENCE_WEIGHTS["metadata"]

        return min(1.0, score)

    @staticmethod
    def _rank_candidates(scored: dict[str, float]) -> list[dict[str, Any]]:
        """Rank candidates by confidence score.

        Args:
            scored: Dict of {generator_name: score}

        Returns:
            List of candidates sorted by confidence (descending)
        """
        ranked = [{"name": name, "confidence": round(score, 4), "evidence": {}} for name, score in scored.items()]

        # Sort descending
        ranked.sort(key=lambda x: x["confidence"], reverse=True)

        # Filter out very low confidence (<0.1)
        ranked = [c for c in ranked if c["confidence"] >= 0.1]

        return ranked
