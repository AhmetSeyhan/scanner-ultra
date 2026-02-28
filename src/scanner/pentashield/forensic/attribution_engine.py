"""Scanner ULTRA — Attribution Engine for FORENSIC DNA.

Coordinates spectral analysis + generator fingerprinting to identify content source.
Main entry point for the FORENSIC DNA technology.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.models.schemas import ForensicDNAResult
from scanner.pentashield.forensic.generator_fingerprinter import GeneratorFingerprinter
from scanner.pentashield.forensic.spectral_analyzer import SpectralAnalyzer

logger = logging.getLogger(__name__)


class AttributionEngine:
    """Main coordinator for FORENSIC DNA analysis."""

    def __init__(self) -> None:
        """Initialize spectral analyzer and generator fingerprinter."""
        self.spectral = SpectralAnalyzer()
        self.fingerprinter = GeneratorFingerprinter()

    def analyze(
        self,
        detector_results: dict[str, dict],
        frames: list[np.ndarray] | None = None,
        defense_results: dict[str, Any] | None = None,
    ) -> ForensicDNAResult:
        """Run full forensic DNA analysis pipeline.

        Pipeline:
          1. Spectral analysis (if frames available)
          2. Generator fingerprinting (spectral + detector scores + metadata)
          3. Build attribution report
          4. Return ForensicDNAResult

        Args:
            detector_results: Dict of detector_name → {score, confidence, details, ...}
            frames: Optional video/image frames for spectral analysis
            defense_results: Optional defense module outputs (metadata_forensics, provenance)

        Returns:
            ForensicDNAResult with generator identification and attribution report
        """
        # Step 1: Spectral analysis (skip if no frames)
        spectral_result = None
        if frames:
            freq_details = detector_results.get("frequency_analysis", {}).get("details")
            spectral_result = self.spectral.analyze(frames, freq_details)
        else:
            logger.debug("No frames provided, skipping spectral analysis")

        # Step 2: Generator fingerprinting
        fingerprint_result = self.fingerprinter.fingerprint(spectral_result, detector_results, defense_results)

        # Step 3: Build attribution report
        attribution_report = self._build_report(fingerprint_result, spectral_result)

        # Step 4: Construct ForensicDNAResult
        generator_detected = False
        if fingerprint_result.top_match and fingerprint_result.top_confidence >= 0.3:
            generator_detected = True

        # Spectral fingerprints (simplified for API)
        spectral_fps = []
        if spectral_result and spectral_result.fingerprints:
            for fp in spectral_result.fingerprints[:2]:  # Include first 2
                spectral_fps.append(
                    {
                        "band_energies": fp.band_energies,
                        "centroid": round(fp.centroid, 4),
                        "has_periodic_peaks": fp.has_periodic_peaks,
                    }
                )

        # Forensic score (weighted combination)
        forensic_score = self._compute_forensic_score(fingerprint_result, spectral_result, detector_results)

        return ForensicDNAResult(
            generator_detected=generator_detected,
            generator_type=fingerprint_result.top_match,
            generator_confidence=round(fingerprint_result.top_confidence, 4),
            spectral_fingerprints=spectral_fps,
            attribution_report=attribution_report,
            forensic_score=round(forensic_score, 4),
        )

    @staticmethod
    def _build_report(
        fingerprint_result: Any,  # FingerprintResult
        spectral_result: Any | None,  # SpectralResult or None
    ) -> dict[str, Any]:
        """Build comprehensive attribution report.

        Args:
            fingerprint_result: Output from GeneratorFingerprinter
            spectral_result: Output from SpectralAnalyzer (or None)

        Returns:
            Attribution report dict
        """
        report: dict[str, Any] = {
            "method": "forensic_dna_v1",
            "generator_candidates": fingerprint_result.candidates[:5],  # Top 5
            "generator_family": fingerprint_result.generator_family,
            "evidence": fingerprint_result.evidence_summary,
            "spectral_analysis": {},
        }

        # Spectral analysis section
        if spectral_result:
            report["spectral_analysis"] = {
                "best_match": spectral_result.best_match,
                "best_match_score": round(spectral_result.best_match_score, 4),
                "num_fingerprints": len(spectral_result.fingerprints),
            }

        return report

    @staticmethod
    def _compute_forensic_score(
        fingerprint_result: Any,
        spectral_result: Any | None,
        detector_results: dict[str, dict],
    ) -> float:
        """Compute overall forensic score (0=no evidence, 1=strong generator match).

        Args:
            fingerprint_result: FingerprintResult
            spectral_result: SpectralResult or None
            detector_results: Detector outputs

        Returns:
            Forensic score (0.0 - 1.0)
        """
        score = 0.0

        # Top candidate confidence (50% weight)
        score += 0.5 * fingerprint_result.top_confidence

        # Spectral match strength (20% weight)
        if spectral_result and spectral_result.best_match_score > 0:
            score += 0.2 * spectral_result.best_match_score

        # GAN artifact presence (15% weight)
        gan_score = detector_results.get("gan_artifact", {}).get("score", 0.0)
        score += 0.15 * gan_score

        # Diffusion artifact presence (15% weight)
        diff_score = detector_results.get("diffusion_artifact", {}).get("score", 0.0)
        score += 0.15 * diff_score

        return min(1.0, score)
