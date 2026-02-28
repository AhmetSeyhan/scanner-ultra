"""PentaShield Engine — Coordinates HYDRA ENGINE + ZERO-DAY SENTINEL.

Single entry point for the orchestrator:
  engine.analyze(...) → PentaShieldResult

The engine runs HYDRA (adversarial defense) and SENTINEL (anomaly detection)
in sequence, then decides whether to override the fusion verdict.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from scanner.models.enums import MediaType, Verdict
from scanner.models.schemas import (
    GhostProtocolResult,
    HydraResult,
    PentaShieldResult,
    SentinelResult,
)
from scanner.pentashield.active_probe.probe_session import ProbeSession
from scanner.pentashield.forensic.attribution_engine import AttributionEngine
from scanner.pentashield.hydra.adversarial_auditor import AdversarialAuditor
from scanner.pentashield.hydra.input_purifier import InputPurifier
from scanner.pentashield.hydra.minority_report import MinorityReport
from scanner.pentashield.hydra.multi_head_ensemble import MultiHeadEnsemble
from scanner.pentashield.sentinel.anomaly_scorer import AnomalyScorer
from scanner.pentashield.sentinel.bio_consistency import BioConsistency
from scanner.pentashield.sentinel.ood_detector import OODDetector
from scanner.pentashield.sentinel.physics_verifier import PhysicsVerifier

logger = logging.getLogger(__name__)


class PentaShieldEngine:
    """Coordinates HYDRA ENGINE + ZERO-DAY SENTINEL.

    Pipeline:
      HYDRA:
        1. Purify input frames (detect adversarial noise)
        2. Multi-head ensemble (3 independent decision heads)
        3. Minority report (dissent tracking)
        4. Adversarial audit (evidence collection)

      SENTINEL:
        1. OOD detection (novel type detection)
        2. Physics verification (lighting, shadows, reflections)
        3. Bio consistency (cross-check PPG + gaze)
        4. Anomaly scoring (combined sentinel score)

      Override Logic:
        - Adversarial detected → force UNCERTAIN
        - Novel type → force UNCERTAIN
        - Strong head disagreement → force UNCERTAIN
    """

    def __init__(self) -> None:
        # HYDRA components
        self.purifier = InputPurifier()
        self.multi_head = MultiHeadEnsemble()
        self.minority = MinorityReport()
        self.auditor = AdversarialAuditor()

        # SENTINEL components
        self.ood = OODDetector()
        self.physics = PhysicsVerifier()
        self.bio = BioConsistency()
        self.anomaly_scorer = AnomalyScorer()

        # FORENSIC DNA
        self.attribution = AttributionEngine()

        # ACTIVE PROBE
        self.probe = ProbeSession()

    async def analyze(
        self,
        detector_results: dict[str, dict],
        fused_score: float,
        fused_confidence: float,
        media_type: MediaType,
        frames: list[np.ndarray] | None = None,
        fps: float = 30.0,
        defense_results: dict[str, Any] | None = None,
    ) -> PentaShieldResult:
        """Run full PentaShield analysis.

        Args:
            detector_results: Dict of detector_name → {score, confidence, details, ...}
            fused_score: Score from fusion engine (0=authentic, 1=fake)
            fused_confidence: Confidence from fusion engine
            media_type: Type of media being analyzed
            frames: Optional video/image frames for physics + purifier analysis
            fps: Frames per second (for active probe latency analysis)
            defense_results: Optional defense module outputs (metadata, provenance)

        Returns:
            PentaShieldResult with hydra + sentinel + forensic + probe sub-results and optional override.
        """
        start = time.perf_counter()

        # === HYDRA ENGINE ===
        hydra = self._run_hydra(detector_results, fused_score, media_type, frames)

        # === ZERO-DAY SENTINEL ===
        sentinel = self._run_sentinel(detector_results, frames)

        # === FORENSIC DNA ===
        forensic_dna = self._run_forensic(detector_results, frames, defense_results)

        # === ACTIVE PROBE ===
        active_probe = self._run_probe(media_type, frames, fps)

        # === GHOST PROTOCOL ===
        ghost_protocol = self._run_ghost_protocol()

        # === Override Logic ===
        override_verdict, override_reason = self._check_overrides(
            hydra, sentinel, fused_score, forensic_dna, active_probe
        )

        elapsed = (time.perf_counter() - start) * 1000

        return PentaShieldResult(
            hydra=hydra,
            sentinel=sentinel,
            forensic_dna=forensic_dna,
            active_probe=active_probe,
            ghost_protocol=ghost_protocol,
            override_verdict=override_verdict,
            override_reason=override_reason,
            processing_time_ms=round(elapsed, 2),
        )

    def _run_hydra(
        self,
        detector_results: dict[str, dict],
        fused_score: float,
        media_type: MediaType,
        frames: list[np.ndarray] | None,
    ) -> HydraResult:
        """Run HYDRA ENGINE: adversarial defense + multi-head ensemble."""
        # Step 1: Purify input (only if frames available)
        purification_applied = False
        adversarial_from_purifier = False
        perturbation_mag = 0.0

        if frames:
            purify_result = self.purifier.purify(frames)
            purification_applied = True
            adversarial_from_purifier = purify_result.adversarial_detected
            perturbation_mag = purify_result.perturbation_magnitude

        # Step 2: Multi-head ensemble
        mh_result = self.multi_head.evaluate(detector_results, media_type, fused_score)

        # Step 3: Minority report
        mr_result = self.minority.analyze(mh_result.head_verdicts, mh_result.agreement)

        # Step 4: Adversarial audit
        audit_result = self.auditor.audit(
            original_results=detector_results,
            purified_results=None,  # Full re-detection on purified not done here
            perturbation_magnitude=perturbation_mag,
        )

        # Combine results
        adversarial_detected = adversarial_from_purifier or audit_result.adversarial_detected

        minority_dict: dict[str, Any] | None = None
        if mr_result.has_dissent:
            minority_dict = mr_result.to_dict()

        return HydraResult(
            adversarial_detected=adversarial_detected,
            purification_applied=purification_applied,
            head_verdicts=[round(v, 4) for v in mh_result.head_verdicts],
            consensus_score=round(mh_result.consensus_score, 4),
            minority_report=minority_dict,
            robustness_score=round(audit_result.robustness_score, 4),
        )

    def _run_sentinel(
        self,
        detector_results: dict[str, dict],
        frames: list[np.ndarray] | None,
    ) -> SentinelResult:
        """Run ZERO-DAY SENTINEL: OOD + physics + bio consistency."""
        # Step 1: OOD detection
        ood_result = self.ood.detect(detector_results)

        # Step 2: Physics verification (only with frames)
        if frames:
            physics_result = self.physics.verify(frames)
        else:
            from scanner.pentashield.sentinel.physics_verifier import PhysicsResult

            physics_result = PhysicsResult()  # defaults to 1.0

        # Step 3: Bio consistency
        bio_result = self.bio.check(detector_results, frames)

        # Step 4: Combine into sentinel result
        sentinel_result = self.anomaly_scorer.score(ood_result, physics_result, bio_result)

        return sentinel_result

    def _run_forensic(
        self,
        detector_results: dict[str, dict],
        frames: list[np.ndarray] | None,
        defense_results: dict[str, Any] | None,
    ):
        """Run FORENSIC DNA: generator fingerprinting + attribution."""
        return self.attribution.analyze(detector_results, frames, defense_results)

    def _run_probe(
        self,
        media_type: MediaType,
        frames: list[np.ndarray] | None,
        fps: float,
    ):
        """Run ACTIVE PROBE: challenge-response liveness verification."""
        return self.probe.run(media_type, frames, fps)

    @staticmethod
    def _run_ghost_protocol():
        """Run GHOST PROTOCOL: edge AI + federated + continual learning.

        NOTE: Ghost Protocol is primarily for training/deployment, not real-time scanning.
        Returns default stub values for API compatibility.
        """
        return GhostProtocolResult(
            edge_model_available=False,
            edge_model_size_mb=0.0,
            edge_inference_time_ms=0.0,
            federated_round=0,
            privacy_budget_used=0.0,
            continual_tasks_learned=0,
            edge_score=0.5,
            edge_confidence=0.1,
        )

    @staticmethod
    def _check_overrides(
        hydra: HydraResult,
        sentinel: SentinelResult,
        fused_score: float,
        forensic_dna: Any = None,  # ForensicDNAResult
        active_probe: Any = None,  # ActiveProbeResult
    ) -> tuple[Verdict | None, str | None]:
        """Determine if PentaShield should override the fusion verdict.

        Override scenarios:
          1. Adversarial attack detected → UNCERTAIN
          2. Novel content type (high OOD) → UNCERTAIN
          3. Severe physics anomalies (score < 0.3) → LIKELY_FAKE
          4. Strong head disagreement + minority dissent → UNCERTAIN
          5. Generator identified (confidence ≥ 0.8) → LIKELY_FAKE
          6. Active probe detected playback → LIKELY_FAKE

        Returns:
            (override_verdict, override_reason) or (None, None) if no override.
        """
        # 1. Adversarial attack → UNCERTAIN
        if hydra.adversarial_detected:
            return (
                Verdict.UNCERTAIN,
                "Adversarial manipulation detected. Results may be unreliable. Manual review required.",
            )

        # 2. Novel type → UNCERTAIN
        if sentinel.is_novel_type:
            return (
                Verdict.UNCERTAIN,
                f"Novel content type detected (OOD score: {sentinel.ood_score:.2f}). "
                f"Existing detectors may not cover this type. "
                f"Manual review recommended.",
            )

        # 3. Severe physics violations → escalate toward FAKE
        if sentinel.physics_score < 0.3:
            anomaly_text = "; ".join(sentinel.physics_anomalies[:3])
            return (
                Verdict.LIKELY_FAKE,
                f"Severe physical inconsistencies detected "
                f"(physics score: {sentinel.physics_score:.2f}). "
                f"Anomalies: {anomaly_text}",
            )

        # 4. Strong dissent in multi-head + low agreement
        if hydra.minority_report is not None:
            dissent_mag = hydra.minority_report.get("dissent_magnitude", 0)
            if dissent_mag > 0.5:
                return (
                    Verdict.UNCERTAIN,
                    f"Strong disagreement among analysis heads "
                    f"(dissent magnitude: {dissent_mag:.2f}). "
                    f"Manual review recommended.",
                )

        # 5. Generator identified with high confidence → LIKELY_FAKE
        if forensic_dna and forensic_dna.generator_detected:
            if forensic_dna.generator_confidence >= 0.8:
                return (
                    Verdict.LIKELY_FAKE,
                    f"Generator identified: {forensic_dna.generator_type} "
                    f"(confidence: {forensic_dna.generator_confidence:.2f}). "
                    f"Forensic analysis strongly indicates synthetic content.",
                )

        # 6. Active probe detected playback → LIKELY_FAKE
        if active_probe and active_probe.probe_verdict == "playback":
            return (
                Verdict.LIKELY_FAKE,
                f"Active probe detected playback/pre-recorded content "
                f"(liveness score: {active_probe.liveness_score:.2f}). "
                f"Not a live interaction.",
            )

        # No override needed
        return (None, None)
