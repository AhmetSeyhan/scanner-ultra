"""Scanner ULTRA — Trust Score Engine.

Final decision layer: 0.0=fake, 1.0=authentic → Verdict + ThreatLevel.
"""

from __future__ import annotations

import logging
from typing import Any

from scanner.models.enums import ThreatLevel, Verdict

logger = logging.getLogger(__name__)

THREAT_MAP = {
    Verdict.FAKE: ThreatLevel.CRITICAL,
    Verdict.LIKELY_FAKE: ThreatLevel.HIGH,
    Verdict.UNCERTAIN: ThreatLevel.MEDIUM,
    Verdict.LIKELY_AUTHENTIC: ThreatLevel.LOW,
    Verdict.AUTHENTIC: ThreatLevel.NONE,
}


class TrustScoreEngine:
    def compute(
        self, fused_score: float, confidence: float, modality_details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        trust_score = max(0.0, min(1.0, 1.0 - fused_score))
        verdict = self._to_verdict(trust_score, confidence)
        threat = THREAT_MAP.get(verdict, ThreatLevel.MEDIUM)
        explanation = self._explain(trust_score, verdict, confidence)
        return {
            "trust_score": round(trust_score, 4),
            "verdict": verdict,
            "threat_level": threat,
            "confidence": round(confidence, 4),
            "explanation": explanation,
        }

    @staticmethod
    def _to_verdict(ts: float, confidence: float = 0.5) -> Verdict:
        """Confidence'a göre adaptif threshold.

        Yüksek confidence → dar uncertain bölgesi → net karar verir.
        Düşük confidence → geniş uncertain bölgesi → muhafazakâr davranır.
        """
        # Uncertain bölgesinin genişliği: confidence yükseldikçe daralır
        # conf=0.9 → uncertain bölgesi [0.47, 0.53]
        # conf=0.5 → uncertain bölgesi [0.40, 0.60]
        # conf=0.2 → uncertain bölgesi [0.35, 0.65]
        half_width = 0.13 - confidence * 0.06  # [0.07, 0.13]
        uncertain_lo = 0.5 - half_width
        uncertain_hi = 0.5 + half_width

        if ts > uncertain_hi:
            # Authentic taraf
            if ts >= 0.82:
                return Verdict.AUTHENTIC
            return Verdict.LIKELY_AUTHENTIC
        if ts < uncertain_lo:
            # Fake taraf
            if ts <= 0.18:
                return Verdict.FAKE
            return Verdict.LIKELY_FAKE
        return Verdict.UNCERTAIN

    @staticmethod
    def _explain(ts: float, verdict: Verdict, conf: float) -> dict[str, str]:
        msgs = {
            Verdict.AUTHENTIC: "Content appears authentic with high confidence.",
            Verdict.LIKELY_AUTHENTIC: "Content is likely authentic, minor anomalies detected.",
            Verdict.UNCERTAIN: "Unable to determine authenticity. Manual review recommended.",
            Verdict.LIKELY_FAKE: "Content shows signs of manipulation.",
            Verdict.FAKE: "Content is highly likely manipulated or synthetic.",
        }
        return {
            "summary": msgs.get(verdict, msgs[Verdict.UNCERTAIN]),
            "trust_score_label": f"{ts:.0%}",
            "confidence_label": f"{conf:.0%}",
        }
