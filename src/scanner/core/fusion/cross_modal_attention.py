"""Scanner ULTRA — Cross-modal attention fusion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ATTENTION_PRIORS = {
    ("visual", "audio"): 0.7,
    ("visual", "text"): 0.4,
    ("audio", "text"): 0.5,
}


class CrossModalAttention:
    """Cross-modal attention-based fusion of detector results."""

    def fuse(
        self,
        visual_results: dict[str, dict[str, Any]],
        audio_results: dict[str, dict[str, Any]],
        text_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        modality_scores: dict[str, dict[str, Any]] = {}

        for name, results in [("visual", visual_results), ("audio", audio_results), ("text", text_results)]:
            if results:
                # Stub, skip ve hata sonuçlarını fusion'dan dışla
                active = {
                    k: r
                    for k, r in results.items()
                    if "stub" not in r.get("method", "")
                    and r.get("status") not in ("skipped", "error", "SKIPPED", "ERROR")
                    and r.get("confidence", 0.0) >= 0.15
                }
                if not active:
                    continue
                scores = [r.get("score", 0.5) for r in active.values()]
                confs = [r.get("confidence", 0.0) for r in active.values()]
                tc = sum(confs) + 1e-8
                # Confidence-weighted mean of scores (düşük-confidence stub'lar neredeyse etkisiz)
                weighted_score = sum(s * c for s, c in zip(scores, confs)) / tc
                # Modality confidence: confidence-weighted mean of confidences
                # (simple mean yerine — stub'ların etkisini azaltır)
                conf_weighted_conf = sum(c * c for c in confs) / tc
                modality_scores[name] = {
                    "score": weighted_score,
                    "confidence": float(conf_weighted_conf),
                    "n_detectors": len(active),
                    "raw_scores": scores,
                }

        if not modality_scores:
            return {"fused_score": 0.5, "confidence": 0.0, "method": "no_modalities"}

        weights = self._compute_attention(modality_scores)
        fs, tw = 0.0, 0.0
        for m, info in modality_scores.items():
            w = weights.get(m, 1.0)
            fs += w * info["score"] * info["confidence"]
            tw += w * info["confidence"]

        agreement = self._agreement(modality_scores)
        return {
            "fused_score": round(fs / tw if tw > 0 else 0.5, 4),
            "confidence": round(min(0.95, agreement * 1.2), 4),
            "attention_weights": {k: round(v, 4) for k, v in weights.items()},
            "modality_scores": {k: round(v["score"], 4) for k, v in modality_scores.items()},
            "agreement": round(agreement, 4),
            "method": "cross_modal_attention",
        }

    @staticmethod
    def _compute_attention(ms: dict[str, dict]) -> dict[str, float]:
        mods = list(ms.keys())
        weights = {m: 1.0 for m in mods}
        for i, m1 in enumerate(mods):
            for m2 in mods[i + 1 :]:
                agr = 1.0 - abs(ms[m1]["score"] - ms[m2]["score"])
                prior = ATTENTION_PRIORS.get((m1, m2), ATTENTION_PRIORS.get((m2, m1), 0.5))
                boost = agr * prior
                weights[m1] += boost
                weights[m2] += boost
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}

    @staticmethod
    def _agreement(ms: dict[str, dict]) -> float:
        if not ms:
            return 0.3
        if len(ms) == 1:
            # Tek modalite: o modalitedeki raw detector skorlarının agreement'ı kullan
            info = next(iter(ms.values()))
            raw = info.get("raw_scores", [info["score"]])
            if len(raw) < 2:
                # Tek detector → modality confidence'ını doğrudan kullan
                return float(info.get("confidence", 0.3))
            # Detector'lar kendi aralarında ne kadar anlaşıyor?
            within_std = float(np.std(raw))
            within_agreement = max(0.0, 1.0 - within_std * 4)
            # Modality confidence ile çarp — hem agreement hem de güven dikkate alınır
            return min(0.9, within_agreement * float(info.get("confidence", 0.5)) * 1.5)
        # Birden fazla modalite: modality score'ları arasındaki agreement
        scores = [info["score"] for info in ms.values()]
        return max(0.0, 1.0 - float(np.std(scores)) * 3)
