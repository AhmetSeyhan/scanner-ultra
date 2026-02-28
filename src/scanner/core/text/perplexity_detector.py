"""Scanner ULTRA — Perplexity-based AI text detector.

AI text tends to have lower, more uniform perplexity.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class PerplexityDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.TEXT

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.TEXT_CONTENT}

    def __init__(self, model_path: str | None = None, device: str = "auto") -> None:
        super().__init__(model_path, device)
        self._tokenizer = None

    async def load_model(self) -> None:
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.to(self.device).eval()
        except ImportError:
            self.model = None

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.text or len(inp.text.strip()) < 50:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="ppl_skip",
                status=DetectorStatus.SKIPPED,
            )
        if self.model is None:
            score, details = self._entropy_fallback(inp.text)
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=score,
                confidence=0.3,
                method="entropy_fallback",
                status=DetectorStatus.PASS,
                details=details,
            )
        try:
            import torch

            text = inp.text.strip()[:2048]
            enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            ids = enc.input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(ids, labels=ids)
            ppl = math.exp(out.loss.item())
            # Per-token perplexity
            with torch.no_grad():
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = ids[..., 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_ppls = torch.exp(token_losses).cpu().numpy()
            ppl_std = float(np.std(token_ppls))
            ppl_mean = float(np.mean(token_ppls))
            if ppl < 30:
                score = 0.8
            elif ppl < 60:
                score = 0.6
            elif ppl > 200:
                score = 0.2
            else:
                score = 0.4
            uniformity = 1.0 / (1.0 + ppl_std / (ppl_mean + 1e-8))
            score = 0.6 * score + 0.4 * uniformity
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=score,
                confidence=0.7,
                method="gpt2_perplexity",
                status=DetectorStatus.PASS,
                details={"perplexity": round(ppl, 2), "ppl_std": round(ppl_std, 2), "n_tokens": len(token_ppls)},
            )
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="ppl_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
            )

    @staticmethod
    def _entropy_fallback(text: str) -> tuple[float, dict]:
        from collections import Counter

        chars = list(text.lower())
        freq = Counter(chars)
        total = len(chars)
        entropy = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
        score = 0.7 if entropy < 3.5 else (0.3 if entropy > 4.5 else 0.5)
        return score, {"char_entropy": round(entropy, 4), "mode": "fallback"}

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "Perplexity Detector (GPT-2)", "params": "124M"}
