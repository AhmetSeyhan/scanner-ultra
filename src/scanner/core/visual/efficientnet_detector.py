"""Scanner ULTRA — EfficientNet-B0 deepfake detector.

Fine-tuned on OpenRL/DeepFakeFace (30K samples, A100 GPU).
Architecture matches Colab training: torchvision efficientnet_b0 + 2-class head.
Weights: weights/best_efficientnet.pt (97% acc, 99.4% AUC)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)

# weights/ dizini: scanner-ultra/weights/best_efficientnet.pt
_WEIGHTS_DIR = Path(__file__).parents[4] / "weights"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "best_efficientnet.pt"


class EfficientNetDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "efficientnet_b0"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    async def load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            # Colab eğitimiyle aynı mimari
            model = efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

            # Weights yolu: model_path > varsayılan weights/best_efficientnet.pt
            weights_path = Path(self.model_path) if self.model_path else _DEFAULT_WEIGHTS

            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                logger.info("EfficientNet weights yüklendi: %s", weights_path)
            else:
                # Fallback: ImageNet pretrained (2-class head random)
                logger.warning("Weights bulunamadı (%s) — ImageNet pretrained kullanılıyor", weights_path)
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

            self.model = model.to(self.device).eval()
        except ImportError:
            logger.warning("torchvision/torch mevcut değil — stub mode")
            self.model = None

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames or ([inp.image] if inp.image is not None else [])
        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="efficientnet_skip",
                status=DetectorStatus.SKIPPED,
            )
        if self.model is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="efficientnet_stub",
                status=DetectorStatus.PASS,
                details={"mode": "stub"},
            )
        try:
            import torch
            import torch.nn.functional as F
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            scores = []
            for frame in frames[:16]:
                tensor = transform(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    prob = (
                        F.softmax(logits, dim=-1)[0, 1].item()
                        if logits.shape[-1] == 2
                        else torch.sigmoid(logits[0, 0]).item()
                    )
                scores.append(prob)
            avg = float(np.mean(scores))
            std = float(np.std(scores))
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=avg,
                confidence=max(0.1, 1.0 - std * 2),
                method="efficientnet_ff++",
                status=DetectorStatus.PASS,
                details={"n_frames": len(scores), "std": round(std, 4)},
            )
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="efficientnet_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
            )

    def get_model_info(self) -> dict[str, Any]:
        weights_loaded = _DEFAULT_WEIGHTS.exists()
        return {
            "name": "EfficientNet-B0",
            "params": "5.3M",
            "input_size": "224x224",
            "training": "OpenRL/DeepFakeFace — 30K samples, A100 GPU",
            "accuracy": "97.04%",
            "auc": "99.36%",
            "weights_file": str(_DEFAULT_WEIGHTS),
            "weights_loaded": weights_loaded,
        }
