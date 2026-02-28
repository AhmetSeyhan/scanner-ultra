"""Scanner ULTRA — Vision Transformer (ViT) deepfake detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scanner.core.base_detector import BaseDetector, DetectorInput, DetectorResult
from scanner.models.enums import DetectorCapability, DetectorStatus, DetectorType

logger = logging.getLogger(__name__)


class ViTDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "vit_forgery"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    async def load_model(self) -> None:
        self._has_deepfake_weights = False
        try:
            import timm

            self.model = timm.create_model("vit_base_patch16_224", pretrained=not bool(self.model_path), num_classes=2)
            if self.model_path:
                import torch

                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self._has_deepfake_weights = True
                logger.info("ViT deepfake weights yüklendi: %s", self.model_path)
            else:
                logger.warning("ViT deepfake weights bulunamadı — ImageNet pretrained, güven düşük")
            self.model.to(self.device).eval()
        except ImportError:
            self.model = None

    async def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames or ([inp.image] if inp.image is not None else [])
        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="vit_skip",
                status=DetectorStatus.SKIPPED,
            )
        if self.model is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.1,
                method="vit_stub",
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
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )
            scores = []
            for f in frames[:16]:
                with torch.no_grad():
                    scores.append(F.softmax(self.model(transform(f).unsqueeze(0).to(self.device)), dim=-1)[0, 1].item())
            avg = float(np.mean(scores))
            raw_conf = max(0.1, 1.0 - float(np.std(scores)) * 2)
            # Deepfake weights yoksa classification head random → güven %20'ye düşür
            confidence = raw_conf if self._has_deepfake_weights else raw_conf * 0.2
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=avg,
                confidence=confidence,
                method="vit_inference" if self._has_deepfake_weights else "vit_imagenet",
                status=DetectorStatus.PASS,
                details={"n_frames": len(scores), "deepfake_weights": self._has_deepfake_weights},
            )
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                method="vit_error",
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
            )

    def get_model_info(self) -> dict[str, Any]:
        return {"name": "ViT-B/16", "params": "86M", "input_size": "224x224"}
