"""Scanner ULTRA — Tiny Model for Edge Deployment.

Knowledge distillation from large ensemble to tiny model (<10MB).
MobileNetV3-Small backbone for edge devices (phones, tablets, IoT).

References:
- MobileNetV3: https://arxiv.org/abs/1905.02244
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TinyModelBackbone(nn.Module):
    """Tiny MobileNetV3-Small inspired backbone (<10MB).

    Optimized for edge devices with limited compute.
    Uses depthwise separable convolutions and inverted residuals.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        """Initialize tiny backbone.

        Args:
            num_classes: Number of output classes (2 for real/fake binary)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )

        # Inverted residual blocks (depthwise separable)
        self.blocks = nn.Sequential(
            self._make_block(16, 16, kernel_size=3, stride=1, expand_ratio=1),
            self._make_block(16, 24, kernel_size=3, stride=2, expand_ratio=4),
            self._make_block(24, 24, kernel_size=3, stride=1, expand_ratio=3),
            self._make_block(24, 40, kernel_size=5, stride=2, expand_ratio=3),
            self._make_block(40, 40, kernel_size=5, stride=1, expand_ratio=3),
            self._make_block(40, 48, kernel_size=5, stride=1, expand_ratio=3),
            self._make_block(48, 96, kernel_size=5, stride=2, expand_ratio=6),
        )

        # Head: Global pooling + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
    ) -> nn.Module:
        """Create inverted residual block."""
        hidden_dim = in_channels * expand_ratio
        use_residual = stride == 1 and in_channels == out_channels

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.Hardswish(),
                ]
            )

        # Depthwise
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(),
            ]
        )

        # Project
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        block = nn.Sequential(*layers)
        if use_residual:
            return _ResidualWrapper(block, in_channels, out_channels)
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class _ResidualWrapper(nn.Module):
    """Wrapper to add residual connection."""

    def __init__(self, block: nn.Module, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = block
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class TinyModel:
    """Knowledge distillation trainer for tiny edge model.

    Distills knowledge from large teacher ensemble to tiny student model.
    Target: <10MB model size, <100ms inference on mobile devices.
    """

    def __init__(
        self,
        teacher_models: list[nn.Module] | None = None,
        temperature: float = 3.0,
        alpha: float = 0.7,
        device: str = "auto",
    ) -> None:
        """Initialize tiny model distiller.

        Args:
            teacher_models: List of teacher models for ensemble distillation
            temperature: Softmax temperature for distillation (higher = softer)
            alpha: Weight for distillation loss vs hard label loss (0.7 = 70% distill)
            device: Device for training (auto/cuda/cpu)
        """
        self.teacher_models = teacher_models or []
        self.temperature = temperature
        self.alpha = alpha
        self.device = self._resolve_device(device)

        # Student model (tiny)
        self.student = TinyModelBackbone(num_classes=2).to(self.device)

        logger.info(f"TinyModel initialized on {self.device}")
        self._log_model_size()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _log_model_size(self) -> None:
        """Log model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.student.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.student.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)
        logger.info(f"Student model size: {size_mb:.2f} MB")

    def distill(
        self,
        train_data: Any,
        val_data: Any | None = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> dict[str, Any]:
        """Distill knowledge from teacher to student.

        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Training history dict
        """
        if not self.teacher_models:
            logger.warning("No teacher models provided, training student from scratch")
            return self._train_from_scratch(train_data, val_data, epochs, lr)

        # Set teachers to eval mode
        for teacher in self.teacher_models:
            teacher.eval()
            teacher.to(self.device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        history: dict[str, list[float]] = {"loss": [], "acc": []}

        logger.info(f"Starting knowledge distillation for {epochs} epochs")

        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for _batch_idx, (inputs, labels) in enumerate(train_data):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Student predictions
                student_logits = self.student(inputs)

                # Teacher predictions (ensemble average)
                with torch.no_grad():
                    teacher_logits_list = [teacher(inputs) for teacher in self.teacher_models]
                    teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)

                # Distillation loss (soft targets)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                    reduction="batchmean",
                ) * (self.temperature**2)

                # Hard label loss
                hard_loss = F.cross_entropy(student_logits, labels)

                # Combined loss
                loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_loss = epoch_loss / len(train_data)
            acc = 100.0 * correct / total
            history["loss"].append(avg_loss)
            history["acc"].append(acc)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        logger.info("Knowledge distillation completed")
        return history

    def _train_from_scratch(
        self,
        train_data: Any,
        val_data: Any | None,
        epochs: int,
        lr: float,
    ) -> dict[str, Any]:
        """Train student model from scratch (no teacher)."""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        history: dict[str, list[float]] = {"loss": [], "acc": []}

        logger.info(f"Training student from scratch for {epochs} epochs")

        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.student(inputs)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_loss = epoch_loss / len(train_data)
            acc = 100.0 * correct / total
            history["loss"].append(avg_loss)
            history["acc"].append(acc)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        return history

    def predict(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Run inference on frames.

        Args:
            frames: List of RGB frames (numpy arrays)

        Returns:
            Prediction dict with score and confidence
        """
        self.student.eval()

        # Preprocess frames
        processed = self._preprocess_frames(frames)
        batch = torch.from_numpy(processed).to(self.device)

        with torch.no_grad():
            logits = self.student(batch)
            probs = F.softmax(logits, dim=1)

        # Average across frames
        avg_prob = probs.mean(dim=0)
        fake_prob = float(avg_prob[1])
        confidence = float(max(avg_prob))

        return {
            "score": fake_prob,
            "confidence": confidence,
            "model": "tiny_model",
            "num_frames": len(frames),
        }

    def _preprocess_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Preprocess frames for model input."""
        processed = []
        for frame in frames:
            # Resize to 224x224
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((224, 224))

            # Convert to tensor and normalize
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            processed.append(arr)

        return np.array(processed)

    def save(self, path: str | Path) -> None:
        """Save student model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.student.state_dict(), path)
        logger.info(f"Tiny model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load student model."""
        self.student.load_state_dict(torch.load(path, map_location=self.device))
        self.student.eval()
        logger.info(f"Tiny model loaded from {path}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        param_count = sum(p.numel() for p in self.student.parameters())
        param_size = sum(p.numel() * p.element_size() for p in self.student.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.student.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)

        return {
            "name": "TinyModel (MobileNetV3-Small)",
            "parameters": param_count,
            "size_mb": round(size_mb, 2),
            "device": self.device,
            "temperature": self.temperature,
            "alpha": self.alpha,
        }
