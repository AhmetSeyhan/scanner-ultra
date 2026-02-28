"""Scanner ULTRA — Model Quantization for Edge Deployment.

Quantize models to INT8/FP16 for reduced size and faster inference.
Maintains accuracy while reducing model size by 4x (FP32 → INT8).

References:
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- Post-Training Quantization: https://arxiv.org/abs/2004.10568
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize PyTorch models to INT8/FP16 for edge deployment.

    Supports:
    - Dynamic quantization (no calibration data needed)
    - Static quantization (with calibration data for best accuracy)
    - Quantization-aware training (QAT)
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Initialize quantizer.

        Args:
            model: PyTorch model to quantize
            device: Device for quantization (CPU recommended for deployment)
        """
        self.model = model.to(device)
        self.device = device
        self.quantized_model: nn.Module | None = None

    def quantize_dynamic(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """Apply dynamic quantization (weights-only, no calibration needed).

        Best for: Models with dynamic input sizes, quick deployment.
        Trade-off: Slightly lower accuracy than static quantization.

        Args:
            dtype: Quantization dtype (torch.qint8 or torch.float16)

        Returns:
            Dynamically quantized model
        """
        logger.info(f"Applying dynamic quantization (dtype={dtype})")

        # Dynamic quantization for linear and LSTM layers
        self.quantized_model = quant.quantize_dynamic(
            self.model,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=dtype,
        )

        self._log_size_reduction()
        return self.quantized_model

    def quantize_static(
        self,
        calibration_data: Any,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """Apply static quantization (weights + activations).

        Requires calibration data for activation range estimation.
        Best accuracy for deployment.

        Args:
            calibration_data: DataLoader with representative samples
            dtype: Quantization dtype (torch.qint8)

        Returns:
            Statically quantized model
        """
        logger.info("Applying static quantization with calibration")

        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig("x86")  # or 'qnnpack' for mobile
        prepared_model = quant.prepare(self.model, inplace=False)

        # Calibrate with representative data
        logger.info("Calibrating activations...")
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to(self.device)
                prepared_model(inputs)

        # Convert to quantized model
        self.quantized_model = quant.convert(prepared_model, inplace=False)

        self._log_size_reduction()
        return self.quantized_model

    def quantize_fp16(self) -> nn.Module:
        """Quantize model to FP16 (half precision).

        Simpler than INT8, still provides ~2x size reduction.
        Good for GPUs with FP16 support.

        Returns:
            FP16 quantized model
        """
        logger.info("Applying FP16 quantization")

        self.quantized_model = self.model.half()

        self._log_size_reduction()
        return self.quantized_model

    def quantization_aware_training(
        self,
        train_data: Any,
        epochs: int = 10,
        lr: float = 0.0001,
    ) -> nn.Module:
        """Perform quantization-aware training (QAT).

        Trains model with quantization simulation for best accuracy.
        Use this for maximum accuracy after quantization.

        Args:
            train_data: Training DataLoader
            epochs: Number of QAT epochs
            lr: Learning rate

        Returns:
            QAT-trained quantized model
        """
        logger.info(f"Starting quantization-aware training for {epochs} epochs")

        # Prepare model for QAT
        self.model.train()
        self.model.qconfig = quant.get_default_qat_qconfig("x86")
        prepared_model = quant.prepare_qat(self.model, inplace=False)

        # Train with quantization simulation
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = prepared_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(train_data)
                logger.info(f"QAT Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # Convert to quantized model
        prepared_model.eval()
        self.quantized_model = quant.convert(prepared_model, inplace=False)

        self._log_size_reduction()
        logger.info("Quantization-aware training completed")
        return self.quantized_model

    def benchmark(self, test_data: Any, num_samples: int = 100) -> dict[str, Any]:
        """Benchmark quantized model vs original.

        Args:
            test_data: Test DataLoader
            num_samples: Number of samples to test

        Returns:
            Benchmark results dict
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")

        logger.info(f"Benchmarking quantized model on {num_samples} samples")

        import time

        # Original model
        self.model.eval()
        original_times = []
        original_correct = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_data):
                if i >= num_samples:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                start = time.perf_counter()
                outputs = self.model(inputs)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                original_times.append(elapsed)
                _, predicted = outputs.max(1)
                original_correct += predicted.eq(labels).sum().item()

        # Quantized model
        self.quantized_model.eval()
        quantized_times = []
        quantized_correct = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_data):
                if i >= num_samples:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                start = time.perf_counter()
                outputs = self.quantized_model(inputs)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                quantized_times.append(elapsed)
                _, predicted = outputs.max(1)
                quantized_correct += predicted.eq(labels).sum().item()

        original_acc = 100.0 * original_correct / (num_samples * inputs.size(0))
        quantized_acc = 100.0 * quantized_correct / (num_samples * inputs.size(0))
        speedup = np.mean(original_times) / np.mean(quantized_times)

        results = {
            "original": {
                "accuracy": round(original_acc, 2),
                "avg_latency_ms": round(float(np.mean(original_times)), 2),
                "std_latency_ms": round(float(np.std(original_times)), 2),
            },
            "quantized": {
                "accuracy": round(quantized_acc, 2),
                "avg_latency_ms": round(float(np.mean(quantized_times)), 2),
                "std_latency_ms": round(float(np.std(quantized_times)), 2),
            },
            "speedup": round(speedup, 2),
            "accuracy_drop": round(original_acc - quantized_acc, 2),
        }

        logger.info(f"Benchmark complete - Speedup: {speedup:.2f}x, Acc drop: {results['accuracy_drop']:.2f}%")
        return results

    def _log_size_reduction(self) -> None:
        """Log model size reduction."""
        if self.quantized_model is None:
            return

        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model)
        reduction = (1 - quantized_size / original_size) * 100

        logger.info(f"Model size: {original_size:.2f} MB → {quantized_size:.2f} MB (reduction: {reduction:.1f}%)")

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024**2)

    def save(self, path: str | Path) -> None:
        """Save quantized model."""
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save quantized model state dict
        torch.save(self.quantized_model.state_dict(), path)

        # Also save as scripted model for deployment
        scripted_path = path.with_suffix(".pt")
        scripted_model = torch.jit.script(self.quantized_model)
        scripted_model.save(str(scripted_path))

        logger.info(f"Quantized model saved to {path} and {scripted_path}")

    def load(self, path: str | Path, scripted: bool = False) -> nn.Module:
        """Load quantized model.

        Args:
            path: Path to saved model
            scripted: Whether to load scripted model

        Returns:
            Loaded quantized model
        """
        path = Path(path)

        if scripted:
            self.quantized_model = torch.jit.load(str(path))
        else:
            # Load state dict (requires model architecture)
            if self.quantized_model is None:
                raise ValueError("Model architecture not initialized")
            self.quantized_model.load_state_dict(torch.load(path))

        self.quantized_model.eval()
        logger.info(f"Quantized model loaded from {path}")
        return self.quantized_model

    def get_quantization_info(self) -> dict[str, Any]:
        """Get quantization information."""
        if self.quantized_model is None:
            return {"quantized": False}

        return {
            "quantized": True,
            "original_size_mb": round(self._get_model_size(self.model), 2),
            "quantized_size_mb": round(self._get_model_size(self.quantized_model), 2),
            "device": self.device,
        }
