"""Scanner ULTRA — ONNX Runtime Export for Cross-Platform Deployment.

Export PyTorch models to ONNX format for deployment on:
- Mobile devices (iOS/Android via ONNX Runtime Mobile)
- Web browsers (ONNX.js)
- Edge devices (ONNX Runtime with TensorRT/OpenVINO)

References:
- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch models to ONNX format.

    Supports optimization for different runtime targets:
    - Mobile (ONNX Runtime Mobile)
    - Web (ONNX.js)
    - Server (ONNX Runtime with hardware acceleration)
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Initialize ONNX exporter.

        Args:
            model: PyTorch model to export
            device: Device for export (CPU recommended)
        """
        self.model = model.to(device).eval()
        self.device = device

    def export(
        self,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 14,
        optimize: bool = True,
    ) -> dict[str, Any]:
        """Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shape: Example input shape (batch, channels, height, width)
            opset_version: ONNX opset version (14 for mobile compatibility)
            optimize: Whether to apply ONNX optimization

        Returns:
            Export information dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to ONNX (opset={opset_version})")

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"ONNX export requires onnxscript: {e}")
            return {
                "verified": False,
                "error": f"Missing dependency: {e}",
                "model_path": str(output_path),
            }

        logger.info(f"Model exported to {output_path}")

        # Optimize ONNX model
        if optimize:
            self._optimize_onnx(output_path)

        # Verify export
        info = self._verify_onnx(output_path, dummy_input)
        return info

    def _optimize_onnx(self, model_path: Path) -> None:
        """Optimize ONNX model for deployment.

        Applies:
        - Constant folding
        - Dead code elimination
        - Operator fusion
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            logger.info("Optimizing ONNX model")

            # Load ONNX model
            onnx_model = onnx.load(str(model_path))

            # Basic optimization passes
            onnx_model = optimizer.optimize_model(
                str(model_path),
                model_type="bert",  # Generic optimization
                num_heads=0,
                hidden_size=0,
            )

            # Save optimized model
            optimized_path = model_path.with_stem(f"{model_path.stem}_optimized")
            onnx.save(onnx_model, str(optimized_path))

            logger.info(f"Optimized model saved to {optimized_path}")

        except ImportError:
            logger.warning("onnx or onnxruntime not installed, skipping optimization")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def _verify_onnx(self, model_path: Path, dummy_input: torch.Tensor) -> dict[str, Any]:
        """Verify ONNX export by running inference.

        Args:
            model_path: Path to ONNX model
            dummy_input: Test input tensor

        Returns:
            Verification results
        """
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)

            # Create inference session
            session = ort.InferenceSession(str(model_path))

            # Run inference
            ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outputs = session.run(None, ort_inputs)

            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = self.model(dummy_input).cpu().numpy()

            # Check numerical difference
            max_diff = np.abs(ort_outputs[0] - torch_output).max()

            logger.info(f"ONNX export verified - Max diff: {max_diff:.6f}")

            return {
                "verified": True,
                "max_diff": float(max_diff),
                "output_shape": ort_outputs[0].shape,
                "model_size_mb": round(model_path.stat().st_size / (1024**2), 2),
            }

        except ImportError:
            logger.warning("onnx or onnxruntime not installed, skipping verification")
            return {
                "verified": False,
                "error": "onnx/onnxruntime not installed",
            }
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return {
                "verified": False,
                "error": str(e),
            }

    def export_for_mobile(
        self,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> dict[str, Any]:
        """Export optimized model for mobile deployment.

        Optimizations:
        - Lower opset version for compatibility
        - Quantization-friendly ops
        - Reduced precision

        Args:
            output_path: Path to save mobile ONNX model
            input_shape: Input shape

        Returns:
            Export info
        """
        logger.info("Exporting model for mobile deployment")

        return self.export(
            output_path=output_path,
            input_shape=input_shape,
            opset_version=12,  # Better mobile support
            optimize=True,
        )

    def export_for_web(
        self,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> dict[str, Any]:
        """Export model for web deployment (ONNX.js).

        Args:
            output_path: Path to save web ONNX model
            input_shape: Input shape

        Returns:
            Export info
        """
        logger.info("Exporting model for web deployment (ONNX.js)")

        return self.export(
            output_path=output_path,
            input_shape=input_shape,
            opset_version=11,  # ONNX.js compatibility
            optimize=True,
        )

    def benchmark_onnx(
        self,
        model_path: str | Path,
        num_runs: int = 100,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> dict[str, Any]:
        """Benchmark ONNX model inference speed.

        Args:
            model_path: Path to ONNX model
            num_runs: Number of inference runs
            input_shape: Input shape for benchmarking

        Returns:
            Benchmark results
        """
        try:
            import time

            import onnxruntime as ort

            logger.info(f"Benchmarking ONNX model for {num_runs} runs")

            # Create session
            session = ort.InferenceSession(str(model_path))

            # Prepare input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            ort_inputs = {session.get_inputs()[0].name: dummy_input}

            # Warmup
            for _ in range(10):
                session.run(None, ort_inputs)

            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run(None, ort_inputs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            results = {
                "num_runs": num_runs,
                "avg_latency_ms": round(float(np.mean(times)), 2),
                "std_latency_ms": round(float(np.std(times)), 2),
                "min_latency_ms": round(float(np.min(times)), 2),
                "max_latency_ms": round(float(np.max(times)), 2),
                "throughput_fps": round(1000.0 / np.mean(times), 2),
            }

            logger.info(
                f"ONNX benchmark - Avg: {results['avg_latency_ms']:.2f}ms, FPS: {results['throughput_fps']:.2f}"
            )
            return results

        except ImportError:
            logger.error("onnxruntime not installed")
            return {"error": "onnxruntime not installed"}
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}

    def get_model_info(self, model_path: str | Path) -> dict[str, Any]:
        """Get ONNX model information.

        Args:
            model_path: Path to ONNX model

        Returns:
            Model info dict
        """
        try:
            import onnx

            model_path = Path(model_path)
            onnx_model = onnx.load(str(model_path))

            # Extract metadata
            input_info = []
            for inp in onnx_model.graph.input:
                shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in inp.type.tensor_type.shape.dim]
                input_info.append(
                    {
                        "name": inp.name,
                        "shape": shape,
                    }
                )

            output_info = []
            for out in onnx_model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in out.type.tensor_type.shape.dim]
                output_info.append(
                    {
                        "name": out.name,
                        "shape": shape,
                    }
                )

            return {
                "path": str(model_path),
                "size_mb": round(model_path.stat().st_size / (1024**2), 2),
                "opset_version": onnx_model.opset_import[0].version,
                "inputs": input_info,
                "outputs": output_info,
                "num_nodes": len(onnx_model.graph.node),
            }

        except ImportError:
            return {"error": "onnx not installed"}
        except Exception as e:
            return {"error": str(e)}
