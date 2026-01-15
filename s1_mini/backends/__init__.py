"""
S1-Mini Inference Backends
==========================

This module provides multiple inference backends for running S1-Mini
on different hardware configurations without Triton dependency.

Available Backends:
- PyTorchBackend: Default PyTorch with optimized compilation
- ONNXBackend: ONNX Runtime with CUDA Execution Provider
- TensorRTBackend: NVIDIA TensorRT for maximum performance

Usage:
------
    from s1_mini.backends import get_best_backend, BackendType

    # Auto-select best available backend
    backend = get_best_backend()
    audio = backend.generate("Hello world")

    # Force specific backend
    from s1_mini.backends import ONNXBackend
    backend = ONNXBackend(checkpoint_path="checkpoints/openaudio-s1-mini")
    backend.load()
    audio = backend.generate("Hello world")
"""

from .base import InferenceBackend, BackendType, BackendInfo
from .pytorch_backend import PyTorchBackend

# Conditionally import optional backends
_onnx_available = False
_tensorrt_available = False
ONNXBackend = None
TensorRTBackend = None

try:
    import onnxruntime
    from .onnx_backend import ONNXBackend
    _onnx_available = True
except ImportError:
    pass

try:
    import tensorrt
    from .tensorrt_backend import TensorRTBackend
    _tensorrt_available = True
except ImportError:
    pass


def get_available_backends() -> list[BackendType]:
    """
    Get list of available backends on this system.

    Returns:
        List of BackendType enums for available backends
    """
    available = [BackendType.PYTORCH]  # Always available

    if _onnx_available:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                available.append(BackendType.ONNX)
        except Exception:
            pass

    if _tensorrt_available:
        try:
            import tensorrt
            available.append(BackendType.TENSORRT)
        except Exception:
            pass

    return available


def get_best_backend(
    checkpoint_path: str = "checkpoints/openaudio-s1-mini",
    device: str = "cuda",
    precision: str = "float16",
) -> InferenceBackend:
    """
    Get the best available inference backend.

    Selection priority:
    1. TensorRT (if available and CUDA)
    2. ONNX Runtime (if available with CUDA EP)
    3. PyTorch (always available)

    Args:
        checkpoint_path: Path to model checkpoints
        device: Device to use ("cuda" or "cpu")
        precision: Model precision ("float16", "bfloat16", "float32")

    Returns:
        Best available InferenceBackend instance
    """
    from loguru import logger

    available = get_available_backends()
    logger.info(f"Available backends: {[b.value for b in available]}")

    # Try backends in order of preference
    if BackendType.TENSORRT in available and device == "cuda":
        try:
            backend = TensorRTBackend(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
            )
            logger.info("Selected TensorRT backend")
            return backend
        except Exception as e:
            logger.warning(f"TensorRT backend failed to initialize: {e}")

    if BackendType.ONNX in available and device == "cuda":
        try:
            backend = ONNXBackend(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
            )
            logger.info("Selected ONNX Runtime backend")
            return backend
        except Exception as e:
            logger.warning(f"ONNX backend failed to initialize: {e}")

    # Fall back to PyTorch (always available)
    backend = PyTorchBackend(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
    )
    logger.info("Selected PyTorch backend")
    return backend


__all__ = [
    "InferenceBackend",
    "BackendType",
    "BackendInfo",
    "PyTorchBackend",
    "ONNXBackend",
    "TensorRTBackend",
    "get_available_backends",
    "get_best_backend",
]
