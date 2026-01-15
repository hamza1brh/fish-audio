"""
S1-Mini Platform-Aware Compilation
==================================

This module handles torch.compile configuration with automatic platform detection.
It provides Triton-based compilation on Linux and graceful fallback on Windows.

The Problem:
------------
Triton (the compiler backend for torch.compile's inductor) only supports Linux.
On Windows, attempting to use inductor backend causes errors or extremely slow
compilation. This module detects the platform and configures the optimal backend.

Platform Support:
-----------------
    Linux + CUDA:   inductor backend (Triton) -> Fast kernels, best performance
    Windows + CUDA: eager backend (no Triton) -> Still faster than no compile
    macOS + MPS:    eager backend             -> Limited optimization
    CPU only:       disabled                  -> Compilation overhead not worth it

Triton Configuration:
---------------------
When Triton is available (Linux), we configure optimal settings:
- coordinate_descent_tuning: Optimize instruction scheduling
- unique_kernel_names: Better debugging/profiling
- fx_graph_cache: Cache compiled graphs across runs (faster restarts)

Usage:
------
    from s1_mini.compilation import compile_model, get_platform_info

    # Automatic platform detection
    compiled_model = compile_model(model, backend="auto")

    # Force specific backend
    compiled_model = compile_model(model, backend="inductor")

    # Check platform capabilities
    info = get_platform_info()
    print(f"Triton available: {info['triton_available']}")

Environment Variables:
----------------------
    S1_MINI_COMPILE=true/false       Enable/disable compilation
    S1_MINI_COMPILE_BACKEND=auto     Backend selection
    S1_MINI_COMPILE_MODE=reduce-overhead  Compilation mode
"""

import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Callable, Any

import torch
from loguru import logger


# =============================================================================
# Platform Detection
# =============================================================================


@dataclass
class PlatformInfo:
    """
    Information about the current platform and available optimizations.

    Attributes:
        system: Operating system name (Linux, Windows, Darwin)
        python_version: Python version string
        torch_version: PyTorch version string
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version if available
        triton_available: Whether Triton can be used
        recommended_backend: Recommended torch.compile backend
        device_name: GPU device name if available
        device_capability: CUDA compute capability (e.g., (8, 6) for RTX 3090)
    """

    system: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    triton_available: bool
    recommended_backend: str
    device_name: Optional[str]
    device_capability: Optional[tuple]


@lru_cache(maxsize=1)
def get_platform_info() -> PlatformInfo:
    """
    Detect platform capabilities and recommended settings.

    This function is cached - it only runs once per process.

    Returns:
        PlatformInfo with detected platform capabilities
    """
    system = platform.system()
    cuda_available = torch.cuda.is_available()

    # CUDA details
    cuda_version = None
    device_name = None
    device_capability = None

    if cuda_available:
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        device_capability = torch.cuda.get_device_capability(0)

    # Triton is only available on Linux with CUDA
    # We check by trying to import it
    triton_available = False
    if system == "Linux" and cuda_available:
        try:
            import triton

            triton_available = True
        except ImportError:
            # Triton not installed, but could be available via torch
            # Check if torch._inductor is configured for triton
            try:
                from torch._inductor import config as inductor_config

                if hasattr(inductor_config, "triton"):
                    triton_available = True
            except ImportError:
                pass

    # Determine recommended backend
    if not cuda_available:
        recommended_backend = "disabled"
    elif triton_available:
        recommended_backend = "inductor"
    elif system == "Windows":
        # Windows without Triton - try cudagraphs first, then eager
        # cudagraphs can provide ~1.3-1.5x speedup by reducing kernel launch overhead
        try:
            from torch._dynamo import list_backends
            if "cudagraphs" in list_backends():
                recommended_backend = "cudagraphs"
            else:
                recommended_backend = "eager"
        except Exception:
            recommended_backend = "eager"
    else:
        recommended_backend = "eager"

    return PlatformInfo(
        system=system,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        triton_available=triton_available,
        recommended_backend=recommended_backend,
        device_name=device_name,
        device_capability=device_capability,
    )


def log_platform_info():
    """Log platform information for debugging."""
    info = get_platform_info()
    logger.info("=" * 60)
    logger.info("S1-Mini Platform Detection")
    logger.info("=" * 60)
    logger.info(f"  OS:              {info.system}")
    logger.info(f"  Python:          {info.python_version}")
    logger.info(f"  PyTorch:         {info.torch_version}")
    logger.info(f"  CUDA Available:  {info.cuda_available}")
    if info.cuda_available:
        logger.info(f"  CUDA Version:    {info.cuda_version}")
        logger.info(f"  GPU:             {info.device_name}")
        logger.info(f"  Compute Cap:     {info.device_capability}")
    logger.info(f"  Triton Available: {info.triton_available}")
    logger.info(f"  Recommended Backend: {info.recommended_backend}")
    logger.info("=" * 60)


# =============================================================================
# Triton Configuration
# =============================================================================


def configure_triton_settings():
    """
    Configure Triton/inductor settings for optimal performance.

    This should be called BEFORE any torch.compile() calls.
    Only effective on Linux with Triton available.

    Settings configured:
    - coordinate_descent_tuning: Better instruction scheduling
    - unique_kernel_names: Improved debugging/profiling
    - fx_graph_cache: Faster subsequent runs (caches compiled graphs)
    """
    info = get_platform_info()

    if not info.triton_available:
        logger.debug("Triton not available, skipping Triton configuration")
        return

    try:
        import torch._inductor.config as inductor_config

        # Enable coordinate descent tuning for better kernel scheduling
        # This can improve performance by 5-15% for attention-heavy models
        inductor_config.coordinate_descent_tuning = True
        logger.debug("Enabled coordinate_descent_tuning")

        # Use unique kernel names for better profiling
        # Makes NVIDIA Nsight and other profilers more useful
        if hasattr(inductor_config, "triton"):
            inductor_config.triton.unique_kernel_names = True
            logger.debug("Enabled triton.unique_kernel_names")

        # Enable FX graph cache for faster subsequent compilations
        # This caches the compiled graphs to disk, speeding up restarts
        if hasattr(inductor_config, "fx_graph_cache"):
            inductor_config.fx_graph_cache = True
            logger.debug("Enabled fx_graph_cache")

        # Additional optimizations for newer PyTorch versions
        if hasattr(inductor_config, "triton") and hasattr(
            inductor_config.triton, "cudagraphs"
        ):
            # CUDA graphs can significantly improve performance for
            # fixed-size operations by reducing launch overhead
            inductor_config.triton.cudagraphs = True
            logger.debug("Enabled CUDA graphs")

        logger.info("Triton configuration completed successfully")

    except ImportError as e:
        logger.warning(f"Failed to configure Triton settings: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error configuring Triton: {e}")


def configure_windows_fallback():
    """
    Configure optimized fallback settings for Windows (no Triton).

    On Windows, we can still benefit from several optimizations:
    - CUDA graphs for reducing kernel launch overhead
    - cuDNN autotuning for convolutions
    - Memory-efficient attention when available
    - Aggressive kernel autotuning via max-autotune mode

    These optimizations can provide 20-40% speedup over naive eager execution.
    """
    info = get_platform_info()

    if info.system != "Windows":
        return

    logger.info("Configuring optimized Windows fallback (no Triton)")

    try:
        from torch._inductor import config as inductor_config

        # Disable Triton-specific settings that would fail on Windows
        if hasattr(inductor_config, "triton"):
            # Don't try to use Triton kernels
            if hasattr(inductor_config.triton, "unique_kernel_names"):
                inductor_config.triton.unique_kernel_names = False

        # Disable coordinate descent tuning (requires Triton)
        inductor_config.coordinate_descent_tuning = False

        # Enable optimizations that work without Triton
        # These use cuDNN and cuBLAS which are available on Windows

        # Enable max-autotune for better kernel selection
        if hasattr(inductor_config, "max_autotune"):
            inductor_config.max_autotune = True
            logger.debug("Enabled max_autotune for kernel selection")

        # Enable GEMM autotuning (matrix multiplication optimization)
        if hasattr(inductor_config, "max_autotune_gemm"):
            inductor_config.max_autotune_gemm = True
            logger.debug("Enabled max_autotune_gemm")

        # Enable graph cache for faster subsequent runs
        if hasattr(inductor_config, "fx_graph_cache"):
            inductor_config.fx_graph_cache = True
            logger.debug("Enabled fx_graph_cache")

        logger.info("Windows fallback configuration completed")

    except Exception as e:
        logger.debug(f"Windows fallback configuration skipped: {e}")

    # Configure cuDNN for optimal performance
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.debug("Enabled cuDNN benchmark and TF32")
    except Exception as e:
        logger.debug(f"cuDNN configuration skipped: {e}")


# =============================================================================
# Model Compilation
# =============================================================================


def compile_model(
    model: torch.nn.Module,
    backend: str = "auto",
    mode: str = "auto",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> torch.nn.Module:
    """
    Compile a model with platform-appropriate settings.

    This function automatically detects the platform and selects
    the optimal compilation backend. On Linux with CUDA, it uses
    the inductor backend (Triton). On Windows, it uses optimized
    eager mode with max-autotune for best performance without Triton.

    Args:
        model: PyTorch model to compile
        backend: Compilation backend
            - "auto": Auto-detect based on platform (recommended)
            - "inductor": Use inductor (requires Triton/Linux)
            - "eager": Use eager mode (works everywhere)
            - "aot_eager": Ahead-of-time eager
            - "disabled": Return model unchanged
        mode: Compilation mode
            - "auto": Auto-select based on platform
            - "default": Balanced compilation
            - "reduce-overhead": Minimize runtime overhead
            - "max-autotune": Maximum optimization (slower compilation)
        fullgraph: Whether to require full graph capture
        dynamic: Whether to support dynamic shapes

    Returns:
        Compiled model (or original if compilation disabled/failed)

    Example:
        >>> model = DualARTransformer(config)
        >>> compiled = compile_model(model, backend="auto")
        >>> # On Linux: uses inductor with Triton
        >>> # On Windows: uses eager with max-autotune
    """
    info = get_platform_info()

    # Resolve "auto" backend
    if backend == "auto":
        backend = info.recommended_backend
        logger.info(f"Auto-detected compilation backend: {backend}")

    # Resolve "auto" mode based on platform
    if mode == "auto":
        if info.triton_available:
            # On Linux with Triton, reduce-overhead is best for inference
            mode = "reduce-overhead"
        else:
            # On Windows without Triton, max-autotune provides better
            # kernel selection via cuDNN/cuBLAS autotuning
            mode = "max-autotune"
        logger.info(f"Auto-selected compilation mode: {mode}")

    # Handle disabled compilation
    if backend == "disabled":
        logger.info("Model compilation disabled")
        return model

    # Configure platform-specific settings
    if info.triton_available:
        configure_triton_settings()
    else:
        configure_windows_fallback()

    # Validate backend for current platform
    if backend == "inductor" and not info.triton_available:
        logger.warning(
            f"inductor backend requested but Triton not available on {info.system}. "
            f"Falling back to cudagraphs backend."
        )
        backend = "cudagraphs"

    # For cudagraphs, we need to handle potential limitations
    if backend == "cudagraphs":
        logger.info("Using cudagraphs backend (reduces kernel launch overhead)")
        # cudagraphs works best with fixed shapes, but can handle some dynamism
        # If it fails, we'll fall back to eager

    # Perform compilation
    logger.info(f"Compiling model with backend={backend}, mode={mode}")

    try:
        compiled_model = torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info("Model compilation successful")
        return compiled_model

    except Exception as e:
        logger.error(f"Model compilation failed: {e}")
        logger.warning("Returning uncompiled model")
        return model


def compile_function(
    fn: Callable,
    backend: str = "auto",
    mode: str = "reduce-overhead",
) -> Callable:
    """
    Compile a function with platform-appropriate settings.

    Similar to compile_model but for standalone functions.
    Useful for compiling decode_one_token and similar functions.

    Args:
        fn: Function to compile
        backend: Compilation backend (see compile_model)
        mode: Compilation mode (see compile_model)

    Returns:
        Compiled function (or original if compilation failed)

    Example:
        >>> @compile_function
        ... def decode_one_token(model, x, input_pos):
        ...     return model.forward(x, input_pos)
    """
    info = get_platform_info()

    if backend == "auto":
        backend = info.recommended_backend

    if backend == "disabled":
        return fn

    if info.triton_available:
        configure_triton_settings()
    else:
        configure_windows_fallback()

    if backend == "inductor" and not info.triton_available:
        backend = "eager"

    try:
        compiled_fn = torch.compile(fn, backend=backend, mode=mode)
        logger.debug(f"Function {fn.__name__} compiled successfully")
        return compiled_fn
    except Exception as e:
        logger.warning(f"Function compilation failed: {e}")
        return fn


# =============================================================================
# Compilation Warmup
# =============================================================================


def warmup_compiled_model(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    num_warmup_runs: int = 3,
) -> None:
    """
    Warm up a compiled model to trigger JIT compilation.

    The first few runs of a compiled model are slow because
    the JIT compiler is generating optimized kernels. This
    function runs warmup iterations to ensure the model is
    fully compiled before serving production traffic.

    Args:
        model: Compiled model to warm up
        sample_input: Sample input tensor for warmup
        num_warmup_runs: Number of warmup iterations

    Note:
        This function runs in inference mode and does not
        compute gradients.
    """
    logger.info(f"Warming up model with {num_warmup_runs} iterations...")

    model.eval()

    with torch.inference_mode():
        for i in range(num_warmup_runs):
            try:
                _ = model(sample_input)
                logger.debug(f"Warmup iteration {i + 1}/{num_warmup_runs} complete")
            except Exception as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")

    # Clear any cached tensors from warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info("Model warmup complete")


# =============================================================================
# Diagnostic Functions
# =============================================================================


def check_compilation_support() -> dict:
    """
    Check what compilation features are supported on this platform.

    Returns:
        Dictionary with support status for various features

    Example:
        >>> support = check_compilation_support()
        >>> if support["inductor"]:
        ...     print("Full Triton acceleration available!")
    """
    info = get_platform_info()

    support = {
        "torch_compile": hasattr(torch, "compile"),
        "inductor": info.triton_available,
        "eager": True,  # Always available
        "aot_eager": True,  # Always available
        "cuda_graphs": info.cuda_available and info.triton_available,
        "flash_attention": False,
        "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    }

    # Check for Flash Attention
    if info.cuda_available:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # Flash attention requires compute capability >= 8.0
            if info.device_capability and info.device_capability[0] >= 8:
                support["flash_attention"] = True
        except ImportError:
            pass

    return support


def get_compilation_recommendations() -> list[str]:
    """
    Get recommendations for optimal compilation settings.

    Returns:
        List of recommendation strings
    """
    info = get_platform_info()
    recommendations = []

    if not info.cuda_available:
        recommendations.append(
            "CUDA not available - compilation provides minimal benefit on CPU"
        )
        return recommendations

    if info.triton_available:
        recommendations.append(
            "Triton available - use backend='inductor' for best performance"
        )
        recommendations.append(
            "Consider mode='max-autotune' for production (slower compile, faster run)"
        )
    else:
        recommendations.append(
            f"Triton not available on {info.system} - using eager fallback"
        )
        if info.system == "Windows":
            recommendations.append(
                "Consider using WSL2 for better performance (Triton support)"
            )

    if info.device_capability:
        major, minor = info.device_capability
        if major >= 8:
            recommendations.append(
                f"GPU supports Flash Attention (SM {major}.{minor})"
            )
        else:
            recommendations.append(
                f"GPU (SM {major}.{minor}) does not support Flash Attention"
            )

    return recommendations


# =============================================================================
# Module Initialization
# =============================================================================


# Log platform info on import (can be disabled via environment variable)
import os

if os.environ.get("S1_MINI_LOG_PLATFORM", "true").lower() == "true":
    # Only log in main process
    if __name__ != "__mp_main__":
        log_platform_info()
