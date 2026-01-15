"""
S1-Mini Attention Optimization
==============================

This module provides attention backend detection and optimization for Windows.
Since Triton-based Flash Attention isn't available on Windows, we need to
detect and use the best available attention implementation.

Attention Backend Priority (Windows):
1. Flash Attention 2 (if installed and GPU supports it)
2. Memory-Efficient Attention (xFormers)
3. PyTorch SDPA with cuDNN backend
4. PyTorch SDPA Math fallback

For reference, attention implementations ranked by speed:
- Flash Attention 2: Fastest (2-4x speedup)
- xFormers Memory Efficient: Fast (1.5-2x speedup)
- PyTorch SDPA (cuDNN): Good (1.2-1.5x speedup)
- PyTorch SDPA (Math): Baseline

Usage:
------
    from s1_mini.attention import (
        get_attention_info,
        configure_optimal_attention,
        get_sdpa_backend,
    )

    # Get available attention backends
    info = get_attention_info()
    print(f"Best backend: {info.recommended_backend}")

    # Configure for optimal performance
    configure_optimal_attention()
"""

import platform
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional, List

import torch
from loguru import logger


class AttentionBackend(Enum):
    """Available attention backends."""
    FLASH_ATTENTION_2 = "flash_attention_2"
    XFORMERS = "xformers"
    SDPA_FLASH = "sdpa_flash"
    SDPA_MEMORY_EFFICIENT = "sdpa_memory_efficient"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_MATH = "sdpa_math"
    UNKNOWN = "unknown"


@dataclass
class AttentionInfo:
    """Information about available attention backends."""

    # Availability flags
    flash_attention_2_available: bool
    xformers_available: bool
    sdpa_available: bool

    # SDPA backend availability
    sdpa_flash_available: bool
    sdpa_memory_efficient_available: bool
    sdpa_cudnn_available: bool

    # GPU info
    compute_capability: Optional[tuple]
    supports_flash_attention: bool  # Requires SM >= 8.0

    # Recommendations
    recommended_backend: AttentionBackend
    fallback_backends: List[AttentionBackend]


@lru_cache(maxsize=1)
def get_attention_info() -> AttentionInfo:
    """
    Detect available attention backends and recommend the best one.

    Returns:
        AttentionInfo with detection results and recommendations
    """
    system = platform.system()

    # Check GPU capability
    compute_capability = None
    supports_flash_attention = False

    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability(0)
        # Flash Attention requires Ampere (SM 8.0) or newer
        supports_flash_attention = compute_capability[0] >= 8

    # Check Flash Attention 2
    flash_attention_2_available = False
    try:
        import flash_attn
        if supports_flash_attention:
            flash_attention_2_available = True
            logger.debug(f"Flash Attention 2 available: v{flash_attn.__version__}")
    except ImportError:
        pass

    # Check xFormers
    xformers_available = False
    try:
        import xformers
        import xformers.ops
        xformers_available = True
        logger.debug(f"xFormers available: v{xformers.__version__}")
    except ImportError:
        pass

    # Check PyTorch SDPA
    sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    # Check SDPA backends
    sdpa_flash_available = False
    sdpa_memory_efficient_available = False
    sdpa_cudnn_available = False

    if sdpa_available and torch.cuda.is_available():
        try:
            from torch.nn.attention import SDPBackend

            # Test each backend availability
            # Flash SDPA requires SM >= 8.0
            sdpa_flash_available = supports_flash_attention

            # Memory efficient is available on most GPUs
            sdpa_memory_efficient_available = True

            # cuDNN attention (PyTorch 2.0+)
            if hasattr(SDPBackend, "CUDNN_ATTENTION"):
                sdpa_cudnn_available = True

        except ImportError:
            # Older PyTorch without SDPBackend enum
            sdpa_flash_available = supports_flash_attention
            sdpa_memory_efficient_available = True

    # Determine recommended backend
    recommended_backend = AttentionBackend.SDPA_MATH
    fallback_backends = []

    if flash_attention_2_available:
        recommended_backend = AttentionBackend.FLASH_ATTENTION_2
        fallback_backends = [
            AttentionBackend.SDPA_FLASH,
            AttentionBackend.XFORMERS,
            AttentionBackend.SDPA_MEMORY_EFFICIENT,
            AttentionBackend.SDPA_MATH,
        ]
    elif sdpa_flash_available:
        recommended_backend = AttentionBackend.SDPA_FLASH
        fallback_backends = [
            AttentionBackend.XFORMERS,
            AttentionBackend.SDPA_MEMORY_EFFICIENT,
            AttentionBackend.SDPA_MATH,
        ]
    elif xformers_available:
        recommended_backend = AttentionBackend.XFORMERS
        fallback_backends = [
            AttentionBackend.SDPA_MEMORY_EFFICIENT,
            AttentionBackend.SDPA_MATH,
        ]
    elif sdpa_memory_efficient_available:
        recommended_backend = AttentionBackend.SDPA_MEMORY_EFFICIENT
        fallback_backends = [AttentionBackend.SDPA_MATH]
    elif sdpa_cudnn_available:
        recommended_backend = AttentionBackend.SDPA_CUDNN
        fallback_backends = [AttentionBackend.SDPA_MATH]

    return AttentionInfo(
        flash_attention_2_available=flash_attention_2_available,
        xformers_available=xformers_available,
        sdpa_available=sdpa_available,
        sdpa_flash_available=sdpa_flash_available,
        sdpa_memory_efficient_available=sdpa_memory_efficient_available,
        sdpa_cudnn_available=sdpa_cudnn_available,
        compute_capability=compute_capability,
        supports_flash_attention=supports_flash_attention,
        recommended_backend=recommended_backend,
        fallback_backends=fallback_backends,
    )


def configure_optimal_attention() -> AttentionBackend:
    """
    Configure PyTorch to use the optimal attention backend.

    This function sets global PyTorch settings to prefer the
    fastest available attention implementation.

    Returns:
        The configured attention backend
    """
    info = get_attention_info()

    logger.info(f"Configuring attention backend: {info.recommended_backend.value}")

    # Enable Flash Attention in SDPA if available
    if info.sdpa_available:
        try:
            # Set default SDPA backend preferences
            # PyTorch will automatically select the fastest available
            torch.backends.cuda.enable_flash_sdp(info.sdpa_flash_available)
            torch.backends.cuda.enable_mem_efficient_sdp(info.sdpa_memory_efficient_available)

            # Enable math fallback as last resort
            torch.backends.cuda.enable_math_sdp(True)

            logger.debug(f"SDPA Flash enabled: {info.sdpa_flash_available}")
            logger.debug(f"SDPA Memory Efficient enabled: {info.sdpa_memory_efficient_available}")

        except AttributeError:
            # Older PyTorch versions don't have these functions
            logger.debug("SDPA backend configuration not available (older PyTorch)")

    # Configure cuDNN for attention
    if torch.cuda.is_available():
        try:
            # Enable cuDNN autotuning for convolutions in attention
            torch.backends.cudnn.benchmark = True

            # Allow TF32 for faster matmul on Ampere+
            if info.compute_capability and info.compute_capability[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.debug("Enabled TF32 for matmul operations")

        except Exception as e:
            logger.debug(f"cuDNN configuration skipped: {e}")

    return info.recommended_backend


def get_sdpa_backend():
    """
    Get the appropriate SDPA backend context manager.

    Returns a context manager that sets the optimal SDPA backend
    for the current hardware.

    Usage:
        with get_sdpa_backend():
            output = F.scaled_dot_product_attention(q, k, v)
    """
    info = get_attention_info()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if info.sdpa_flash_available:
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        elif info.sdpa_memory_efficient_available:
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        elif info.sdpa_cudnn_available and hasattr(SDPBackend, "CUDNN_ATTENTION"):
            return sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
        else:
            return sdpa_kernel(SDPBackend.MATH)

    except ImportError:
        # Return a no-op context manager for older PyTorch
        from contextlib import nullcontext
        return nullcontext()


def log_attention_info():
    """Log detailed attention backend information."""
    info = get_attention_info()

    logger.info("=" * 60)
    logger.info("S1-Mini Attention Backend Detection")
    logger.info("=" * 60)

    if info.compute_capability:
        logger.info(f"  GPU Compute Capability: SM {info.compute_capability[0]}.{info.compute_capability[1]}")
        logger.info(f"  Flash Attention Support: {'Yes' if info.supports_flash_attention else 'No (requires SM >= 8.0)'}")
    else:
        logger.info("  GPU: Not available")

    logger.info("")
    logger.info("  Available Backends:")
    logger.info(f"    Flash Attention 2:     {'Yes' if info.flash_attention_2_available else 'No'}")
    logger.info(f"    xFormers:              {'Yes' if info.xformers_available else 'No'}")
    logger.info(f"    PyTorch SDPA:          {'Yes' if info.sdpa_available else 'No'}")
    logger.info(f"      - Flash:             {'Yes' if info.sdpa_flash_available else 'No'}")
    logger.info(f"      - Memory Efficient:  {'Yes' if info.sdpa_memory_efficient_available else 'No'}")
    logger.info(f"      - cuDNN:             {'Yes' if info.sdpa_cudnn_available else 'No'}")

    logger.info("")
    logger.info(f"  Recommended Backend: {info.recommended_backend.value}")

    if info.fallback_backends:
        fallback_str = " -> ".join(b.value for b in info.fallback_backends)
        logger.info(f"  Fallback Chain: {fallback_str}")

    logger.info("=" * 60)


def get_attention_recommendations() -> List[str]:
    """
    Get recommendations for improving attention performance.

    Returns:
        List of actionable recommendations
    """
    info = get_attention_info()
    recommendations = []

    if not info.supports_flash_attention:
        recommendations.append(
            "GPU does not support Flash Attention (requires RTX 3000 series or newer). "
            "Consider upgrading for 2-4x attention speedup."
        )

    if info.supports_flash_attention and not info.flash_attention_2_available:
        recommendations.append(
            "Flash Attention 2 not installed. Install with: "
            "pip install flash-attn --no-build-isolation"
        )

    if not info.xformers_available:
        recommendations.append(
            "xFormers not installed. Install with: pip install xformers"
        )

    if info.recommended_backend == AttentionBackend.SDPA_MATH:
        recommendations.append(
            "Using slowest attention backend (Math). "
            "Install Flash Attention or xFormers for better performance."
        )

    if not recommendations:
        recommendations.append(
            f"Optimal attention backend configured: {info.recommended_backend.value}"
        )

    return recommendations


# =============================================================================
# Attention Wrapper Functions
# =============================================================================


def efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Compute attention using the best available backend.

    This function automatically selects the fastest attention
    implementation based on what's available on the system.

    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention

    Returns:
        Attention output tensor
    """
    info = get_attention_info()

    # Try Flash Attention 2 first
    if info.flash_attention_2_available and attn_mask is None:
        try:
            from flash_attn import flash_attn_func

            # Flash Attention expects [batch, seq_len, heads, head_dim]
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=is_causal,
            )

            return out.transpose(1, 2)

        except Exception as e:
            logger.debug(f"Flash Attention 2 failed, falling back: {e}")

    # Try xFormers
    if info.xformers_available:
        try:
            import xformers.ops as xops

            # xFormers expects [batch, seq_len, heads, head_dim]
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            out = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=xops.LowerTriangularMask() if is_causal else attn_mask,
                p=dropout_p,
            )

            return out.transpose(1, 2)

        except Exception as e:
            logger.debug(f"xFormers attention failed, falling back: {e}")

    # Fall back to PyTorch SDPA
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
