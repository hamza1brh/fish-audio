"""
S1-Mini Production Inference Engine
====================================

A production-ready inference engine for Fish-Speech S1-Mini TTS model.

This module provides:
- Platform-aware compilation (Triton on Linux, fallback on Windows)
- Optimized VRAM management (no aggressive cache clearing)
- Request batching for improved throughput
- KV cache persistence and prefix caching
- Comprehensive monitoring and metrics
- Production-ready FastAPI server

Target Deployment:
- AWS SageMaker (Linux with Triton support)
- Local development on Windows (with graceful fallback)

Usage:
------
    from s1_mini import ProductionTTSEngine, EngineConfig

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
    )

    engine = ProductionTTSEngine(config)
    engine.warmup()

    # Single request
    result = engine.generate(text="Hello world", reference_audio=audio_bytes)

    # Or use the production server
    from s1_mini.server import create_app
    app = create_app(engine)

Architecture:
-------------
    Text Input
        │
        ▼
    ┌─────────────────────────────┐
    │  DualARTransformer          │  (text → semantic tokens)
    │  - 32 main transformer layers│
    │  - 4 fast layers for codebooks│
    │  - KV cache with persistence │
    └─────────────────────────────┘
        │
        ▼
    Semantic VQ Tokens (8 codebooks)
        │
        ▼
    ┌─────────────────────────────┐
    │  DAC Decoder                │  (tokens → audio)
    │  - Modified Descript Audio  │
    │    Codec                    │
    └─────────────────────────────┘
        │
        ▼
    Audio Output (44.1kHz WAV)

Key Improvements over Original:
-------------------------------
1. No aggressive VRAM clearing - models stay hot in memory
2. Platform detection - Triton on Linux, eager mode on Windows
3. Request batching - process multiple requests in parallel
4. KV cache persistence - reuse computation across requests
5. Comprehensive metrics - latency, throughput, VRAM tracking
6. Production server - health checks, timeouts, graceful shutdown

Author: Fish-Speech Team
License: Apache 2.0
"""

__version__ = "1.1.0"
__author__ = "Fish-Speech Team"

# Core exports
from s1_mini.config import EngineConfig, ServerConfig
from s1_mini.engine import (
    ProductionTTSEngine,
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationRequest,
    GenerationResponse,
)
from s1_mini.exceptions import (
    S1MiniError,
    ModelLoadError,
    InferenceError,
    TimeoutError,
    VRAMError,
)

# Batch processing
from s1_mini.batch_queue import BatchQueue, BatchRequest, Batch
from s1_mini.batch_worker import BatchedModelWorker, create_batched_model_worker

# Platform detection and compilation
from s1_mini.compilation import (
    get_platform_info,
    compile_model,
    check_compilation_support,
)

# Attention optimization
from s1_mini.attention import (
    get_attention_info,
    configure_optimal_attention,
    AttentionBackend,
)

# Backend system
from s1_mini.backends import (
    get_available_backends,
    get_best_backend,
    BackendType,
    PyTorchBackend,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "EngineConfig",
    "ServerConfig",
    # Engine
    "ProductionTTSEngine",
    "GenerationRequest",
    "GenerationResponse",
    "BatchGenerationRequest",
    "BatchGenerationResponse",
    # Batch processing
    "BatchQueue",
    "BatchRequest",
    "Batch",
    "BatchedModelWorker",
    "create_batched_model_worker",
    # Exceptions
    "S1MiniError",
    "ModelLoadError",
    "InferenceError",
    "TimeoutError",
    "VRAMError",
    # Platform detection
    "get_platform_info",
    "compile_model",
    "check_compilation_support",
    # Attention
    "get_attention_info",
    "configure_optimal_attention",
    "AttentionBackend",
    # Backends
    "get_available_backends",
    "get_best_backend",
    "BackendType",
    "PyTorchBackend",
]
