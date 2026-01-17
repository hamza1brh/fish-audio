"""
S1-Mini Model Manager
=====================

This module handles the lifecycle of TTS models:
- Loading LLAMA (text-to-semantic) model
- Loading DAC (semantic-to-audio) decoder
- Model warmup for optimal performance
- Persistent model state management
- Optimized VRAM handling

Key Improvements over Original:
-------------------------------
1. Models loaded ONCE and kept in memory persistently
2. No aggressive VRAM clearing after each generation
3. Platform-aware compilation (Triton on Linux, fallback on Windows)
4. Proper warmup to ensure JIT compilation completes
5. Health check methods for production monitoring

Architecture:
-------------
    ModelManager
    ├── LLAMA Model (DualARTransformer)
    │   ├── 32 main transformer layers
    │   ├── 4 fast layers for codebook prediction
    │   └── KV cache (persistent, not cleared)
    │
    └── DAC Decoder
        └── Audio reconstruction from VQ tokens

Usage:
------
    from s1_mini.model_manager import ModelManager
    from s1_mini.config import EngineConfig

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
    )

    manager = ModelManager(config)
    manager.load_models()
    manager.warmup()

    # Models are now ready for inference
    # They will stay in VRAM until explicitly unloaded

Environment Variables:
----------------------
    S1_MINI_CHECKPOINT_PATH   Model checkpoint directory
    S1_MINI_DEVICE            Target device (cuda, cpu, mps)
    S1_MINI_PRECISION         Model precision (float16, bfloat16, float32)
    S1_MINI_COMPILE           Enable torch.compile (true/false)
"""

import gc
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Callable, Any

import torch
from loguru import logger

from s1_mini.config import EngineConfig
from s1_mini.compilation import (
    compile_model,
    compile_function,
    get_platform_info,
    log_platform_info,
)
from s1_mini.exceptions import ModelLoadError, VRAMError, OOMError
from s1_mini.utils import Timer, get_vram_info, clear_vram_cache


# =============================================================================
# Model Loading Functions
# =============================================================================


def load_llama_model(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    compile: bool = False,
    compile_backend: str = "auto",
) -> Tuple[Any, Callable]:
    """
    Load the LLAMA (text-to-semantic) model.

    This function loads the DualARTransformer model and optionally
    compiles it for faster inference.

    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Target device (cuda, cpu, mps)
        precision: Model precision (torch.float16, torch.bfloat16, etc.)
        compile: Whether to use torch.compile
        compile_backend: Compilation backend (auto, inductor, eager)

    Returns:
        Tuple of (model, decode_one_token_function)

    Raises:
        ModelLoadError: If model loading fails
    """
    logger.info(f"Loading LLAMA model from {checkpoint_path}")

    try:
        # Import here to avoid circular imports
        from fish_speech.models.text2semantic.llama import DualARTransformer
        from fish_speech.models.text2semantic.inference import decode_one_token_ar

        # Load model from pretrained checkpoint
        with Timer("model_load"):
            model = DualARTransformer.from_pretrained(
                checkpoint_path, load_weights=True
            )

        # Move to device and set precision
        model = model.to(device=device, dtype=precision)
        model.eval()

        logger.info(f"Model loaded to {device} with {precision}")

        # Pre-create fixed parameter tensors to avoid runtime allocation
        # These are used during sampling and should not be recreated per-request
        model.fixed_temperature = torch.tensor(
            0.7, device=device, dtype=torch.float
        )
        model.fixed_top_p = torch.tensor(
            0.7, device=device, dtype=torch.float
        )
        model.fixed_repetition_penalty = torch.tensor(
            1.5, device=device, dtype=torch.float
        )

        # Mark cache as not initialized (will be set up on first inference)
        model._cache_setup_done = False

        # Set up the decode function
        decode_one_token = decode_one_token_ar

        # Optionally compile the decode function
        if compile:
            logger.info(f"Compiling decode function with backend={compile_backend}")
            decode_one_token = compile_function(
                decode_one_token,
                backend=compile_backend,
                mode="reduce-overhead",
            )

        # Log model statistics
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        return model, decode_one_token

    except FileNotFoundError as e:
        raise ModelLoadError(
            f"Checkpoint not found: {checkpoint_path}",
            checkpoint_path=checkpoint_path,
        ) from e
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load LLAMA model: {e}",
            checkpoint_path=checkpoint_path,
        ) from e


def load_decoder_model(
    config_name: str,
    checkpoint_path: str,
    device: str,
) -> Any:
    """
    Load the DAC decoder model.

    The DAC (Descript Audio Codec) decoder converts VQ tokens
    back into audio waveforms.

    Args:
        config_name: Hydra config name (e.g., "modded_dac_vq")
        checkpoint_path: Path to decoder checkpoint
        device: Target device

    Returns:
        Loaded DAC decoder model

    Raises:
        ModelLoadError: If loading fails
    """
    logger.info(f"Loading decoder from {checkpoint_path}")

    try:
        from fish_speech.models.dac.inference import load_model

        with Timer("decoder_load"):
            model = load_model(
                config_name=config_name,
                checkpoint_path=checkpoint_path,
                device=device,
            )

        logger.info(f"Decoder loaded to {device}")
        return model

    except FileNotFoundError as e:
        raise ModelLoadError(
            f"Decoder checkpoint not found: {checkpoint_path}",
            checkpoint_path=checkpoint_path,
        ) from e
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load decoder: {e}",
            checkpoint_path=checkpoint_path,
        ) from e


# =============================================================================
# Thread-Safe Model Queue
# =============================================================================


def create_model_worker(
    model: Any,
    decode_one_token: Callable,
    device: str,
    max_batch_size: int = 1,
) -> queue.Queue:
    """
    Create a thread-safe worker for model inference.

    This creates a worker thread that processes inference requests
    from a queue. The model stays loaded in the worker thread and
    is never unloaded between requests.

    Key Difference from Original:
    - Model is passed in already loaded (not loaded inside worker)
    - No torch.cuda.empty_cache() after each request
    - KV cache is set up once and reused

    Args:
        model: Pre-loaded LLAMA model
        decode_one_token: Decode function (possibly compiled)
        device: Target device

    Returns:
        Input queue for submitting requests
    """
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        """
        Worker thread that processes inference requests.

        The worker:
        1. Sets up KV cache once on first request
        2. Processes requests from the queue
        3. Puts results in per-request response queues
        4. Does NOT clear VRAM cache after each request (key optimization)
        """
        nonlocal model

        # Import generation function
        from fish_speech.models.text2semantic.inference import generate_long

        # Set up KV cache once (persistent across all requests)
        # Use max_batch_size for batched inference support
        with torch.device(device):
            model.setup_caches(
                max_batch_size=max_batch_size,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True
        logger.info(f"KV cache initialized with max_batch_size={max_batch_size}")

        # Signal that initialization is complete
        init_event.set()

        # Process requests indefinitely
        while True:
            try:
                # Get next request (blocks until available)
                item = input_queue.get()

                # None signals shutdown
                if item is None:
                    logger.info("Worker received shutdown signal")
                    break

                # Extract request data
                kwargs = item.request
                response_queue = item.response_queue

                try:
                    # Run generation
                    for chunk in generate_long(
                        model=model,
                        decode_one_token=decode_one_token,
                        **kwargs,
                    ):
                        # Put each chunk in response queue
                        from fish_speech.models.text2semantic.inference import (
                            WrappedGenerateResponse,
                        )

                        response_queue.put(
                            WrappedGenerateResponse(
                                status="success",
                                response=chunk,
                            )
                        )

                    # IMPORTANT: We do NOT clear VRAM cache here!
                    # This is a key optimization - keeping the cache hot
                    # improves subsequent request latency.
                    #
                    # Original code had:
                    # if torch.cuda.is_available():
                    #     torch.cuda.empty_cache()
                    #
                    # We only clear on OOM or explicit request.

                except torch.cuda.OutOfMemoryError as e:
                    # Only clear cache on OOM
                    logger.warning(f"OOM during generation, clearing cache: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    from fish_speech.models.text2semantic.inference import (
                        WrappedGenerateResponse,
                    )

                    response_queue.put(
                        WrappedGenerateResponse(
                            status="error",
                            response=OOMError(str(e)),
                        )
                    )

                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    from fish_speech.models.text2semantic.inference import (
                        WrappedGenerateResponse,
                    )

                    response_queue.put(
                        WrappedGenerateResponse(status="error", response=e)
                    )

            except Exception as e:
                logger.error(f"Worker error: {e}")

    # Start worker thread
    worker_thread = threading.Thread(target=worker, daemon=True, name="LLAMAWorker")
    worker_thread.start()

    # Wait for initialization to complete
    init_event.wait(timeout=60)
    if not init_event.is_set():
        raise ModelLoadError("Worker initialization timed out")

    logger.info("Model worker started and ready")
    return input_queue


# =============================================================================
# Model Manager Class
# =============================================================================


class ModelManager:
    """
    Manages the lifecycle of TTS models.

    This class handles:
    - Loading LLAMA and DAC models
    - Setting up the inference worker thread
    - Model warmup for optimal first-request latency
    - Health checks for production monitoring
    - Clean shutdown

    Key Improvements:
    -----------------
    1. Persistent Model Loading: Models are loaded once and stay in VRAM
    2. No Aggressive Cache Clearing: VRAM cache only cleared on OOM
    3. Platform-Aware Compilation: Triton on Linux, eager on Windows
    4. Proper Warmup: Ensures JIT compilation before serving traffic

    Attributes:
        config: Engine configuration
        llama_model: Loaded LLAMA model
        decoder_model: Loaded DAC decoder
        llama_queue: Thread-safe queue for LLAMA requests
        is_ready: Whether models are loaded and warmed up

    Usage:
        >>> manager = ModelManager(config)
        >>> manager.load_models()
        >>> manager.warmup()
        >>> assert manager.is_ready
        >>> # Now ready for inference
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize the model manager.

        Args:
            config: Engine configuration
        """
        self.config = config

        # Model state
        self.llama_model = None
        self.decoder_model = None
        self.decode_one_token = None
        self.llama_queue = None

        # Status flags
        self._models_loaded = False
        self._warmed_up = False

        # Log platform info on initialization
        log_platform_info()

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and warmed up."""
        return self._models_loaded and self._warmed_up

    @property
    def device(self) -> torch.device:
        """Get the target device."""
        return torch.device(self.config.device)

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_models(self) -> None:
        """
        Load all models (LLAMA and DAC decoder).

        This method loads models and sets up the inference worker.
        It should be called once at startup.

        Raises:
            ModelLoadError: If any model fails to load
        """
        if self._models_loaded:
            logger.warning("Models already loaded, skipping")
            return

        logger.info("=" * 60)
        logger.info("Loading S1-Mini Models")
        logger.info("=" * 60)

        # Log VRAM before loading
        vram_before = get_vram_info()
        logger.info(f"VRAM before loading: {vram_before['used_gb']:.2f} GB used")

        # Resolve paths
        checkpoint_path = Path(self.config.checkpoint_path)
        llama_path = str(checkpoint_path)
        decoder_path = str(checkpoint_path / "codec.pth")

        # Load LLAMA model
        self.llama_model, self.decode_one_token = load_llama_model(
            checkpoint_path=llama_path,
            device=self.config.device,
            precision=self.config.torch_dtype,
            compile=self.config.should_compile,
            compile_backend=self.config.resolved_compile_backend,
        )

        # Load DAC decoder
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_path,
            device=self.config.device,
        )

        # Create worker thread
        # Note: The regular worker uses batch_size=1 for sequential processing.
        # Batched inference uses the BatchedModelWorker with larger batch sizes.
        self.llama_queue = create_model_worker(
            model=self.llama_model,
            decode_one_token=self.decode_one_token,
            device=self.config.device,
            max_batch_size=1,  # Sequential worker always uses batch_size=1
        )

        self._models_loaded = True

        # Log VRAM after loading
        vram_after = get_vram_info()
        logger.info(f"VRAM after loading: {vram_after['used_gb']:.2f} GB used")
        logger.info(
            f"Model memory footprint: {vram_after['used_gb'] - vram_before['used_gb']:.2f} GB"
        )

        logger.info("=" * 60)
        logger.info("All models loaded successfully")
        logger.info("=" * 60)

    def warmup(self, num_iterations: int = 2) -> None:
        """
        Warm up models to ensure JIT compilation completes.

        The first few inference runs are slow because:
        1. CUDA kernels are being compiled
        2. torch.compile is tracing and optimizing
        3. Memory pools are being allocated

        This method runs dummy inferences to "warm up" the models,
        ensuring production traffic gets optimal latency.

        Args:
            num_iterations: Number of warmup iterations

        Raises:
            RuntimeError: If warmup fails
        """
        if not self._models_loaded:
            raise RuntimeError("Models must be loaded before warmup")

        if self._warmed_up:
            logger.warning("Models already warmed up, skipping")
            return

        logger.info(f"Warming up models ({num_iterations} iterations)...")

        # Import required classes
        from fish_speech.utils.schema import ServeTTSRequest

        # Create warmup request
        warmup_request = ServeTTSRequest(
            text="Hello world. This is a warmup request.",
            references=[],
            reference_id=None,
            max_new_tokens=512,  # Shorter for warmup
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )

        for i in range(num_iterations):
            logger.info(f"Warmup iteration {i + 1}/{num_iterations}")

            try:
                with Timer(f"warmup_{i + 1}"):
                    # Run inference
                    from fish_speech.inference_engine import TTSInferenceEngine

                    engine = TTSInferenceEngine(
                        llama_queue=self.llama_queue,
                        decoder_model=self.decoder_model,
                        precision=self.config.torch_dtype,
                        compile=self.config.should_compile,
                    )

                    # Consume all results (ignore output)
                    for result in engine.inference(warmup_request):
                        pass

                logger.info(f"Warmup iteration {i + 1} complete")

            except Exception as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")
                # Continue with next iteration

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._warmed_up = True
        logger.info("Model warmup complete")

    # =========================================================================
    # Health Checks
    # =========================================================================

    def health_check(self) -> dict:
        """
        Perform a health check on the models.

        Returns:
            Dictionary with health status:
            - healthy: Overall health status
            - models_loaded: Whether models are loaded
            - warmed_up: Whether warmup completed
            - vram: VRAM usage information
            - device: Target device
        """
        vram = get_vram_info()

        return {
            "healthy": self.is_ready,
            "models_loaded": self._models_loaded,
            "warmed_up": self._warmed_up,
            "device": self.config.device,
            "precision": self.config.precision,
            "compile_enabled": self.config.should_compile,
            "compile_backend": self.config.resolved_compile_backend,
            "vram": {
                "total_gb": round(vram["total_gb"], 2),
                "used_gb": round(vram["used_gb"], 2),
                "free_gb": round(vram["free_gb"], 2),
                "utilization_percent": round(vram["utilization"], 1),
            },
        }

    def get_vram_usage(self) -> dict:
        """Get current VRAM usage."""
        return get_vram_info()

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear_cache(self, force: bool = False) -> None:
        """
        Clear VRAM cache.

        This should only be called in exceptional circumstances
        (e.g., after OOM error or before shutdown).

        Args:
            force: If True, also run garbage collection
        """
        logger.info("Clearing VRAM cache")
        clear_vram_cache(force=force)

    def shutdown(self) -> None:
        """
        Cleanly shut down the model manager.

        This method:
        1. Signals the worker thread to stop
        2. Clears VRAM cache
        3. Releases model references
        """
        logger.info("Shutting down model manager")

        # Signal worker to stop
        if self.llama_queue is not None:
            self.llama_queue.put(None)

        # Clear cache
        self.clear_cache(force=True)

        # Release references
        self.llama_model = None
        self.decoder_model = None
        self.decode_one_token = None
        self.llama_queue = None

        self._models_loaded = False
        self._warmed_up = False

        logger.info("Model manager shutdown complete")

    def __enter__(self) -> "ModelManager":
        """Context manager entry."""
        self.load_models()
        self.warmup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
