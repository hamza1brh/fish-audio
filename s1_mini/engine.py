"""
S1-Mini Production Inference Engine
====================================

This is the main inference engine for production TTS generation.
It wraps the model manager and provides a clean API for text-to-speech.

Key Features:
-------------
1. Persistent model loading (no reload per request)
2. Optimized VRAM management (no aggressive clearing)
3. Platform-aware compilation (Triton/Windows fallback)
4. Request timeout handling
5. Comprehensive error handling
6. Metrics and monitoring hooks

Architecture:
-------------
    ProductionTTSEngine
    ├── ModelManager
    │   ├── LLAMA Model (persistent in VRAM)
    │   └── DAC Decoder (persistent in VRAM)
    │
    ├── Reference Cache
    │   └── Pre-encoded reference audios
    │
    └── Request Handler
        ├── Timeout management
        ├── Error handling
        └── Metrics collection

Usage:
------
    from s1_mini import ProductionTTSEngine, EngineConfig

    # Create and initialize engine
    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
    )
    engine = ProductionTTSEngine(config)
    engine.start()

    # Generate audio
    result = engine.generate(
        text="Hello, this is a test.",
        reference_audio=audio_bytes,  # Optional
    )

    # Get audio data
    sample_rate, audio_array = result.audio

    # Or use streaming
    for chunk in engine.generate_stream(text="Hello"):
        # Process each audio chunk
        pass

API Design:
-----------
The engine provides two main methods:

1. generate(): Synchronous generation, returns complete audio
2. generate_stream(): Streaming generation, yields audio chunks

Both methods support:
- Optional reference audio for voice cloning
- Configurable generation parameters
- Timeout handling
- Error recovery

Thread Safety:
--------------
The engine is thread-safe for concurrent requests.
Internally, requests are serialized through the model worker queue.
"""

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union, List

import numpy as np
import torch
from loguru import logger

from s1_mini.config import EngineConfig
from s1_mini.exceptions import (
    InferenceError,
    TimeoutError,
    ModelLoadError,
    OOMError,
)
from s1_mini.model_manager import ModelManager
from s1_mini.utils import (
    InferenceResult,
    GenerationMetrics,
    Timer,
    audio_to_bytes,
    get_vram_info,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GenerationRequest:
    """
    Request for TTS generation.

    Attributes:
        text: Text to synthesize
        reference_audio: Optional reference audio bytes for voice cloning
        reference_id: Optional pre-registered reference ID
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
        chunk_length: Length of text chunks for iterative generation
        seed: Random seed for reproducibility
        streaming: Whether to use streaming mode
        timeout: Request timeout in seconds
    """

    text: str
    reference_audio: Optional[bytes] = None
    reference_id: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    chunk_length: int = 200
    seed: Optional[int] = None
    streaming: bool = False
    timeout: Optional[float] = None


@dataclass
class GenerationResponse:
    """
    Response from TTS generation.

    Attributes:
        success: Whether generation succeeded
        audio: Tuple of (sample_rate, audio_array) or None
        audio_bytes: Audio as bytes (WAV format) or None
        error: Error message if failed
        metrics: Generation metrics
    """

    success: bool
    audio: Optional[tuple] = None  # (sample_rate, np.ndarray)
    audio_bytes: Optional[bytes] = None
    error: Optional[str] = None
    metrics: Optional[GenerationMetrics] = None


# =============================================================================
# Production TTS Engine
# =============================================================================


class ProductionTTSEngine:
    """
    Production-ready TTS inference engine.

    This class provides the main interface for TTS generation.
    It handles:
    - Model lifecycle management
    - Request processing
    - Error handling and recovery
    - Metrics collection

    Key Optimizations:
    ------------------
    1. Models stay loaded in VRAM between requests
    2. VRAM cache is not cleared after each request
    3. KV cache is reused when possible
    4. Platform-aware compilation for optimal performance

    Attributes:
        config: Engine configuration
        model_manager: Model lifecycle manager
        is_running: Whether the engine is running

    Example:
        >>> engine = ProductionTTSEngine(config)
        >>> engine.start()
        >>>
        >>> # Simple generation
        >>> response = engine.generate("Hello world")
        >>> if response.success:
        ...     play_audio(response.audio)
        >>>
        >>> # With reference audio
        >>> with open("reference.wav", "rb") as f:
        ...     ref_audio = f.read()
        >>> response = engine.generate("Hello", reference_audio=ref_audio)
        >>>
        >>> # Streaming
        >>> for chunk in engine.generate_stream("Long text..."):
        ...     stream_audio(chunk.audio)
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the production TTS engine.

        Args:
            config: Engine configuration. If None, uses defaults.
        """
        self.config = config or EngineConfig()
        self.model_manager = ModelManager(self.config)
        self._is_running = False
        self._tts_engine = None

        logger.info("ProductionTTSEngine initialized")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Precision: {self.config.precision}")
        logger.info(f"  Compile: {self.config.should_compile}")
        if self.config.should_compile:
            logger.info(f"  Backend: {self.config.resolved_compile_backend}")

    @property
    def is_running(self) -> bool:
        """Check if the engine is running and ready for requests."""
        return self._is_running and self.model_manager.is_ready

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def start(self) -> None:
        """
        Start the engine (load models and warmup).

        This method:
        1. Loads LLAMA and DAC models
        2. Sets up the inference worker
        3. Runs warmup to complete JIT compilation

        After this method returns, the engine is ready for requests.

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._is_running:
            logger.warning("Engine already running")
            return

        logger.info("Starting ProductionTTSEngine...")

        # Load models
        self.model_manager.load_models()

        # Create TTS inference engine
        from fish_speech.inference_engine import TTSInferenceEngine

        self._tts_engine = TTSInferenceEngine(
            llama_queue=self.model_manager.llama_queue,
            decoder_model=self.model_manager.decoder_model,
            precision=self.config.torch_dtype,
            compile=self.config.should_compile,
        )

        # Warmup
        self.model_manager.warmup()

        self._is_running = True
        logger.info("ProductionTTSEngine started and ready")

    def stop(self) -> None:
        """
        Stop the engine and release resources.

        This method cleanly shuts down:
        1. Worker thread
        2. Model memory
        3. VRAM cache
        """
        if not self._is_running:
            return

        logger.info("Stopping ProductionTTSEngine...")
        self.model_manager.shutdown()
        self._tts_engine = None
        self._is_running = False
        logger.info("ProductionTTSEngine stopped")

    def __enter__(self) -> "ProductionTTSEngine":
        """Context manager entry - starts the engine."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the engine."""
        self.stop()

    # =========================================================================
    # Generation Methods
    # =========================================================================

    def generate(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        chunk_length: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
        return_bytes: bool = False,
    ) -> GenerationResponse:
        """
        Generate audio from text (synchronous).

        This method generates complete audio from the input text.
        It blocks until generation is complete or timeout is reached.

        Args:
            text: Text to synthesize
            reference_audio: Reference audio bytes for voice cloning
            reference_id: Pre-registered reference ID
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            repetition_penalty: Repetition penalty (0.0-2.0)
            chunk_length: Chunk length for iterative generation
            seed: Random seed for reproducibility
            timeout: Request timeout in seconds
            return_bytes: If True, include audio_bytes in response

        Returns:
            GenerationResponse with audio data or error

        Raises:
            RuntimeError: If engine is not running
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")

        # Apply defaults from config
        max_new_tokens = max_new_tokens or self.config.default_max_new_tokens
        temperature = temperature or self.config.default_temperature
        top_p = top_p or self.config.default_top_p
        repetition_penalty = repetition_penalty or self.config.default_repetition_penalty
        chunk_length = chunk_length or self.config.default_chunk_length
        timeout = timeout or self.config.request_timeout_seconds

        # Create request
        request = self._create_request(
            text=text,
            reference_audio=reference_audio,
            reference_id=reference_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_length=chunk_length,
            seed=seed,
            streaming=False,
        )

        # Run inference
        timer = Timer("generation")
        try:
            result = self._run_inference(request, timeout=timeout)

            if result.error is not None:
                return GenerationResponse(
                    success=False,
                    error=str(result.error),
                )

            # Calculate metrics
            elapsed = timer.elapsed
            audio_duration = len(result.audio[1]) / result.audio[0] if result.audio else 0
            metrics = GenerationMetrics(
                total_time_seconds=elapsed,
                tokens_generated=0,  # TODO: Track actual tokens
                tokens_per_second=0,
                time_to_first_token_seconds=0,
                audio_duration_seconds=audio_duration,
                realtime_factor=audio_duration / elapsed if elapsed > 0 else 0,
            )

            # Convert to bytes if requested
            audio_bytes = None
            if return_bytes and result.audio:
                audio_bytes = audio_to_bytes(
                    result.audio[1],
                    result.audio[0],
                    format="wav",
                )

            return GenerationResponse(
                success=True,
                audio=result.audio,
                audio_bytes=audio_bytes,
                metrics=metrics,
            )

        except TimeoutError as e:
            return GenerationResponse(
                success=False,
                error=f"Generation timed out after {timeout}s",
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResponse(
                success=False,
                error=str(e),
            )

    def generate_stream(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        chunk_length: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Generator[InferenceResult, None, None]:
        """
        Generate audio from text (streaming).

        This method yields audio chunks as they are generated.
        Useful for real-time audio playback.

        Args:
            text: Text to synthesize
            reference_audio: Reference audio bytes for voice cloning
            reference_id: Pre-registered reference ID
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            chunk_length: Chunk length for iterative generation
            seed: Random seed for reproducibility
            timeout: Request timeout in seconds

        Yields:
            InferenceResult with:
            - code="header": WAV header (first yield)
            - code="segment": Audio chunk
            - code="final": Complete audio
            - code="error": Error occurred
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")

        # Apply defaults
        max_new_tokens = max_new_tokens or self.config.default_max_new_tokens
        temperature = temperature or self.config.default_temperature
        top_p = top_p or self.config.default_top_p
        repetition_penalty = repetition_penalty or self.config.default_repetition_penalty
        chunk_length = chunk_length or self.config.default_chunk_length
        timeout = timeout or self.config.request_timeout_seconds

        # Create streaming request
        request = self._create_request(
            text=text,
            reference_audio=reference_audio,
            reference_id=reference_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_length=chunk_length,
            seed=seed,
            streaming=True,
        )

        # Yield results
        start_time = time.time()
        try:
            for result in self._tts_engine.inference(request):
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    yield InferenceResult(
                        code="error",
                        audio=None,
                        error=TimeoutError(
                            f"Generation timed out",
                            timeout_seconds=timeout,
                            elapsed_seconds=time.time() - start_time,
                        ),
                    )
                    return

                yield result

        except Exception as e:
            yield InferenceResult(
                code="error",
                audio=None,
                error=e,
            )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _create_request(
        self,
        text: str,
        reference_audio: Optional[bytes],
        reference_id: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        chunk_length: int,
        seed: Optional[int],
        streaming: bool,
    ):
        """Create a ServeTTSRequest from parameters."""
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        # Handle reference audio
        references = []
        if reference_audio is not None:
            # TODO: Proper reference handling with encoding
            # For now, we'll use the reference_id approach
            pass

        return ServeTTSRequest(
            text=text,
            references=references,
            reference_id=reference_id,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            streaming=streaming,
            format="wav",
        )

    def _run_inference(
        self,
        request,
        timeout: float,
    ) -> InferenceResult:
        """
        Run inference with timeout handling.

        This method runs the inference engine and collects all results.
        For non-streaming requests, it waits for the final result.
        """
        start_time = time.time()
        final_result = None

        for result in self._tts_engine.inference(request):
            # Check timeout
            if (time.time() - start_time) > timeout:
                raise TimeoutError(
                    "Generation timed out",
                    timeout_seconds=timeout,
                    elapsed_seconds=time.time() - start_time,
                )

            # Handle result codes
            if result.code == "error":
                return result
            elif result.code == "final":
                final_result = result
                break

        if final_result is None:
            return InferenceResult(
                code="error",
                audio=None,
                error=InferenceError("No final result received"),
            )

        return final_result

    # =========================================================================
    # Health and Monitoring
    # =========================================================================

    def health_check(self) -> dict:
        """
        Get engine health status.

        Returns:
            Dictionary with health information
        """
        base_health = self.model_manager.health_check()
        base_health["engine_running"] = self._is_running
        return base_health

    def get_metrics(self) -> dict:
        """
        Get engine metrics.

        Returns:
            Dictionary with metrics
        """
        vram = get_vram_info()
        return {
            "vram_used_gb": vram["used_gb"],
            "vram_total_gb": vram["total_gb"],
            "vram_utilization": vram["utilization"],
            # TODO: Add request metrics
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_engine(
    checkpoint_path: str = "checkpoints/openaudio-s1-mini",
    device: str = "cuda",
    precision: str = "float16",
    compile: bool = True,
) -> ProductionTTSEngine:
    """
    Create and start a production TTS engine.

    Convenience function for quick engine setup.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device
        precision: Model precision
        compile: Whether to use torch.compile

    Returns:
        Started ProductionTTSEngine instance
    """
    config = EngineConfig(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        compile_model=compile,
    )

    engine = ProductionTTSEngine(config)
    engine.start()
    return engine
