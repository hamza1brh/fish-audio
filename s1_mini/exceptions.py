"""
S1-Mini Custom Exceptions
=========================

This module defines custom exceptions for the S1-Mini inference engine.
These exceptions provide clear error categorization for:
- Model loading failures
- Inference errors
- Timeout handling
- VRAM management issues

Exception Hierarchy:
--------------------
    S1MiniError (base)
    ├── ModelLoadError      - Failed to load model weights
    ├── InferenceError      - Error during inference
    │   ├── TokenizationError   - Text tokenization failed
    │   └── GenerationError     - Token generation failed
    ├── TimeoutError        - Request exceeded timeout
    ├── VRAMError           - GPU memory issues
    │   ├── OOMError            - Out of memory
    │   └── AllocationError     - Failed to allocate tensors
    └── ConfigurationError  - Invalid configuration

Usage:
------
    from s1_mini.exceptions import InferenceError, TimeoutError

    try:
        result = engine.generate(text)
    except TimeoutError as e:
        logger.warning(f"Request timed out after {e.timeout_seconds}s")
    except InferenceError as e:
        logger.error(f"Inference failed: {e}")
"""


class S1MiniError(Exception):
    """
    Base exception for all S1-Mini errors.

    All custom exceptions in this module inherit from this class,
    making it easy to catch any S1-Mini related error:

        try:
            result = engine.generate(text)
        except S1MiniError as e:
            # Handle any S1-Mini error
            logger.error(f"S1-Mini error: {e}")
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Model Loading Errors
# =============================================================================


class ModelLoadError(S1MiniError):
    """
    Raised when model loading fails.

    Common causes:
    - Checkpoint file not found
    - Corrupted checkpoint
    - Incompatible model architecture
    - Insufficient VRAM for model weights

    Example:
        try:
            engine.load_model(checkpoint_path)
        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e}")
            logger.info(f"Checkpoint path: {e.details.get('checkpoint_path')}")
    """

    def __init__(self, message: str, checkpoint_path: str = None, **kwargs):
        details = {"checkpoint_path": checkpoint_path, **kwargs}
        super().__init__(message, details)
        self.checkpoint_path = checkpoint_path


# =============================================================================
# Inference Errors
# =============================================================================


class InferenceError(S1MiniError):
    """
    Base class for inference-related errors.

    Raised when the inference pipeline fails after model is loaded.
    """

    def __init__(self, message: str, stage: str = None, **kwargs):
        """
        Args:
            message: Error description
            stage: Pipeline stage where error occurred
                   (e.g., 'tokenization', 'generation', 'decoding')
        """
        details = {"stage": stage, **kwargs}
        super().__init__(message, details)
        self.stage = stage


class TokenizationError(InferenceError):
    """
    Raised when text tokenization fails.

    Common causes:
    - Empty or invalid text input
    - Unsupported characters
    - Text exceeds maximum length

    Example:
        try:
            tokens = engine.tokenize(text)
        except TokenizationError as e:
            logger.error(f"Tokenization failed: {e}")
    """

    def __init__(self, message: str, text: str = None, **kwargs):
        super().__init__(message, stage="tokenization", text=text[:100] if text else None, **kwargs)


class GenerationError(InferenceError):
    """
    Raised when token generation fails.

    Common causes:
    - Invalid prompt tokens
    - Generation diverged (repeated tokens)
    - Internal model error

    Example:
        try:
            tokens = engine.generate_tokens(prompt)
        except GenerationError as e:
            logger.error(f"Generation failed at step {e.details.get('step')}")
    """

    def __init__(self, message: str, step: int = None, **kwargs):
        super().__init__(message, stage="generation", step=step, **kwargs)


class DecodingError(InferenceError):
    """
    Raised when audio decoding fails.

    Common causes:
    - Invalid VQ tokens
    - DAC decoder error
    - Audio reconstruction failure
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(message, stage="decoding", **kwargs)


# =============================================================================
# Timeout Errors
# =============================================================================


class TimeoutError(S1MiniError):
    """
    Raised when a request exceeds the configured timeout.

    This is a soft timeout - the request is cancelled but the
    engine remains operational.

    Attributes:
        timeout_seconds: The timeout value that was exceeded
        elapsed_seconds: How long the request ran before timeout

    Example:
        try:
            result = engine.generate(text, timeout=30)
        except TimeoutError as e:
            logger.warning(
                f"Request timed out after {e.elapsed_seconds:.1f}s "
                f"(limit: {e.timeout_seconds}s)"
            )
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float = None,
        elapsed_seconds: float = None,
        **kwargs,
    ):
        details = {
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds,
            **kwargs,
        }
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


# =============================================================================
# VRAM Errors
# =============================================================================


class VRAMError(S1MiniError):
    """
    Base class for GPU memory related errors.

    Provides context about VRAM state when error occurred.
    """

    def __init__(
        self,
        message: str,
        vram_used_gb: float = None,
        vram_total_gb: float = None,
        **kwargs,
    ):
        details = {
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            **kwargs,
        }
        super().__init__(message, details)
        self.vram_used_gb = vram_used_gb
        self.vram_total_gb = vram_total_gb


class OOMError(VRAMError):
    """
    Raised when GPU runs out of memory.

    This triggers automatic VRAM cleanup. If cleanup succeeds,
    the request may be retried.

    Example:
        try:
            result = engine.generate(long_text)
        except OOMError as e:
            logger.error(
                f"OOM: Using {e.vram_used_gb:.1f}GB of {e.vram_total_gb:.1f}GB"
            )
            # Engine will automatically clean up, retry may succeed
    """

    def __init__(self, message: str = "CUDA out of memory", **kwargs):
        super().__init__(message, **kwargs)


class AllocationError(VRAMError):
    """
    Raised when tensor allocation fails.

    Different from OOM - this may indicate fragmentation
    rather than total memory exhaustion.
    """

    def __init__(self, message: str, tensor_shape: tuple = None, **kwargs):
        super().__init__(message, tensor_shape=tensor_shape, **kwargs)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(S1MiniError):
    """
    Raised when engine configuration is invalid.

    Common causes:
    - Invalid device specification
    - Unsupported precision
    - Conflicting options

    Example:
        try:
            config = EngineConfig(device="tpu")  # Invalid
        except ConfigurationError as e:
            logger.error(f"Invalid config: {e}")
    """

    def __init__(self, message: str, parameter: str = None, value=None, **kwargs):
        details = {"parameter": parameter, "value": value, **kwargs}
        super().__init__(message, details)
        self.parameter = parameter
        self.value = value


# =============================================================================
# Request Queue Errors
# =============================================================================


class QueueFullError(S1MiniError):
    """
    Raised when the request queue is full.

    This indicates the server is overloaded and cannot accept
    more requests. Clients should implement backoff/retry.
    """

    def __init__(
        self,
        message: str = "Request queue is full",
        queue_size: int = None,
        **kwargs,
    ):
        details = {"queue_size": queue_size, **kwargs}
        super().__init__(message, details)
        self.queue_size = queue_size


class RequestCancelledError(S1MiniError):
    """
    Raised when a request is cancelled before completion.

    This can happen due to:
    - Client disconnect
    - Manual cancellation
    - Server shutdown
    """

    def __init__(self, message: str = "Request was cancelled", **kwargs):
        super().__init__(message, **kwargs)
