"""
S1-Mini Configuration Management
================================

This module provides configuration classes for the S1-Mini inference engine.
Configuration can be set via:
1. Direct Python initialization
2. Environment variables (prefixed with S1_MINI_)
3. Configuration files (YAML/JSON)

Environment Variable Mapping:
-----------------------------
    S1_MINI_DEVICE          -> device
    S1_MINI_PRECISION       -> precision
    S1_MINI_COMPILE         -> compile_model
    S1_MINI_COMPILE_BACKEND -> compile_backend
    S1_MINI_BATCH_SIZE      -> max_batch_size
    S1_MINI_BATCH_TIMEOUT   -> batch_timeout_ms
    S1_MINI_REQUEST_TIMEOUT -> request_timeout_seconds
    S1_MINI_VRAM_THRESHOLD  -> vram_clear_threshold_gb

Usage:
------
    # Direct initialization
    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
    )

    # From environment
    config = EngineConfig.from_env()

    # With validation
    config = EngineConfig(device="invalid")  # Raises ConfigurationError
"""

import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, Union

import torch

from s1_mini.exceptions import ConfigurationError


# =============================================================================
# Enums for Type Safety
# =============================================================================


class DeviceType(str, Enum):
    """Supported device types."""

    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon

    @classmethod
    def from_string(cls, value: str) -> "DeviceType":
        """Parse device string, handling cuda:0 format."""
        if value.startswith("cuda"):
            return cls.CUDA
        return cls(value.lower())


class PrecisionType(str, Enum):
    """Supported precision types."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_torch_dtype(self) -> torch.dtype:
        """Convert to PyTorch dtype."""
        mapping = {
            self.FLOAT32: torch.float32,
            self.FLOAT16: torch.float16,
            self.BFLOAT16: torch.bfloat16,
        }
        return mapping[self]


class CompileBackend(str, Enum):
    """
    torch.compile backend options.

    AUTO: Automatically select based on platform
        - Linux with CUDA: inductor (uses Triton)
        - Windows: eager (no Triton)
        - CPU: eager

    INDUCTOR: Use inductor backend (requires Triton on Linux)
    EAGER: Use eager backend (works everywhere, slower)
    DISABLED: Don't use torch.compile
    """

    AUTO = "auto"
    INDUCTOR = "inductor"
    EAGER = "eager"
    AOT_EAGER = "aot_eager"
    DISABLED = "disabled"


# =============================================================================
# Helper Functions
# =============================================================================


def get_env(key: str, default: str = None) -> Optional[str]:
    """
    Get environment variable with S1_MINI_ prefix.

    Args:
        key: Variable name (without prefix)
        default: Default value if not set

    Returns:
        Environment variable value or default
    """
    return os.environ.get(f"S1_MINI_{key.upper()}", default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = None) -> Optional[int]:
    """Get integer environment variable."""
    value = get_env(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid integer value for S1_MINI_{key}: {value}",
            parameter=key,
            value=value,
        )


def get_env_float(key: str, default: float = None) -> Optional[float]:
    """Get float environment variable."""
    value = get_env(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid float value for S1_MINI_{key}: {value}",
            parameter=key,
            value=value,
        )


def detect_optimal_backend() -> CompileBackend:
    """
    Detect the optimal torch.compile backend for current platform.

    Returns:
        CompileBackend.INDUCTOR for Linux with CUDA (Triton support)
        CompileBackend.EAGER for Windows/macOS (no Triton)
        CompileBackend.DISABLED for CPU-only systems
    """
    system = platform.system()
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        # CPU-only: compilation provides minimal benefit
        return CompileBackend.DISABLED

    if system == "Linux":
        # Linux with CUDA: Triton is available
        return CompileBackend.INDUCTOR

    if system == "Windows":
        # Windows: Triton not supported, use eager fallback
        # This still provides some benefit from graph capture
        return CompileBackend.EAGER

    if system == "Darwin":
        # macOS: MPS backend, eager is safest
        return CompileBackend.EAGER

    # Unknown system, disable compilation
    return CompileBackend.DISABLED


def detect_optimal_device() -> str:
    """
    Detect the optimal device for inference.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_optimal_precision(device: str) -> PrecisionType:
    """
    Detect optimal precision for the given device.

    Args:
        device: Target device

    Returns:
        float16 for CUDA (most efficient)
        float32 for CPU (no fp16 acceleration)
        float32 for MPS (fp16 support is limited)
    """
    if device.startswith("cuda"):
        # Check if bfloat16 is well supported (Ampere+)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere or newer
                return PrecisionType.BFLOAT16
        return PrecisionType.FLOAT16

    # CPU and MPS: use float32
    return PrecisionType.FLOAT32


# =============================================================================
# Engine Configuration
# =============================================================================


@dataclass
class EngineConfig:
    """
    Configuration for the S1-Mini inference engine.

    This dataclass holds all configuration options for model loading,
    inference, and optimization. Values can be set directly or loaded
    from environment variables.

    Attributes:
        checkpoint_path: Path to model checkpoint directory
        device: Device to run inference on ('cuda', 'cpu', 'mps')
        precision: Model precision ('float32', 'float16', 'bfloat16')
        compile_model: Whether to use torch.compile for optimization
        compile_backend: Backend for torch.compile ('auto', 'inductor', 'eager')
        compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

    Batching Options:
        max_batch_size: Maximum requests per batch (default: 1)
        batch_timeout_ms: How long to wait for batch to fill (default: 50ms)

    VRAM Management:
        vram_clear_on_oom_only: Only clear VRAM cache on OOM errors
        vram_clear_threshold_gb: Clear cache when free VRAM below this

    Timeouts:
        request_timeout_seconds: Maximum time per request (default: 60s)
        warmup_timeout_seconds: Maximum time for warmup (default: 120s)

    Cache Settings:
        kv_cache_max_entries: Maximum KV cache entries for prefix caching
        reference_cache_max_entries: Maximum cached reference audios

    Example:
        >>> config = EngineConfig(
        ...     checkpoint_path="checkpoints/openaudio-s1-mini",
        ...     device="cuda",
        ...     precision="float16",
        ...     compile_model=True,
        ...     max_batch_size=4,
        ... )
    """

    # ==========================================================================
    # Model Configuration
    # ==========================================================================

    checkpoint_path: Union[str, Path] = field(
        default="checkpoints/openaudio-s1-mini",
        metadata={"help": "Path to model checkpoint directory"},
    )

    device: str = field(
        default_factory=detect_optimal_device,
        metadata={"help": "Device to run inference on (cuda, cpu, mps)"},
    )

    precision: str = field(
        default="float16",
        metadata={"help": "Model precision (float32, float16, bfloat16)"},
    )

    # ==========================================================================
    # Compilation Configuration
    # ==========================================================================

    compile_model: bool = field(
        default=True,
        metadata={"help": "Whether to use torch.compile for optimization"},
    )

    compile_backend: str = field(
        default="auto",
        metadata={
            "help": "Backend for torch.compile (auto, inductor, eager, disabled)"
        },
    )

    compile_mode: str = field(
        default="reduce-overhead",
        metadata={
            "help": "Compilation mode (default, reduce-overhead, max-autotune)"
        },
    )

    # ==========================================================================
    # Batching Configuration
    # ==========================================================================

    max_batch_size: int = field(
        default=1,
        metadata={"help": "Maximum number of requests per batch"},
    )

    batch_timeout_ms: int = field(
        default=50,
        metadata={"help": "Milliseconds to wait for batch to fill"},
    )

    # ==========================================================================
    # VRAM Management
    # ==========================================================================

    vram_clear_on_oom_only: bool = field(
        default=True,
        metadata={
            "help": "Only clear VRAM cache on OOM errors (recommended for production)"
        },
    )

    vram_clear_threshold_gb: float = field(
        default=2.0,
        metadata={"help": "Clear cache when free VRAM drops below this (GB)"},
    )

    vram_reserved_gb: float = field(
        default=1.0,
        metadata={"help": "Reserved VRAM for system overhead (GB)"},
    )

    # ==========================================================================
    # Timeout Configuration
    # ==========================================================================

    request_timeout_seconds: float = field(
        default=60.0,
        metadata={"help": "Maximum time for a single request (seconds)"},
    )

    warmup_timeout_seconds: float = field(
        default=120.0,
        metadata={"help": "Maximum time for model warmup (seconds)"},
    )

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================

    kv_cache_max_entries: int = field(
        default=100,
        metadata={"help": "Maximum KV cache entries for prefix caching"},
    )

    reference_cache_max_entries: int = field(
        default=1000,
        metadata={"help": "Maximum number of cached reference audios"},
    )

    reference_cache_memory_mb: int = field(
        default=2048,
        metadata={"help": "Maximum memory for reference cache (MB)"},
    )

    # ==========================================================================
    # Generation Defaults
    # ==========================================================================

    default_max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Default maximum new tokens to generate"},
    )

    default_temperature: float = field(
        default=0.7,
        metadata={"help": "Default sampling temperature"},
    )

    default_top_p: float = field(
        default=0.8,
        metadata={"help": "Default top-p (nucleus) sampling threshold"},
    )

    default_repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Default repetition penalty"},
    )

    default_chunk_length: int = field(
        default=200,
        metadata={"help": "Default chunk length for iterative generation"},
    )

    # ==========================================================================
    # Validation and Post-Init
    # ==========================================================================

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._resolve_auto_settings()

    def _validate(self):
        """Validate all configuration values."""
        # Validate device
        valid_devices = ["cuda", "cpu", "mps"]
        device_base = self.device.split(":")[0]
        if device_base not in valid_devices:
            raise ConfigurationError(
                f"Invalid device: {self.device}. Must be one of {valid_devices}",
                parameter="device",
                value=self.device,
            )

        # Validate precision
        valid_precisions = ["float32", "float16", "bfloat16"]
        if self.precision not in valid_precisions:
            raise ConfigurationError(
                f"Invalid precision: {self.precision}. Must be one of {valid_precisions}",
                parameter="precision",
                value=self.precision,
            )

        # Validate compile backend
        valid_backends = ["auto", "inductor", "eager", "aot_eager", "disabled"]
        if self.compile_backend not in valid_backends:
            raise ConfigurationError(
                f"Invalid compile_backend: {self.compile_backend}. "
                f"Must be one of {valid_backends}",
                parameter="compile_backend",
                value=self.compile_backend,
            )

        # Validate numeric ranges
        if self.max_batch_size < 1:
            raise ConfigurationError(
                "max_batch_size must be >= 1",
                parameter="max_batch_size",
                value=self.max_batch_size,
            )

        if self.request_timeout_seconds <= 0:
            raise ConfigurationError(
                "request_timeout_seconds must be > 0",
                parameter="request_timeout_seconds",
                value=self.request_timeout_seconds,
            )

    def _resolve_auto_settings(self):
        """Resolve 'auto' settings based on platform detection."""
        # Resolve compile backend
        if self.compile_backend == "auto":
            detected = detect_optimal_backend()
            self._resolved_compile_backend = detected.value
        else:
            self._resolved_compile_backend = self.compile_backend

        # Resolve precision if device changed
        # (No auto-precision currently, but could be added)

    @property
    def resolved_compile_backend(self) -> str:
        """Get the resolved compile backend (after auto-detection)."""
        return getattr(self, "_resolved_compile_backend", self.compile_backend)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype from precision string."""
        return PrecisionType(self.precision).to_torch_dtype()

    @property
    def should_compile(self) -> bool:
        """Whether to actually compile the model."""
        return self.compile_model and self.resolved_compile_backend != "disabled"

    # ==========================================================================
    # Factory Methods
    # ==========================================================================

    @classmethod
    def from_env(cls, **overrides) -> "EngineConfig":
        """
        Create configuration from environment variables.

        Environment variables are prefixed with S1_MINI_.
        Any keyword arguments override environment values.

        Example:
            >>> os.environ["S1_MINI_DEVICE"] = "cuda:1"
            >>> config = EngineConfig.from_env(precision="float32")
        """
        env_values = {
            "checkpoint_path": get_env("CHECKPOINT_PATH"),
            "device": get_env("DEVICE"),
            "precision": get_env("PRECISION"),
            "compile_model": get_env_bool("COMPILE", default=True),
            "compile_backend": get_env("COMPILE_BACKEND", default="auto"),
            "max_batch_size": get_env_int("BATCH_SIZE", default=1),
            "batch_timeout_ms": get_env_int("BATCH_TIMEOUT", default=50),
            "request_timeout_seconds": get_env_float("REQUEST_TIMEOUT", default=60.0),
            "vram_clear_on_oom_only": get_env_bool("VRAM_CLEAR_ON_OOM_ONLY", default=True),
            "vram_clear_threshold_gb": get_env_float("VRAM_THRESHOLD", default=2.0),
        }

        # Remove None values (use defaults)
        env_values = {k: v for k, v in env_values.items() if v is not None}

        # Apply overrides
        env_values.update(overrides)

        return cls(**env_values)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "precision": self.precision,
            "compile_model": self.compile_model,
            "compile_backend": self.compile_backend,
            "resolved_compile_backend": self.resolved_compile_backend,
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "request_timeout_seconds": self.request_timeout_seconds,
            "vram_clear_on_oom_only": self.vram_clear_on_oom_only,
        }


# =============================================================================
# Server Configuration
# =============================================================================


@dataclass
class ServerConfig:
    """
    Configuration for the production HTTP server.

    Attributes:
        host: Server bind address
        port: Server port
        workers: Number of worker processes (NOTE: Must be 1 for GPU)
        log_level: Logging level
        cors_origins: Allowed CORS origins
        api_key: Optional API key for authentication
    """

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # Must be 1 for GPU memory sharing
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Request limits
    max_request_size_mb: int = 50
    max_text_length: int = 10000
    max_concurrent_requests: int = 100

    # Timeouts
    keepalive_timeout: int = 65
    graceful_shutdown_timeout: int = 30

    def __post_init__(self):
        """Validate server configuration."""
        if self.workers > 1:
            import warnings

            warnings.warn(
                "Using workers > 1 with GPU inference can cause VRAM conflicts. "
                "Each worker loads its own model copy. "
                "Set workers=1 unless you have multiple GPUs.",
                RuntimeWarning,
            )

    @classmethod
    def from_env(cls, **overrides) -> "ServerConfig":
        """Create server configuration from environment variables."""
        env_values = {
            "host": get_env("SERVER_HOST", "0.0.0.0"),
            "port": get_env_int("SERVER_PORT", 8080),
            "workers": get_env_int("SERVER_WORKERS", 1),
            "log_level": get_env("LOG_LEVEL", "info"),
            "api_key": get_env("API_KEY"),
            "enable_metrics": get_env_bool("METRICS_ENABLED", True),
        }

        env_values.update(overrides)
        return cls(**env_values)
