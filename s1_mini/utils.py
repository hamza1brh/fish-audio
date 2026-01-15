"""
S1-Mini Utility Functions
=========================

Common utility functions used throughout the S1-Mini inference engine.

Contents:
---------
- Audio processing utilities
- Tensor utilities
- Timing and profiling helpers
- Memory management helpers
- Type definitions and protocols
"""

import gc
import io
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, Union, BinaryIO

import numpy as np
import torch
from loguru import logger


# =============================================================================
# Type Definitions
# =============================================================================


@dataclass
class InferenceResult:
    """
    Result of a single inference step or complete generation.

    Attributes:
        code: Result type
            - "header": WAV header for streaming
            - "segment": Audio segment (streaming mode)
            - "final": Complete audio (non-streaming)
            - "error": Error occurred
        audio: Tuple of (sample_rate, audio_array) or None for errors
        error: Exception if code is "error", None otherwise
        metadata: Optional metadata (timing, tokens generated, etc.)
    """

    code: str  # "header", "segment", "final", "error"
    audio: Optional[Tuple[int, np.ndarray]]
    error: Optional[Exception]
    metadata: Optional[dict] = None


@dataclass
class GenerationMetrics:
    """
    Metrics collected during generation.

    Useful for monitoring and debugging performance.
    """

    total_time_seconds: float
    tokens_generated: int
    tokens_per_second: float
    time_to_first_token_seconds: float
    audio_duration_seconds: float
    realtime_factor: float  # audio_duration / generation_time


# =============================================================================
# Audio Utilities
# =============================================================================


def wav_chunk_header(
    sample_rate: int = 44100,
    bits_per_sample: int = 16,
    channels: int = 1,
) -> bytes:
    """
    Generate a WAV header for streaming audio.

    This creates a WAV header with unknown file size (set to max uint32).
    Useful for streaming responses where total length is unknown.

    Args:
        sample_rate: Audio sample rate (default: 44100 Hz)
        bits_per_sample: Bits per sample (default: 16)
        channels: Number of channels (default: 1 mono)

    Returns:
        44-byte WAV header as bytes

    Note:
        The file size and data chunk size are set to 0xFFFFFFFF
        to indicate streaming/unknown length.
    """
    # WAV file format specification:
    # Offset  Size  Description
    # 0       4     "RIFF" chunk ID
    # 4       4     File size - 8 (unknown for streaming, use max)
    # 8       4     "WAVE" format
    # 12      4     "fmt " subchunk ID
    # 16      4     Subchunk size (16 for PCM)
    # 20      2     Audio format (1 = PCM)
    # 22      2     Number of channels
    # 24      4     Sample rate
    # 28      4     Byte rate
    # 32      2     Block align
    # 34      2     Bits per sample
    # 36      4     "data" subchunk ID
    # 40      4     Data size (unknown for streaming)

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,  # File size (unknown)
        b"WAVE",
        b"fmt ",
        16,  # Subchunk size
        1,  # Audio format (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,  # Data size (unknown)
    )

    return header


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    format: str = "wav",
) -> bytes:
    """
    Convert audio numpy array to bytes.

    Args:
        audio: Audio data as numpy array (float32, range [-1, 1])
        sample_rate: Sample rate in Hz
        format: Output format ("wav" or "raw")

    Returns:
        Audio data as bytes
    """
    # Normalize to int16 range
    audio_int16 = (audio * 32767).astype(np.int16)

    if format == "raw":
        return audio_int16.tobytes()

    # WAV format
    buffer = io.BytesIO()

    # Write WAV header
    header = wav_chunk_header(sample_rate=sample_rate)
    buffer.write(header)

    # Write audio data
    buffer.write(audio_int16.tobytes())

    # Update file size in header
    file_size = buffer.tell() - 8
    buffer.seek(4)
    buffer.write(struct.pack("<I", file_size))

    # Update data chunk size
    data_size = len(audio_int16) * 2  # 2 bytes per sample
    buffer.seek(40)
    buffer.write(struct.pack("<I", data_size))

    return buffer.getvalue()


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Uses linear interpolation for simplicity. For production,
    consider using librosa.resample or torchaudio.functional.resample.

    Args:
        audio: Input audio array
        original_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if original_sr == target_sr:
        return audio

    duration = len(audio) / original_sr
    new_length = int(duration * target_sr)

    # Simple linear interpolation
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


# =============================================================================
# Tensor Utilities
# =============================================================================


def move_to_device(
    tensor_or_dict: Union[torch.Tensor, dict, list],
    device: torch.device,
) -> Union[torch.Tensor, dict, list]:
    """
    Recursively move tensors to a device.

    Handles nested dictionaries and lists of tensors.

    Args:
        tensor_or_dict: Tensor, dict, or list to move
        device: Target device

    Returns:
        Structure with tensors on target device
    """
    if isinstance(tensor_or_dict, torch.Tensor):
        return tensor_or_dict.to(device)
    elif isinstance(tensor_or_dict, dict):
        return {k: move_to_device(v, device) for k, v in tensor_or_dict.items()}
    elif isinstance(tensor_or_dict, list):
        return [move_to_device(v, device) for v in tensor_or_dict]
    else:
        return tensor_or_dict


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """
    Get memory usage of a tensor in megabytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        Memory usage in MB
    """
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)


# =============================================================================
# VRAM Management
# =============================================================================


def get_vram_info() -> dict:
    """
    Get current VRAM usage information.

    Returns:
        Dictionary with VRAM statistics:
        - total_gb: Total VRAM
        - used_gb: Currently used VRAM
        - free_gb: Available VRAM
        - cached_gb: PyTorch cached memory
        - utilization: Usage percentage
    """
    if not torch.cuda.is_available():
        return {
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "cached_gb": 0,
            "utilization": 0,
        }

    # Get memory stats
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    cached = torch.cuda.memory_reserved(0)
    free = total - cached

    return {
        "total_gb": total / (1024**3),
        "used_gb": allocated / (1024**3),
        "free_gb": free / (1024**3),
        "cached_gb": cached / (1024**3),
        "utilization": (cached / total) * 100,
    }


def clear_vram_cache(force: bool = False) -> dict:
    """
    Clear PyTorch VRAM cache.

    This releases cached memory back to the GPU, which can help
    with fragmentation. However, frequent clearing can hurt
    performance due to reallocation overhead.

    Args:
        force: If True, also run Python garbage collection

    Returns:
        VRAM info after clearing
    """
    if not torch.cuda.is_available():
        return get_vram_info()

    # Record before state
    before = get_vram_info()

    # Clear cache
    torch.cuda.empty_cache()

    if force:
        gc.collect()
        torch.cuda.empty_cache()

    # Record after state
    after = get_vram_info()

    freed_gb = before["cached_gb"] - after["cached_gb"]
    if freed_gb > 0.01:  # Only log if meaningful
        logger.debug(f"Cleared {freed_gb:.2f} GB from VRAM cache")

    return after


@contextmanager
def vram_monitor(operation_name: str = "operation"):
    """
    Context manager to monitor VRAM usage during an operation.

    Usage:
        with vram_monitor("model_forward"):
            output = model(input)
        # Logs VRAM delta after operation

    Args:
        operation_name: Name for logging
    """
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(0)

    yield

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated(0)

    delta_mb = (after - before) / (1024 * 1024)
    if abs(delta_mb) > 10:  # Only log significant changes
        logger.debug(f"VRAM delta for {operation_name}: {delta_mb:+.1f} MB")


# =============================================================================
# Timing Utilities
# =============================================================================


class Timer:
    """
    Simple timer for measuring operation duration.

    Usage:
        timer = Timer()
        # ... do work ...
        print(f"Took {timer.elapsed:.2f}s")

        # Or as context manager:
        with Timer() as t:
            # ... do work ...
        print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self, name: str = None):
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time = None

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self.end_time = time.perf_counter()
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, *args):
        self.stop()
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.3f}s")


@contextmanager
def timed_operation(name: str) -> Generator[Timer, None, None]:
    """
    Context manager for timing operations with logging.

    Args:
        name: Operation name for logging

    Yields:
        Timer instance

    Usage:
        with timed_operation("inference") as timer:
            result = model(input)
        # Automatically logs: "inference: 1.234s"
    """
    timer = Timer(name)
    try:
        yield timer
    finally:
        timer.stop()
        logger.info(f"{name}: {timer.elapsed:.3f}s")


# =============================================================================
# Seed Management
# =============================================================================


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Sets seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# =============================================================================
# Safe Context Managers
# =============================================================================


@contextmanager
def inference_mode():
    """
    Context manager for inference mode with CUDA optimizations.

    Combines:
    - torch.inference_mode() - Disables autograd
    - torch.cuda.amp.autocast() - Mixed precision (if CUDA)
    - CUDA synchronization on exit
    """
    with torch.inference_mode():
        yield

    # Ensure all CUDA operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def autocast_context(device_type: str, dtype: torch.dtype):
    """
    Context manager for automatic mixed precision.

    Handles MPS devices which don't support autocast.

    Args:
        device_type: Device type ("cuda", "cpu", "mps")
        dtype: Target dtype for autocast
    """
    if device_type == "mps":
        # MPS doesn't support autocast well
        yield
    else:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield


# =============================================================================
# Batch Utilities
# =============================================================================


def pad_sequences(
    sequences: list[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length sequences to the same length.

    Args:
        sequences: List of 1D tensors
        padding_value: Value to use for padding
        max_length: Maximum length (if None, uses longest sequence)

    Returns:
        Tuple of:
        - Padded tensor of shape (batch, max_length)
        - Lengths tensor of shape (batch,)
    """
    lengths = torch.tensor([len(seq) for seq in sequences])

    if max_length is None:
        max_length = lengths.max().item()

    batch_size = len(sequences)
    padded = torch.full(
        (batch_size, max_length),
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )

    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    return padded, lengths


def create_attention_mask(
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Create attention mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths (batch,)
        max_length: Maximum sequence length

    Returns:
        Attention mask of shape (batch, max_length)
        where True = attend, False = ignore
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_length, device=lengths.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1)
    mask = mask < lengths.unsqueeze(1)
    return mask
