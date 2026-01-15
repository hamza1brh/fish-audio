"""
S1-Mini Backend Base Class
==========================

Abstract base class for all inference backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional, List, Tuple
import numpy as np


class BackendType(Enum):
    """Available backend types."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    type: BackendType
    version: str
    device: str
    precision: str
    is_loaded: bool
    supports_streaming: bool
    estimated_rtf: float  # Estimated real-time factor
    vram_usage_mb: float


@dataclass
class GenerationResult:
    """Result from audio generation."""
    audio: np.ndarray
    sample_rate: int
    generation_time_s: float
    audio_duration_s: float
    rtf: float  # Real-time factor


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    All backends must implement these methods to provide
    a consistent API for TTS generation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        precision: str = "float16",
    ):
        """
        Initialize the backend.

        Args:
            checkpoint_path: Path to model checkpoints
            device: Device to use ("cuda" or "cpu")
            precision: Model precision ("float16", "bfloat16", "float32")
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self._is_loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Backend type enum."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Whether models are loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """
        Load models into memory.

        This should be called before any generation.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload models from memory.

        Frees GPU memory when backend is no longer needed.
        """
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
    ) -> GenerationResult:
        """
        Generate audio from text.

        Args:
            text: Text to synthesize
            reference_audio: Optional reference audio for voice cloning
            reference_text: Optional transcript of reference audio
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            GenerationResult with audio and metadata
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        chunk_length: int = 200,
        **kwargs,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Generate audio in streaming mode.

        Yields audio chunks as they are generated.

        Args:
            text: Text to synthesize
            reference_audio: Optional reference audio
            reference_text: Optional reference transcript
            chunk_length: Characters per chunk
            **kwargs: Additional generation parameters

        Yields:
            Tuple of (audio_chunk, sample_rate)
        """
        pass

    @abstractmethod
    def get_info(self) -> BackendInfo:
        """
        Get information about this backend.

        Returns:
            BackendInfo with backend details
        """
        pass

    def warmup(self, num_iterations: int = 3) -> None:
        """
        Warm up the backend with sample generations.

        This triggers JIT compilation and caches for faster
        subsequent generations.

        Args:
            num_iterations: Number of warmup generations
        """
        if not self.is_loaded:
            self.load()

        from loguru import logger
        logger.info(f"Warming up {self.name} with {num_iterations} iterations...")

        warmup_texts = [
            "Hello, this is a warmup test.",
            "The quick brown fox jumps over the lazy dog.",
            "This is the final warmup iteration.",
        ]

        for i in range(min(num_iterations, len(warmup_texts))):
            try:
                self.generate(warmup_texts[i], max_tokens=256)
                logger.debug(f"Warmup iteration {i + 1} complete")
            except Exception as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")

        logger.info("Backend warmup complete")

    def benchmark(
        self,
        texts: Optional[List[str]] = None,
        num_runs: int = 3,
    ) -> dict:
        """
        Benchmark this backend's performance.

        Args:
            texts: List of texts to benchmark with
            num_runs: Number of runs per text

        Returns:
            Dictionary with benchmark results
        """
        if texts is None:
            texts = [
                "Hello world.",
                "The quick brown fox jumps over the lazy dog.",
                "This is a longer text to test the model's performance with more content.",
            ]

        if not self.is_loaded:
            self.load()

        import time
        results = []

        for text in texts:
            text_results = []
            for _ in range(num_runs):
                start = time.perf_counter()
                result = self.generate(text)
                elapsed = time.perf_counter() - start
                text_results.append({
                    "generation_time": elapsed,
                    "audio_duration": result.audio_duration_s,
                    "rtf": result.rtf,
                })
            results.append({
                "text_length": len(text),
                "runs": text_results,
                "avg_rtf": sum(r["rtf"] for r in text_results) / len(text_results),
            })

        return {
            "backend": self.name,
            "device": self.device,
            "precision": self.precision,
            "results": results,
            "overall_avg_rtf": sum(r["avg_rtf"] for r in results) / len(results),
        }

    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, precision={self.precision}, loaded={self.is_loaded})"
