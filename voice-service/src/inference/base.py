"""Abstract TTS backend interface for model swapping."""

from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class TTSBackend(ABC):
    """Abstract interface for TTS inference backends.

    Implement this to add new models (S1-mini, Orpheus, etc.)
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is initialized."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        text: str,
        voice_tokens: np.ndarray | None = None,
        voice_text: str | None = None,
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """Generate audio chunks as they're produced.

        Args:
            text: Text to synthesize.
            voice_tokens: Pre-encoded voice tokens (.npy).
            voice_text: Transcript of reference voice.
            **kwargs: Model-specific params (temperature, top_p, etc.)

        Yields:
            Audio chunks as numpy arrays (float32).
        """
        ...

    @abstractmethod
    def encode_reference(self, audio_bytes: bytes) -> np.ndarray:
        """Encode reference audio to voice tokens.

        Args:
            audio_bytes: Raw audio file bytes.

        Returns:
            Voice tokens as numpy array (save as .npy).
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Load model to GPU."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release GPU memory."""
        ...

    def warmup(self) -> None:
        """Run dummy inference for JIT compilation."""
        pass



