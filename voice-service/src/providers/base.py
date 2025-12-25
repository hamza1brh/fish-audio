"""Abstract TTS Provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal


@dataclass
class VoiceInfo:
    """Voice metadata."""

    voice_id: str
    name: str
    description: str | None = None
    preview_url: str | None = None
    language: str | None = None


@dataclass
class SynthesisResult:
    """Result of audio synthesis."""

    audio_data: bytes
    sample_rate: int
    format: str
    duration_seconds: float | None = None


class TTSProvider(ABC):
    """Abstract base class for TTS providers.

    Implementations must support both streaming and non-streaming synthesis.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if provider is initialized and ready."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, connect to APIs, etc.)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
        format: Literal["wav", "mp3", "opus", "flac", "pcm"] = "wav",
        streaming: bool = False,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_id: Optional voice identifier for multi-voice support.
            reference_audio: Optional reference audio bytes for zero-shot cloning.
            reference_text: Optional transcript of reference audio.
            format: Output audio format.
            streaming: If True, yield audio chunks as they're generated.
            **kwargs: Provider-specific parameters (temperature, top_p, etc.)

        Yields:
            Audio bytes (complete file or streaming chunks).
        """
        ...

    @abstractmethod
    async def get_voices(self) -> list[VoiceInfo]:
        """List available voices.

        Returns:
            List of available voice configurations.
        """
        ...

    async def health_check(self) -> dict:
        """Perform health check.

        Returns:
            Health status dictionary.
        """
        return {
            "provider": self.name,
            "ready": self.is_ready,
        }











