"""Abstract reference storage interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ReferenceAudio:
    """Reference audio with metadata for zero-shot cloning."""

    voice_id: str
    audio_data: bytes
    transcript: str
    sample_rate: int = 24000
    language: str | None = None
    name: str | None = None


class ReferenceStorage(ABC):
    """Abstract base class for reference audio storage.

    Implementations handle fetching reference audio from various sources
    (HuggingFace, local filesystem, S3, etc.) for zero-shot voice cloning.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Storage backend name."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage connection."""
        ...

    @abstractmethod
    async def get_reference(self, voice_id: str) -> ReferenceAudio:
        """Fetch reference audio by voice ID.

        Args:
            voice_id: Unique identifier for the voice.

        Returns:
            ReferenceAudio with audio data and transcript.

        Raises:
            KeyError: If voice_id not found.
            IOError: If fetch fails.
        """
        ...

    @abstractmethod
    async def list_voices(self) -> list[str]:
        """List available voice IDs.

        Returns:
            List of voice identifiers.
        """
        ...

    @abstractmethod
    async def has_voice(self, voice_id: str) -> bool:
        """Check if voice exists.

        Args:
            voice_id: Voice identifier to check.

        Returns:
            True if voice exists, False otherwise.
        """
        ...

    async def list_references(self) -> list[ReferenceAudio]:
        """List all reference audio objects.

        Returns:
            List of ReferenceAudio objects.
        """
        voice_ids = await self.list_voices()
        references = []
        for voice_id in voice_ids:
            try:
                ref = await self.get_reference(voice_id)
                references.append(ref)
            except (KeyError, IOError):
                continue
        return references

    async def delete_reference(self, voice_id: str) -> bool:
        """Delete a reference voice.

        Args:
            voice_id: Voice identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        if not await self.has_voice(voice_id):
            return False
        raise NotImplementedError("delete_reference must be implemented by subclass")

    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass











