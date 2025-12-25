"""Mock TTS provider for testing."""

import io
import struct
from typing import AsyncIterator, Literal

import numpy as np

from src.providers.base import TTSProvider, VoiceInfo


class MockProvider(TTSProvider):
    """Mock TTS provider that generates silence for testing."""

    def __init__(self, sample_rate: int = 24000, **kwargs) -> None:
        self._sample_rate = sample_rate
        self._ready = False

    @property
    def name(self) -> str:
        return "mock"

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def initialize(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

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
        duration_seconds = max(1.0, len(text) * 0.05)
        num_samples = int(self._sample_rate * duration_seconds)

        # Generate a simple sine wave tone (440Hz A note) so audio is audible
        t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
        frequency = 440.0
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        if format == "pcm":
            pcm_data = (audio * 32767).astype(np.int16).tobytes()
            if streaming:
                chunk_size = self._sample_rate // 10 * 2
                for i in range(0, len(pcm_data), chunk_size):
                    yield pcm_data[i : i + chunk_size]
            else:
                yield pcm_data
        else:
            wav_data = self._create_wav(audio)
            if streaming:
                yield wav_data[:44]
                chunk_size = self._sample_rate // 10 * 2
                for i in range(44, len(wav_data), chunk_size):
                    yield wav_data[i : i + chunk_size]
            else:
                yield wav_data

    async def get_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                voice_id="mock_default",
                name="Mock Default",
                description="Default mock voice for testing",
            ),
        ]

    def _create_wav(self, audio: np.ndarray) -> bytes:
        """Create WAV file bytes from audio array."""
        buffer = io.BytesIO()
        pcm_data = (audio * 32767).astype(np.int16)

        # WAV header
        buffer.write(b"RIFF")
        buffer.write(struct.pack("<I", 36 + len(pcm_data) * 2))
        buffer.write(b"WAVE")
        buffer.write(b"fmt ")
        buffer.write(struct.pack("<I", 16))
        buffer.write(struct.pack("<H", 1))
        buffer.write(struct.pack("<H", 1))
        buffer.write(struct.pack("<I", self._sample_rate))
        buffer.write(struct.pack("<I", self._sample_rate * 2))
        buffer.write(struct.pack("<H", 2))
        buffer.write(struct.pack("<H", 16))
        buffer.write(b"data")
        buffer.write(struct.pack("<I", len(pcm_data) * 2))
        buffer.write(pcm_data.tobytes())

        return buffer.getvalue()

