"""Tests for TTS providers."""

import os

import pytest

# Force mock provider for tests
os.environ["VOICE_TTS_PROVIDER"] = "mock"

from src.providers.base import TTSProvider, VoiceInfo
from src.providers.mock import MockProvider
from src.providers.registry import (
    get_provider,
    list_providers,
    register_provider,
    shutdown_providers,
)


class TestVoiceInfo:
    """Tests for VoiceInfo dataclass."""

    def test_voice_info_creation(self):
        voice = VoiceInfo(
            voice_id="test_voice",
            name="Test Voice",
            description="A test voice",
        )
        assert voice.voice_id == "test_voice"
        assert voice.name == "Test Voice"

    def test_voice_info_optional_fields(self):
        voice = VoiceInfo(voice_id="minimal", name="Minimal")
        assert voice.description is None
        assert voice.preview_url is None
        assert voice.language is None


class TestMockProvider:
    """Tests for MockProvider."""

    @pytest.fixture
    async def provider(self):
        p = MockProvider()
        await p.initialize()
        yield p
        await p.shutdown()

    @pytest.mark.asyncio
    async def test_initialize(self):
        p = MockProvider()
        assert not p.is_ready

        await p.initialize()
        assert p.is_ready
        assert p.name == "mock"

        await p.shutdown()
        assert not p.is_ready

    @pytest.mark.asyncio
    async def test_synthesize_wav_non_streaming(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="Hello world",
            format="wav",
            streaming=False,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        audio_bytes = chunks[0]

        # Verify WAV header
        assert len(audio_bytes) > 44
        assert audio_bytes[:4] == b"RIFF"
        assert audio_bytes[8:12] == b"WAVE"

    @pytest.mark.asyncio
    async def test_synthesize_wav_streaming(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="This is a longer text for streaming test purposes",
            format="wav",
            streaming=True,
        ):
            chunks.append(chunk)

        # Should have multiple chunks for longer text
        assert len(chunks) > 1

        # First chunk should have WAV header
        assert chunks[0][:4] == b"RIFF"
        
        # Verify all chunks together form valid WAV
        full_audio = b"".join(chunks)
        assert full_audio[:4] == b"RIFF"
        assert full_audio[8:12] == b"WAVE"

    @pytest.mark.asyncio
    async def test_synthesize_wav_streaming_short_text(self, provider):
        """Test streaming with short text (may produce single chunk)."""
        chunks = []
        async for chunk in provider.synthesize(
            text="Short",
            format="wav",
            streaming=True,
        ):
            chunks.append(chunk)

        # Should have at least one chunk
        assert len(chunks) >= 1
        assert chunks[0][:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_synthesize_streaming_chunk_sizes(self, provider):
        """Test that streaming chunks are reasonable sizes."""
        chunks = []
        async for chunk in provider.synthesize(
            text="This is a test to verify chunk sizes during streaming.",
            format="pcm",
            streaming=True,
        ):
            chunks.append(chunk)
            # Each chunk should be non-empty
            assert len(chunk) > 0

        assert len(chunks) >= 1
        total_size = sum(len(c) for c in chunks)
        assert total_size > 0

    @pytest.mark.asyncio
    async def test_synthesize_pcm_format(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="PCM test",
            format="pcm",
            streaming=False,
        ):
            chunks.append(chunk)

        audio_bytes = b"".join(chunks)
        assert len(audio_bytes) > 0

        # PCM should NOT have WAV header
        assert audio_bytes[:4] != b"RIFF"

    @pytest.mark.asyncio
    async def test_synthesize_pcm_streaming(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="PCM streaming test with longer text",
            format="pcm",
            streaming=True,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 0

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_id(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="Test with voice ID",
            voice_id="test_voice",
            format="wav",
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, provider):
        chunks = []
        async for chunk in provider.synthesize(
            text="",
            format="wav",
        ):
            chunks.append(chunk)

        # Should still produce some audio (at least header)
        audio_bytes = b"".join(chunks)
        assert len(audio_bytes) >= 44

    @pytest.mark.asyncio
    async def test_synthesize_generates_audible_tone(self, provider):
        """Verify mock generates non-silent audio (sine wave)."""
        chunks = []
        async for chunk in provider.synthesize(
            text="Audible test",
            format="pcm",
            streaming=False,
        ):
            chunks.append(chunk)

        audio_bytes = b"".join(chunks)

        # PCM16 samples should have non-zero values (not silence)
        import struct

        samples = struct.unpack(f"<{len(audio_bytes)//2}h", audio_bytes)
        max_amplitude = max(abs(s) for s in samples)
        assert max_amplitude > 1000  # Should have significant amplitude

    @pytest.mark.asyncio
    async def test_get_voices(self, provider):
        voices = await provider.get_voices()

        assert len(voices) >= 1
        assert isinstance(voices[0], VoiceInfo)
        assert voices[0].voice_id == "mock_default"
        assert voices[0].name == "Mock Default"

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        health = await provider.health_check()

        assert health["provider"] == "mock"
        assert health["ready"] is True

    @pytest.mark.asyncio
    async def test_synthesize_streaming_vs_non_streaming(self, provider):
        """Test that streaming and non-streaming produce similar results."""
        text = "Comparison test for streaming versus non-streaming."
        
        streaming_chunks = []
        async for chunk in provider.synthesize(
            text=text,
            format="pcm",
            streaming=True,
        ):
            streaming_chunks.append(chunk)
        
        non_streaming_chunks = []
        async for chunk in provider.synthesize(
            text=text,
            format="pcm",
            streaming=False,
        ):
            non_streaming_chunks.append(chunk)
        
        streaming_audio = b"".join(streaming_chunks)
        non_streaming_audio = b"".join(non_streaming_chunks)
        
        # Both should produce audio
        assert len(streaming_audio) > 0
        assert len(non_streaming_audio) > 0
        
        # Lengths should be similar (within 10% due to chunking)
        length_diff = abs(len(streaming_audio) - len(non_streaming_audio))
        max_length = max(len(streaming_audio), len(non_streaming_audio))
        assert length_diff / max_length < 0.1

    @pytest.mark.asyncio
    async def test_synthesize_all_formats_streaming(self, provider):
        """Test all formats work with streaming."""
        text = "Format test."
        formats = ["wav", "pcm"]
        
        for fmt in formats:
            chunks = []
            async for chunk in provider.synthesize(
                text=text,
                format=fmt,
                streaming=True,
            ):
                chunks.append(chunk)
            
            assert len(chunks) >= 1
            total_audio = b"".join(chunks)
            assert len(total_audio) > 0


class TestProviderRegistry:
    """Tests for provider registry."""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        yield
        await shutdown_providers()

    def test_register_provider(self):
        register_provider("test_mock", MockProvider)
        assert "test_mock" in list_providers()

    def test_list_providers(self):
        register_provider("list_test", MockProvider)
        providers = list_providers()

        assert isinstance(providers, list)
        assert "list_test" in providers

    @pytest.mark.asyncio
    async def test_get_provider_creates_instance(self):
        register_provider("get_test", MockProvider)
        provider = await get_provider("get_test")

        assert provider is not None
        assert provider.is_ready
        assert provider.name == "mock"

    @pytest.mark.asyncio
    async def test_get_provider_returns_same_instance(self):
        register_provider("same_test", MockProvider)

        provider1 = await get_provider("same_test")
        provider2 = await get_provider("same_test")

        assert provider1 is provider2

    @pytest.mark.asyncio
    async def test_get_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            await get_provider("nonexistent_provider_xyz")

    @pytest.mark.asyncio
    async def test_shutdown_providers(self):
        register_provider("shutdown_test", MockProvider)
        provider = await get_provider("shutdown_test")
        assert provider.is_ready

        await shutdown_providers()
        # Provider should be shutdown (but we can't check is_ready after clear)


class TestTTSProviderInterface:
    """Tests for TTSProvider abstract interface."""

    def test_provider_is_abstract(self):
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            TTSProvider()

    def test_mock_provider_implements_interface(self):
        provider = MockProvider()

        # Should have all required methods
        assert hasattr(provider, "initialize")
        assert hasattr(provider, "shutdown")
        assert hasattr(provider, "synthesize")
        assert hasattr(provider, "get_voices")
        assert hasattr(provider, "health_check")

        # Should have required properties
        assert hasattr(provider, "name")
        assert hasattr(provider, "is_ready")
