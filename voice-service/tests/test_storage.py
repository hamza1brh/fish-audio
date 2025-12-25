"""Tests for reference storage backends."""

import tempfile
from pathlib import Path

import pytest

from src.storage.base import ReferenceAudio, ReferenceStorage
from src.storage.local import LocalStorage


class TestReferenceAudio:
    """Tests for ReferenceAudio dataclass."""

    def test_reference_audio_creation(self):
        ref = ReferenceAudio(
            voice_id="test_voice",
            audio_data=b"fake audio data",
            transcript="Test transcript",
        )
        assert ref.voice_id == "test_voice"
        assert ref.audio_data == b"fake audio data"
        assert ref.transcript == "Test transcript"

    def test_reference_audio_defaults(self):
        ref = ReferenceAudio(
            voice_id="minimal",
            audio_data=b"data",
            transcript="text",
        )
        assert ref.sample_rate == 24000
        assert ref.language is None
        assert ref.name is None


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    async def storage(self, temp_dir):
        s = LocalStorage(temp_dir)
        await s.initialize()
        yield s

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, temp_dir):
        storage = LocalStorage(temp_dir / "new_dir")
        await storage.initialize()

        assert (temp_dir / "new_dir").exists()

    @pytest.mark.asyncio
    async def test_name_property(self, storage):
        assert storage.name == "local"

    @pytest.mark.asyncio
    async def test_add_reference(self, storage, temp_dir):
        await storage.add_reference(
            voice_id="test_voice",
            audio_data=b"RIFF....WAVEfmt ",
            transcript="Test transcript",
        )

        assert (temp_dir / "test_voice" / "sample.wav").exists()
        assert (temp_dir / "test_voice" / "sample.lab").exists()

    @pytest.mark.asyncio
    async def test_get_reference(self, storage, temp_dir):
        await storage.add_reference(
            voice_id="get_test",
            audio_data=b"fake audio bytes",
            transcript="Get test transcript",
        )

        ref = await storage.get_reference("get_test")

        assert ref.voice_id == "get_test"
        assert ref.audio_data == b"fake audio bytes"
        assert ref.transcript == "Get test transcript"

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises(self, storage):
        with pytest.raises(KeyError):
            await storage.get_reference("nonexistent_voice")

    @pytest.mark.asyncio
    async def test_list_voices(self, storage):
        await storage.add_reference("voice1", b"data1", "text1")
        await storage.add_reference("voice2", b"data2", "text2")

        voices = await storage.list_voices()

        assert "voice1" in voices
        assert "voice2" in voices

    @pytest.mark.asyncio
    async def test_has_voice_true(self, storage):
        await storage.add_reference("exists", b"data", "text")

        result = await storage.has_voice("exists")
        assert result is True

    @pytest.mark.asyncio
    async def test_has_voice_false(self, storage):
        result = await storage.has_voice("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_caching(self, storage):
        await storage.add_reference("cached", b"data", "text")

        ref1 = await storage.get_reference("cached")
        ref2 = await storage.get_reference("cached")

        assert ref1 is ref2


class TestStorageInterface:
    """Tests for ReferenceStorage abstract interface."""

    def test_storage_is_abstract(self):
        with pytest.raises(TypeError):
            ReferenceStorage()

    def test_local_storage_implements_interface(self):
        storage = LocalStorage(Path("/tmp"))

        assert hasattr(storage, "initialize")
        assert hasattr(storage, "get_reference")
        assert hasattr(storage, "list_voices")
        assert hasattr(storage, "has_voice")
        assert hasattr(storage, "name")
