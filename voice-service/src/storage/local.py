"""Local filesystem reference storage."""

from pathlib import Path

import aiofiles
from loguru import logger

from src.storage.base import ReferenceAudio, ReferenceStorage

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class LocalStorage(ReferenceStorage):
    """Local filesystem storage for reference audio.

    Expected structure:
        references/
            voice_id_1/
                sample.wav
                sample.lab  (transcript)
            voice_id_2/
                audio.mp3
                audio.lab
    """

    def __init__(self, base_path: Path | str = "references") -> None:
        self._base_path = Path(base_path)
        self._cache: dict[str, ReferenceAudio] = {}

    @property
    def name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized at {self._base_path}")

    async def get_reference(self, voice_id: str) -> ReferenceAudio:
        if voice_id in self._cache:
            return self._cache[voice_id]

        voice_dir = self._base_path / voice_id
        if not voice_dir.exists():
            raise KeyError(f"Voice '{voice_id}' not found in {self._base_path}")

        audio_file = self._find_audio_file(voice_dir)
        if not audio_file:
            raise KeyError(f"No audio file found for voice '{voice_id}'")

        lab_file = audio_file.with_suffix(".lab")
        if not lab_file.exists():
            lab_file = voice_dir / "sample.lab"

        transcript = ""
        if lab_file.exists():
            async with aiofiles.open(lab_file, "r", encoding="utf-8") as f:
                transcript = await f.read()
                transcript = transcript.strip()

        async with aiofiles.open(audio_file, "rb") as f:
            audio_data = await f.read()

        reference = ReferenceAudio(
            voice_id=voice_id,
            audio_data=audio_data,
            transcript=transcript,
            name=voice_id,
        )
        self._cache[voice_id] = reference
        return reference

    async def list_voices(self) -> list[str]:
        if not self._base_path.exists():
            return []

        voices = []
        for item in self._base_path.iterdir():
            if item.is_dir() and self._find_audio_file(item):
                voices.append(item.name)
        return sorted(voices)

    async def has_voice(self, voice_id: str) -> bool:
        voice_dir = self._base_path / voice_id
        if not voice_dir.exists():
            return False
        return self._find_audio_file(voice_dir) is not None

    def _find_audio_file(self, directory: Path) -> Path | None:
        """Find first audio file in directory."""
        for ext in AUDIO_EXTENSIONS:
            for audio_file in directory.glob(f"*{ext}"):
                return audio_file
        return None

    async def add_reference(
        self, voice_id: str, audio_data: bytes, transcript: str, audio_ext: str = ".wav"
    ) -> None:
        """Add a reference to local storage.

        Args:
            voice_id: Unique voice identifier.
            audio_data: Audio file bytes.
            transcript: Transcript text.
            audio_ext: Audio file extension.
        """
        voice_dir = self._base_path / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        audio_file = voice_dir / f"sample{audio_ext}"
        lab_file = voice_dir / "sample.lab"

        async with aiofiles.open(audio_file, "wb") as f:
            await f.write(audio_data)

        async with aiofiles.open(lab_file, "w", encoding="utf-8") as f:
            await f.write(transcript)

        if voice_id in self._cache:
            del self._cache[voice_id]

        logger.info(f"Added reference: {voice_id}")

    async def delete_reference(self, voice_id: str) -> bool:
        """Delete a reference voice.

        Args:
            voice_id: Voice identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        voice_dir = self._base_path / voice_id
        if not voice_dir.exists():
            return False

        import shutil

        shutil.rmtree(voice_dir)
        if voice_id in self._cache:
            del self._cache[voice_id]

        logger.info(f"Deleted reference: {voice_id}")
        return True

