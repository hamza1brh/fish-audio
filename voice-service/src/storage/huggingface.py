"""HuggingFace Hub reference storage."""

import asyncio
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from src.storage.base import ReferenceAudio, ReferenceStorage
from src.storage.local import LocalStorage

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class HuggingFaceStorage(ReferenceStorage):
    """HuggingFace Hub storage for reference audio.

    Supports two repository structures:

    1. Flat structure:
        repo/
            voice_id_1.wav
            voice_id_1.lab
            voice_id_2.wav
            voice_id_2.lab

    2. Directory structure:
        repo/
            voice_id_1/
                sample.wav
                sample.lab
            voice_id_2/
                sample.wav
                sample.lab

    Downloads files on first access and caches locally.
    """

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        cache_dir: Path | str = "cache/references",
        revision: str = "main",
    ) -> None:
        self._repo_id = repo_id
        self._token = token
        self._cache_dir = Path(cache_dir)
        self._revision = revision
        self._local_cache = LocalStorage(self._cache_dir)
        self._voice_map: dict[str, dict[str, str]] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        return "huggingface"

    async def initialize(self) -> None:
        if self._initialized:
            return

        await self._local_cache.initialize()

        if not self._repo_id:
            logger.warning("No HuggingFace repo configured, HF storage unavailable")
            self._initialized = True
            return

        loop = asyncio.get_event_loop()
        try:
            files = await loop.run_in_executor(
                None, lambda: list_repo_files(self._repo_id, token=self._token)
            )
            self._parse_repo_structure(files)
            logger.info(f"HuggingFaceStorage: Found {len(self._voice_map)} voices in {self._repo_id}")
        except Exception as e:
            logger.error(f"Failed to list HuggingFace repo: {e}")

        self._initialized = True

    def _parse_repo_structure(self, files: list[str]) -> None:
        """Parse repository files to build voice map."""
        audio_files: dict[str, str] = {}
        lab_files: dict[str, str] = {}

        for filepath in files:
            path = Path(filepath)
            ext = path.suffix.lower()

            if ext in AUDIO_EXTENSIONS:
                if "/" in filepath:
                    voice_id = path.parent.name
                else:
                    voice_id = path.stem
                audio_files[voice_id] = filepath

            elif ext == ".lab":
                if "/" in filepath:
                    voice_id = path.parent.name
                else:
                    voice_id = path.stem
                lab_files[voice_id] = filepath

        for voice_id, audio_path in audio_files.items():
            self._voice_map[voice_id] = {
                "audio": audio_path,
                "lab": lab_files.get(voice_id, ""),
            }

    async def get_reference(self, voice_id: str) -> ReferenceAudio:
        if await self._local_cache.has_voice(voice_id):
            return await self._local_cache.get_reference(voice_id)

        if voice_id not in self._voice_map:
            raise KeyError(f"Voice '{voice_id}' not found in {self._repo_id}")

        paths = self._voice_map[voice_id]
        audio_data, transcript = await self._download_voice(voice_id, paths)

        audio_ext = Path(paths["audio"]).suffix
        await self._local_cache.add_reference(voice_id, audio_data, transcript, audio_ext)

        return await self._local_cache.get_reference(voice_id)

    async def _download_voice(
        self, voice_id: str, paths: dict[str, str]
    ) -> tuple[bytes, str]:
        """Download audio and transcript from HuggingFace."""
        loop = asyncio.get_event_loop()

        audio_path = await loop.run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=self._repo_id,
                filename=paths["audio"],
                token=self._token,
                revision=self._revision,
            ),
        )

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        transcript = ""
        if paths.get("lab"):
            try:
                lab_path = await loop.run_in_executor(
                    None,
                    lambda: hf_hub_download(
                        repo_id=self._repo_id,
                        filename=paths["lab"],
                        token=self._token,
                        revision=self._revision,
                    ),
                )
                with open(lab_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to download transcript for {voice_id}: {e}")

        logger.info(f"Downloaded reference: {voice_id}")
        return audio_data, transcript

    async def list_voices(self) -> list[str]:
        local_voices = await self._local_cache.list_voices()
        hf_voices = list(self._voice_map.keys())
        return sorted(set(local_voices) | set(hf_voices))

    async def has_voice(self, voice_id: str) -> bool:
        if await self._local_cache.has_voice(voice_id):
            return True
        return voice_id in self._voice_map

