"""S3 reference storage for production deployments."""

import asyncio
from pathlib import Path

from loguru import logger

from src.storage.base import ReferenceAudio, ReferenceStorage
from src.storage.local import LocalStorage

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class S3Storage(ReferenceStorage):
    """S3 storage for reference audio.

    Structure:
        s3://bucket/prefix/
            voice_id_1/
                sample.wav
                sample.lab
            voice_id_2/
                sample.wav
                sample.lab
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "references/",
        region: str = "us-east-1",
        cache_dir: Path | str = "cache/references",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._region = region
        self._cache_dir = Path(cache_dir)
        self._local_cache = LocalStorage(self._cache_dir)
        self._voice_map: dict[str, dict[str, str]] = {}
        self._s3_client = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "s3"

    async def initialize(self) -> None:
        if self._initialized:
            return

        await self._local_cache.initialize()

        if not self._bucket:
            logger.warning("No S3 bucket configured, S3 storage unavailable")
            self._initialized = True
            return

        try:
            import boto3

            loop = asyncio.get_event_loop()
            self._s3_client = await loop.run_in_executor(
                None, lambda: boto3.client("s3", region_name=self._region)
            )

            await self._scan_bucket()
            logger.info(f"S3Storage: Found {len(self._voice_map)} voices in s3://{self._bucket}/{self._prefix}")
        except ImportError:
            logger.error("boto3 not installed, S3 storage unavailable")
        except Exception as e:
            logger.error(f"Failed to initialize S3 storage: {e}")

        self._initialized = True

    async def _scan_bucket(self) -> None:
        """Scan S3 bucket for voice references."""
        if not self._s3_client:
            return

        loop = asyncio.get_event_loop()
        paginator = self._s3_client.get_paginator("list_objects_v2")

        audio_files: dict[str, str] = {}
        lab_files: dict[str, str] = {}

        async def list_objects():
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel_path = key[len(self._prefix):]
                    if "/" not in rel_path:
                        continue

                    voice_id = rel_path.split("/")[0]
                    ext = Path(key).suffix.lower()

                    if ext in AUDIO_EXTENSIONS:
                        audio_files[voice_id] = key
                    elif ext == ".lab":
                        lab_files[voice_id] = key

        await loop.run_in_executor(None, list_objects)

        for voice_id, audio_key in audio_files.items():
            self._voice_map[voice_id] = {
                "audio": audio_key,
                "lab": lab_files.get(voice_id, ""),
            }

    async def get_reference(self, voice_id: str) -> ReferenceAudio:
        if await self._local_cache.has_voice(voice_id):
            return await self._local_cache.get_reference(voice_id)

        if voice_id not in self._voice_map:
            raise KeyError(f"Voice '{voice_id}' not found in s3://{self._bucket}/{self._prefix}")

        paths = self._voice_map[voice_id]
        audio_data, transcript = await self._download_voice(voice_id, paths)

        audio_ext = Path(paths["audio"]).suffix
        await self._local_cache.add_reference(voice_id, audio_data, transcript, audio_ext)

        return await self._local_cache.get_reference(voice_id)

    async def _download_voice(
        self, voice_id: str, paths: dict[str, str]
    ) -> tuple[bytes, str]:
        """Download audio and transcript from S3."""
        if not self._s3_client:
            raise RuntimeError("S3 client not initialized")

        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._s3_client.get_object(Bucket=self._bucket, Key=paths["audio"]),
        )
        audio_data = response["Body"].read()

        transcript = ""
        if paths.get("lab"):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._s3_client.get_object(Bucket=self._bucket, Key=paths["lab"]),
                )
                transcript = response["Body"].read().decode("utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to download transcript for {voice_id}: {e}")

        logger.info(f"Downloaded reference from S3: {voice_id}")
        return audio_data, transcript

    async def list_voices(self) -> list[str]:
        local_voices = await self._local_cache.list_voices()
        s3_voices = list(self._voice_map.keys())
        return sorted(set(local_voices) | set(s3_voices))

    async def has_voice(self, voice_id: str) -> bool:
        if await self._local_cache.has_voice(voice_id):
            return True
        return voice_id in self._voice_map

