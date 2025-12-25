"""ElevenLabs TTS provider implementation."""

import io
from typing import AsyncIterator, Literal

import httpx
from loguru import logger

from src.config import settings
from src.providers.base import TTSProvider, VoiceInfo


class ElevenLabsProvider(TTSProvider):
    """TTS Provider using ElevenLabs API.

    Provides a fallback option when local models are unavailable
    or for comparison purposes.
    """

    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.elevenlabs_api_key
        self._model = model or settings.elevenlabs_model
        self._client: httpx.AsyncClient | None = None
        self._voices_cache: list[VoiceInfo] = []

    @property
    def name(self) -> str:
        return "elevenlabs"

    @property
    def is_ready(self) -> bool:
        return self._client is not None and bool(self._api_key)

    async def initialize(self) -> None:
        if not self._api_key:
            logger.warning("ElevenLabs API key not configured")
            return

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

        try:
            await self._fetch_voices()
            logger.info(f"ElevenLabsProvider initialized with {len(self._voices_cache)} voices")
        except Exception as e:
            logger.error(f"Failed to fetch ElevenLabs voices: {e}")

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("ElevenLabsProvider shutdown")

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
        if not self._client:
            raise RuntimeError("ElevenLabsProvider not initialized")

        voice = voice_id or "21m00Tcm4TlvDq8ikWAM"

        format_map = {
            "wav": "pcm_24000",
            "mp3": "mp3_44100_128",
            "pcm": "pcm_24000",
        }
        output_format = format_map.get(format, "mp3_44100_128")

        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
            },
        }

        if streaming:
            url = f"/text-to-speech/{voice}/stream"
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                params={"output_format": output_format},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    yield chunk
        else:
            url = f"/text-to-speech/{voice}"
            response = await self._client.post(
                url,
                json=payload,
                params={"output_format": output_format},
            )
            response.raise_for_status()
            yield response.content

    async def get_voices(self) -> list[VoiceInfo]:
        if not self._voices_cache:
            await self._fetch_voices()
        return self._voices_cache

    async def _fetch_voices(self) -> None:
        """Fetch available voices from ElevenLabs API."""
        if not self._client:
            return

        response = await self._client.get("/voices")
        response.raise_for_status()
        data = response.json()

        self._voices_cache = []
        for voice in data.get("voices", []):
            self._voices_cache.append(
                VoiceInfo(
                    voice_id=voice["voice_id"],
                    name=voice["name"],
                    description=voice.get("description", ""),
                    preview_url=voice.get("preview_url"),
                )
            )

    async def health_check(self) -> dict:
        status = await super().health_check()
        status["model"] = self._model
        status["voices_count"] = len(self._voices_cache)
        return status

