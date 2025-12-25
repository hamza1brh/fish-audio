"""S1 Mini TTS provider - thin wrapper over TTSBackend."""

import asyncio
import io
import queue
from pathlib import Path
from typing import AsyncIterator, Literal

import numpy as np
import soundfile as sf
from loguru import logger

from src.config import settings
from src.core.voice_cache import VoiceCache
from src.inference.base import TTSBackend
from src.inference.engine import S1MiniBackend
from src.providers.base import TTSProvider, VoiceInfo
from src.storage.base import ReferenceStorage
from src.utils.model_downloader import download_s1_mini_model


class S1MiniProvider(TTSProvider):
    """TTS Provider using TTSBackend interface.

    Thin wrapper that delegates to any TTSBackend implementation.
    """

    def __init__(
        self,
        backend: TTSBackend | None = None,
        voice_cache: VoiceCache | None = None,
        storage: ReferenceStorage | None = None,
        checkpoint_path: Path | None = None,
        codec_path: Path | None = None,
        device: str | None = None,
        compile: bool | None = None,
        half: bool | None = None,
    ) -> None:
        self._backend = backend
        self._voice_cache = voice_cache
        self._storage = storage

        self._checkpoint_path = checkpoint_path or settings.s1_checkpoint_path
        self._codec_path = codec_path or settings.codec_path
        self._device = device or settings.s1_device
        self._compile = compile if compile is not None else settings.s1_compile
        self._half = half if half is not None else settings.s1_half

    @property
    def name(self) -> str:
        return "s1_mini"

    @property
    def is_ready(self) -> bool:
        return self._backend is not None and self._backend.is_ready

    @property
    def backend(self) -> TTSBackend | None:
        return self._backend

    async def initialize(self) -> None:
        loop = asyncio.get_event_loop()

        if self._backend is None:
            checkpoint_path = self._checkpoint_path
            if settings.s1_download_model and not checkpoint_path:
                logger.info("Downloading model from HuggingFace...")
                checkpoint_path = await loop.run_in_executor(
                    None,
                    lambda: download_s1_mini_model(
                        repo_id=settings.s1_model_repo,
                        token=settings.hf_token,
                    ),
                )
                self._checkpoint_path = checkpoint_path

            if not checkpoint_path:
                raise ValueError(
                    "Checkpoint path not set. Set VOICE_S1_CHECKPOINT_PATH or "
                    "VOICE_S1_DOWNLOAD_MODEL=true to download from HuggingFace."
                )

            codec_path = self._codec_path or (checkpoint_path / "codec.pth")

            self._backend = S1MiniBackend(
                checkpoint_path=checkpoint_path,
                codec_path=codec_path,
                device=self._device,
                compile=self._compile,
                half=self._half,
            )

        await loop.run_in_executor(None, self._backend.initialize)

        if self._voice_cache is None:
            self._voice_cache = VoiceCache(settings.voices_path)
        await loop.run_in_executor(None, self._voice_cache.load_all)

        if self._storage:
            await self._storage.initialize()

        logger.info(f"S1MiniProvider initialized on {self._device}")

    async def shutdown(self) -> None:
        if self._backend:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._backend.shutdown)
            self._backend = None

        logger.info("S1MiniProvider shutdown")

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
        if not self._backend or not self._backend.is_ready:
            raise RuntimeError("S1MiniProvider not initialized")

        voice_tokens, voice_text = await self._get_voice_tokens(
            voice_id, reference_audio, reference_text
        )

        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)
        repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        chunk_length = kwargs.get("chunk_length", 200)

        if streaming:
            if format == "wav":
                yield self._backend.create_wav_header()

            async for chunk in self._stream_from_backend(
                text=text,
                voice_tokens=voice_tokens,
                voice_text=voice_text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
            ):
                yield chunk
        else:
            audio = await self._generate_complete(
                text=text,
                voice_tokens=voice_tokens,
                voice_text=voice_text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            yield self._encode_audio(audio, self._backend.sample_rate, format)

    async def _stream_from_backend(
        self,
        text: str,
        voice_tokens: np.ndarray | None,
        voice_text: str | None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated."""
        chunk_queue: queue.Queue = queue.Queue()
        loop = asyncio.get_event_loop()

        def run_generation():
            try:
                for chunk in self._backend.generate_stream(
                    text=text,
                    voice_tokens=voice_tokens,
                    voice_text=voice_text,
                    **kwargs,
                ):
                    chunk_queue.put(("chunk", chunk))
            except Exception as e:
                chunk_queue.put(("error", e))
            finally:
                chunk_queue.put(("done", None))

        loop.run_in_executor(None, run_generation)

        while True:
            msg_type, data = await loop.run_in_executor(None, chunk_queue.get)
            if msg_type == "done":
                break
            elif msg_type == "error":
                raise data
            else:
                yield self._to_pcm16(data)

    async def _generate_complete(
        self,
        text: str,
        voice_tokens: np.ndarray | None,
        voice_text: str | None,
        **kwargs,
    ) -> np.ndarray:
        """Generate complete audio (non-streaming)."""
        loop = asyncio.get_event_loop()

        def run():
            segments = list(
                self._backend.generate_stream(
                    text=text,
                    voice_tokens=voice_tokens,
                    voice_text=voice_text,
                    chunk_length=0,
                    **kwargs,
                )
            )
            if not segments:
                raise RuntimeError("No audio generated")
            return np.concatenate(segments) if len(segments) > 1 else segments[0]

        return await loop.run_in_executor(None, run)

    async def _get_voice_tokens(
        self,
        voice_id: str | None,
        reference_audio: bytes | None,
        reference_text: str | None,
    ) -> tuple[np.ndarray | None, str | None]:
        """Get voice tokens from cache, storage, or encode from audio."""
        if voice_id and self._voice_cache:
            cached = self._voice_cache.get(voice_id)
            if cached:
                return cached

        if voice_id and self._storage:
            try:
                ref = await self._storage.get_reference(voice_id)
                reference_audio = ref.audio_data
                reference_text = ref.transcript
            except KeyError:
                logger.warning(f"Voice '{voice_id}' not found in storage")

        if reference_audio:
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(
                None,
                lambda: self._backend.encode_reference(reference_audio),
            )
            return tokens, reference_text

        return None, None

    async def get_voices(self) -> list[VoiceInfo]:
        voices = [
            VoiceInfo(
                voice_id="default",
                name="Default",
                description="Default S1 Mini voice (no reference)",
            )
        ]

        if self._voice_cache:
            for vid in self._voice_cache.list_voices():
                voices.append(
                    VoiceInfo(
                        voice_id=vid,
                        name=vid,
                        description=f"Pre-encoded voice: {vid}",
                    )
                )

        if self._storage:
            voice_ids = await self._storage.list_voices()
            for vid in voice_ids:
                if not any(v.voice_id == vid for v in voices):
                    voices.append(
                        VoiceInfo(
                            voice_id=vid,
                            name=vid,
                            description=f"Reference voice: {vid}",
                        )
                    )

        return voices

    async def health_check(self) -> dict:
        status = await super().health_check()
        status.update(
            {
                "device": self._device,
                "compile": self._compile,
                "sample_rate": self._backend.sample_rate if self._backend else None,
                "cached_voices": len(self._voice_cache.list_voices())
                if self._voice_cache
                else 0,
            }
        )
        return status

    def _to_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float audio to PCM16 bytes."""
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        return pcm.tobytes()

    def _encode_audio(self, audio: np.ndarray, sample_rate: int, format: str) -> bytes:
        """Encode audio to requested format."""
        buffer = io.BytesIO()

        if format == "pcm":
            return self._to_pcm16(audio)

        format_map = {
            "wav": "WAV",
            "mp3": "MP3",
            "opus": "OGG",
            "flac": "FLAC",
        }

        sf.write(buffer, audio, sample_rate, format=format_map.get(format, "WAV"))
        return buffer.getvalue()
