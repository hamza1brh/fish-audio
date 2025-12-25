"""S1 Mini inference backend."""

import gc
import io
import queue
import struct
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from loguru import logger

from src.inference.base import TTSBackend
from src.inference.lora_manager import LoRAManager


class S1MiniBackend(TTSBackend):
    """OpenAudio S1-mini TTS backend.

    Implements TTSBackend interface for easy model swapping.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        codec_path: Path,
        device: str = "cuda",
        compile: bool = False,
        half: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._codec_path = codec_path
        self._device = device
        self._compile = compile
        self._half = half

        self._precision = torch.half if half else torch.bfloat16

        self._llama_queue: queue.Queue | None = None
        self._decoder_model = None
        self._sample_rate_value = 24000
        self._lora_manager = LoRAManager()

        self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_value

    @property
    def device(self) -> str:
        return self._device

    @property
    def lora_manager(self) -> LoRAManager:
        return self._lora_manager

    def initialize(self) -> None:
        """Load model to GPU."""
        if self._initialized:
            return

        self._setup_python_path()

        logger.info(f"Initializing S1 Mini backend on {self._device}")
        logger.info(f"  Checkpoint: {self._checkpoint_path}")
        logger.info(f"  Codec: {self._codec_path}")
        logger.info(f"  Compile: {self._compile}")

        self._load_llama_model()
        self._load_decoder_model()

        self._initialized = True
        logger.info("S1 Mini backend initialized")

    def _setup_python_path(self) -> None:
        """Ensure fish_speech is available as package."""
        try:
            import fish_speech  # noqa: F401

            logger.info("Using fish_speech from installed package")
        except ImportError:
            logger.error(
                "fish_speech package not found. Install it with: pip install fish-speech"
            )
            raise ImportError(
                "fish_speech is required for S1 Mini. Install with: pip install fish-speech"
            )

    def _load_llama_model(self) -> None:
        """Load the LLaMA text-to-semantic model."""
        try:
            from fish_speech.models.text2semantic.inference import (
                launch_thread_safe_queue,
            )

            self._llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(self._checkpoint_path),
                device=self._device,
                precision=self._precision,
                compile=self._compile,
            )
            logger.info("LLaMA model loaded")
        except Exception as e:
            logger.error(f"Failed to load LLaMA model: {e}")
            raise

    def _load_decoder_model(self) -> None:
        """Load the DAC decoder model."""
        try:
            from fish_speech.models.dac.inference import load_model as load_decoder

            self._decoder_model = load_decoder(
                config_name="modded_dac_vq",
                checkpoint_path=str(self._codec_path),
                device=self._device,
            )

            if hasattr(self._decoder_model, "sample_rate"):
                self._sample_rate_value = self._decoder_model.sample_rate

            logger.info(f"Decoder model loaded (sample_rate={self._sample_rate_value})")
        except Exception as e:
            logger.error(f"Failed to load decoder model: {e}")
            raise

    def shutdown(self) -> None:
        """Release GPU memory."""
        self._llama_queue = None
        self._decoder_model = None
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("S1 Mini backend shutdown")

    def warmup(self) -> None:
        """Run dummy inference for JIT compilation."""
        if not self._initialized:
            return

        logger.info("Warming up S1 Mini backend...")
        try:
            list(self.generate_stream("Hello", max_new_tokens=64))
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def encode_reference(self, audio_bytes: bytes) -> np.ndarray:
        """Encode reference audio to voice tokens.

        Args:
            audio_bytes: Raw audio file bytes.

        Returns:
            Voice tokens as numpy array.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")

        import soundfile as sf

        audio_buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_buffer)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        if sr != self._sample_rate_value:
            try:
                import torchaudio.transforms as T

                resampler = T.Resample(sr, self._sample_rate_value)
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                audio = resampler(audio_tensor).squeeze(0).numpy()
            except ImportError:
                logger.warning("torchaudio not available for resampling")

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            indices, _ = self._decoder_model.encode(
                audio_tensor.unsqueeze(0),
                torch.tensor([audio_tensor.shape[-1]], device=self._device),
            )

        return indices.cpu().numpy()

    def generate_stream(
        self,
        text: str,
        voice_tokens: np.ndarray | None = None,
        voice_text: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        chunk_length: int = 200,
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """Generate audio chunks as they're produced.

        Args:
            text: Text to synthesize.
            voice_tokens: Pre-encoded voice tokens (.npy).
            voice_text: Transcript of reference voice.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            repetition_penalty: Repetition penalty.
            chunk_length: Characters per chunk.
            **kwargs: Additional params (ignored).

        Yields:
            Audio chunks as numpy arrays (float32).
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        from fish_speech.models.text2semantic.inference import (
            GenerateRequest,
            GenerateResponse,
            WrappedGenerateResponse,
        )
        from fish_speech.utils import autocast_exclude_mps

        prompt_tokens = [voice_tokens] if voice_tokens is not None else []
        prompt_texts = [voice_text] if voice_text else []

        request = dict(
            device=self._decoder_model.device,
            max_new_tokens=max_new_tokens,
            text=text,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=self._compile,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )

        response_queue = queue.Queue()
        self._llama_queue.put(
            GenerateRequest(request=request, response_queue=response_queue)
        )

        while True:
            wrapped_result: WrappedGenerateResponse = response_queue.get()

            if wrapped_result.status == "error":
                error = wrapped_result.response
                raise RuntimeError(f"Generation error: {error}")

            result: GenerateResponse = wrapped_result.response

            if result.action != "next":
                with autocast_exclude_mps(
                    device_type=self._decoder_model.device.type,
                    dtype=self._precision,
                ):
                    segment = self._decode_vq_tokens(result.codes)
                yield segment
            else:
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _decode_vq_tokens(self, codes: torch.Tensor) -> np.ndarray:
        """Decode VQ tokens to audio."""
        with torch.inference_mode():
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)

            feature_lengths = torch.tensor(
                [codes.shape[-1]], device=self._device, dtype=torch.long
            )
            audio, _ = self._decoder_model.decode(
                indices=codes,
                feature_lengths=feature_lengths,
            )

        return audio.float().cpu().numpy().squeeze()

    def generate(
        self,
        text: str,
        voice_tokens: np.ndarray | None = None,
        voice_text: str | None = None,
        streaming: bool = False,
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """Generate audio (convenience wrapper for generate_stream).

        Args:
            text: Text to synthesize.
            voice_tokens: Pre-encoded voice tokens.
            voice_text: Transcript of reference voice.
            streaming: If True, yield chunks as produced. If False, yield single array.
            **kwargs: Passed to generate_stream.

        Yields:
            Audio chunks (streaming=True) or single concatenated array (streaming=False).
        """
        if streaming:
            yield from self.generate_stream(
                text=text,
                voice_tokens=voice_tokens,
                voice_text=voice_text,
                **kwargs,
            )
        else:
            chunks = list(
                self.generate_stream(
                    text=text,
                    voice_tokens=voice_tokens,
                    voice_text=voice_text,
                    **kwargs,
                )
            )
            if chunks:
                yield np.concatenate(chunks)

    def generate_batch(
        self,
        requests: list[dict],
    ) -> list[np.ndarray]:
        """Generate audio for multiple requests sequentially.

        Args:
            requests: List of dicts with 'text' and optional generation params.

        Returns:
            List of audio arrays, one per request.
        """
        results = []
        for req in requests:
            req_copy = dict(req)
            text = req_copy.pop("text")
            chunks = list(self.generate_stream(text=text, **req_copy))
            if chunks:
                results.append(np.concatenate(chunks))
            else:
                results.append(np.array([], dtype=np.float32))
        return results

    def create_wav_header(self, sample_rate: int | None = None) -> bytes:
        """Create WAV header for streaming."""
        sr = sample_rate or self._sample_rate_value
        buffer = io.BytesIO()

        buffer.write(b"RIFF")
        buffer.write(struct.pack("<I", 0xFFFFFFFF))
        buffer.write(b"WAVE")
        buffer.write(b"fmt ")
        buffer.write(struct.pack("<I", 16))
        buffer.write(struct.pack("<H", 1))
        buffer.write(struct.pack("<H", 1))
        buffer.write(struct.pack("<I", sr))
        buffer.write(struct.pack("<I", sr * 2))
        buffer.write(struct.pack("<H", 2))
        buffer.write(struct.pack("<H", 16))
        buffer.write(b"data")
        buffer.write(struct.pack("<I", 0xFFFFFFFF))

        return buffer.getvalue()


# Backwards compatibility alias
S1MiniEngine = S1MiniBackend
