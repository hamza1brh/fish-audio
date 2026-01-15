"""
S1-Mini PyTorch Backend
=======================

PyTorch-based inference backend with optimized compilation.
This is the default backend that works on all platforms.
"""

import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .base import InferenceBackend, BackendType, BackendInfo, GenerationResult


class PyTorchBackend(InferenceBackend):
    """
    PyTorch inference backend with platform-aware optimization.

    This backend uses:
    - torch.compile with optimal settings for the platform
    - Flash Attention when available
    - cuDNN autotuning
    - TF32 on Ampere+ GPUs
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/openaudio-s1-mini",
        device: str = "cuda",
        precision: str = "float16",
        compile_model: bool = True,
    ):
        """
        Initialize PyTorch backend.

        Args:
            checkpoint_path: Path to model checkpoints
            device: Device to use ("cuda" or "cpu")
            precision: Model precision ("float16", "bfloat16", "float32")
            compile_model: Whether to use torch.compile
        """
        super().__init__(checkpoint_path, device, precision)
        self.compile_model = compile_model
        self._model_manager = None
        self._tts_engine = None
        self._sample_rate = 44100

    @property
    def name(self) -> str:
        return "PyTorch (Optimized)"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PYTORCH

    def load(self) -> None:
        """Load models into memory."""
        if self._is_loaded:
            logger.warning("Models already loaded")
            return

        logger.info(f"Loading PyTorch backend from {self.checkpoint_path}")

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import and configure
        from s1_mini.config import EngineConfig
        from s1_mini.model_manager import ModelManager
        from s1_mini.compilation import compile_model as do_compile
        from s1_mini.attention import configure_optimal_attention
        from fish_speech.inference_engine import TTSInferenceEngine

        # Configure optimal attention
        configure_optimal_attention()

        # Create config
        config = EngineConfig(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile_model=self.compile_model,
        )

        # Load models
        self._model_manager = ModelManager(config)
        self._model_manager.load_models()

        # Create TTS engine
        self._tts_engine = TTSInferenceEngine(
            llama_queue=self._model_manager.llama_queue,
            decoder_model=self._model_manager.decoder_model,
            precision=config.torch_dtype,
            compile=self.compile_model,
        )

        self._is_loaded = True
        logger.info("PyTorch backend loaded successfully")

    def unload(self) -> None:
        """Unload models from memory."""
        if not self._is_loaded:
            return

        logger.info("Unloading PyTorch backend")

        if self._model_manager:
            self._model_manager.shutdown()
            self._model_manager = None

        self._tts_engine = None
        self._is_loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PyTorch backend unloaded")

    def generate(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
    ) -> GenerationResult:
        """Generate audio from text."""
        if not self._is_loaded:
            raise RuntimeError("Backend not loaded. Call load() first.")

        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        # Build references
        references = []
        if reference_audio and reference_text:
            references.append(ServeReferenceAudio(
                audio=reference_audio,
                text=reference_text,
            ))

        # Create request
        request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=max_tokens,
            chunk_length=200,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            format="wav",
        )

        # Generate
        start_time = time.perf_counter()

        audio_data = None
        sample_rate = self._sample_rate

        for result in self._tts_engine.inference(request):
            if result.code == "final" and result.audio is not None:
                sample_rate, audio_data = result.audio
                break

        generation_time = time.perf_counter() - start_time

        if audio_data is None:
            raise RuntimeError("No audio generated")

        audio_duration = len(audio_data) / sample_rate
        rtf = audio_duration / generation_time if generation_time > 0 else 0

        return GenerationResult(
            audio=audio_data,
            sample_rate=sample_rate,
            generation_time_s=generation_time,
            audio_duration_s=audio_duration,
            rtf=rtf,
        )

    def generate_stream(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        chunk_length: int = 200,
        **kwargs,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate audio in streaming mode."""
        if not self._is_loaded:
            raise RuntimeError("Backend not loaded. Call load() first.")

        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        # Build references
        references = []
        if reference_audio and reference_text:
            references.append(ServeReferenceAudio(
                audio=reference_audio,
                text=reference_text,
            ))

        # Create request
        request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=kwargs.get("max_tokens", 2048),
            chunk_length=chunk_length,
            top_p=kwargs.get("top_p", 0.7),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            temperature=kwargs.get("temperature", 0.7),
            format="wav",
        )

        # Stream generation
        for result in self._tts_engine.inference(request):
            if result.audio is not None:
                sample_rate, audio_data = result.audio
                yield audio_data, sample_rate

    def get_info(self) -> BackendInfo:
        """Get backend information."""
        vram_usage = 0.0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024 * 1024)

        return BackendInfo(
            name=self.name,
            type=self.backend_type,
            version=torch.__version__,
            device=self.device,
            precision=self.precision,
            is_loaded=self._is_loaded,
            supports_streaming=True,
            estimated_rtf=0.7,  # Conservative estimate for Windows
            vram_usage_mb=vram_usage,
        )

    def get_vram_usage_mb(self) -> float:
        """Get current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
