"""Service configuration with environment variable support."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Required environment variables for S1 Mini provider:
        VOICE_S1_CHECKPOINT_PATH: Path to S1 Mini model checkpoint directory
    """

    model_config = SettingsConfigDict(
        env_prefix="VOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    # TTS Provider selection
    tts_provider: Literal["s1_mini", "elevenlabs", "mock"] = "mock"

    # S1 Mini settings
    # Model can be downloaded from HuggingFace or provided via local path
    s1_model_repo: str = "fishaudio/openaudio-s1-mini"
    s1_checkpoint_path: Path | None = None
    s1_codec_path: Path | None = None
    s1_download_model: bool = False
    s1_compile: bool = False
    s1_device: str = "cuda"
    s1_half: bool = False

    # LoRA settings
    lora_path: Path | None = None
    lora_enabled: bool = False

    # Voice tokens cache (pre-encoded .npy files)
    voices_path: Path = Path("voices")

    # Reference storage settings
    reference_storage: Literal["huggingface", "local", "s3"] = "local"
    hf_reference_repo: str = ""
    hf_token: str | None = None
    local_reference_path: Path = Path("references")

    # S3 settings (for production)
    s3_bucket: str = ""
    s3_prefix: str = "references/"
    aws_region: str = "us-east-1"

    # ElevenLabs settings
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"

    # Inference settings
    max_text_length: int = 2000
    default_chunk_length: int = 200
    streaming_enabled: bool = True

    @field_validator("s1_checkpoint_path", mode="before")
    @classmethod
    def validate_checkpoint_path(cls, v):
        if v is None or v == "":
            return None
        path = Path(v)
        # Resolve relative paths relative to voice-service root directory (standalone)
        if not path.is_absolute():
            # Path(__file__) = voice-service/src/config.py
            # parent.parent = voice-service root
            voice_service_root = Path(__file__).parent.parent
            path = (voice_service_root / path).resolve()
        return path


    @property
    def codec_path(self) -> Path | None:
        """Get codec path, defaulting to checkpoint_path/codec.pth."""
        if self.s1_codec_path:
            return self.s1_codec_path
        if self.s1_checkpoint_path:
            return self.s1_checkpoint_path / "codec.pth"
        return None

    def validate_s1_mini_config(self) -> None:
        """Validate S1 Mini configuration. Raises if invalid."""
        if self.tts_provider != "s1_mini":
            return

        if not self.s1_checkpoint_path and not self.s1_download_model:
            raise ValueError(
                "Either VOICE_S1_CHECKPOINT_PATH or VOICE_S1_DOWNLOAD_MODEL=true is required. "
                "Set checkpoint path or enable automatic download from HuggingFace."
            )


settings = Settings()


