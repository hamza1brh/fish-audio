"""Configuration and environment variable loading for Voice Call Neymar."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LiveKitConfig:
    """LiveKit server configuration."""
    url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")


@dataclass
class GroqConfig:
    """Groq API configuration."""
    api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    stt_model: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")


@dataclass
class ElevenLabsConfig:
    """ElevenLabs API configuration."""
    api_key: str = os.getenv("ELEVEN_LABS_KEY", "")
    voice_id: str = os.getenv("ELEVEN_LABS_VOICE_ID", "")
    # eleven_multilingual_v2 for best multi-language support
    model: str = os.getenv("ELEVEN_LABS_MODEL", "eleven_multilingual_v2")


@dataclass
class AppConfig:
    """Application configuration."""
    livekit: LiveKitConfig
    groq: GroqConfig
    elevenlabs: ElevenLabsConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            livekit=LiveKitConfig(),
            groq=GroqConfig(),
            elevenlabs=ElevenLabsConfig(),
        )

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of missing keys."""
        missing = []
        if not self.groq.api_key:
            missing.append("GROQ_API_KEY")
        if not self.elevenlabs.api_key:
            missing.append("ELEVEN_LABS_KEY")
        if not self.elevenlabs.voice_id:
            missing.append("ELEVEN_LABS_VOICE_ID")
        return missing


config = AppConfig.from_env()
