"""API request/response schemas - OpenAI compatible."""

from typing import Literal

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request schema."""

    model: str = Field(default="s1-mini", description="TTS model to use")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: str = Field(default="default", description="Voice ID for synthesis")
    response_format: Literal["wav", "mp3", "opus", "flac", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed (unused)")
    stream: bool = Field(default=False, description="Enable streaming response")


class TTSRequestExtended(TTSRequest):
    """Extended TTS request with additional Fish Speech parameters."""

    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)
    max_new_tokens: int = Field(default=1024, ge=100, le=4096)
    chunk_length: int = Field(default=200, ge=100, le=300)
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    provider: str
    models_loaded: bool
    device: str
    streaming_enabled: bool
    lora_active: str | None = None


class VoiceInfo(BaseModel):
    """Voice information schema."""

    voice_id: str
    name: str
    description: str | None = None
    preview_url: str | None = None
    language: str | None = None


class VoicesResponse(BaseModel):
    """List voices response."""

    voices: list[VoiceInfo]


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str | None = None
    code: str | None = None











