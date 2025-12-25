"""FastAPI routes - OpenAI-compatible TTS API with WebSocket support."""

from typing import AsyncIterator

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    TTSRequest,
    TTSRequestExtended,
    VoicesResponse,
)
from src.config import settings
from src.core.voice_cache import VoiceCache
from src.inference.base import TTSBackend
from src.providers.base import TTSProvider
from src.storage.base import ReferenceStorage

router = APIRouter()


def get_provider(request: Request) -> TTSProvider:
    """Dependency to get the active TTS provider."""
    provider = request.app.state.provider
    if not provider or not provider.is_ready:
        raise HTTPException(
            status_code=503,
            detail="TTS provider not available",
        )
    return provider


def get_storage(request: Request) -> ReferenceStorage | None:
    """Dependency to get the reference storage."""
    return getattr(request.app.state, "storage", None)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    provider = request.app.state.provider

    if not provider:
        return HealthResponse(
            status="unhealthy",
            provider="none",
            models_loaded=False,
            device="unknown",
            streaming_enabled=settings.streaming_enabled,
        )

    health = await provider.health_check()

    return HealthResponse(
        status="healthy" if provider.is_ready else "degraded",
        provider=provider.name,
        models_loaded=provider.is_ready,
        device=health.get("device", "unknown"),
        streaming_enabled=settings.streaming_enabled,
        lora_active=health.get("lora_active"),
    )


@router.post("/v1/audio/speech")
async def create_speech(
    request: TTSRequest,
    provider: TTSProvider = Depends(get_provider),
):
    """OpenAI-compatible TTS endpoint.

    Generates audio from the input text using the specified voice.
    """
    if len(request.input) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.max_text_length} characters.",
        )

    logger.info(
        f"TTS request: voice={request.voice}, format={request.response_format}, "
        f"stream={request.stream}, text_len={len(request.input)}"
    )

    try:
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }
        content_type = content_types.get(request.response_format, "audio/wav")

        if request.stream:
            return StreamingResponse(
                _stream_audio(provider, request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            audio_bytes = await _generate_audio(provider, request)
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                },
            )

    except RuntimeError as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/v1/audio/speech/extended")
async def create_speech_extended(
    request: TTSRequestExtended,
    provider: TTSProvider = Depends(get_provider),
):
    """Extended TTS endpoint with additional parameters.

    Provides access to Fish Speech-specific parameters like temperature,
    top_p, repetition_penalty, etc.
    """
    if len(request.input) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.max_text_length} characters.",
        )

    logger.info(
        f"Extended TTS request: voice={request.voice}, temp={request.temperature}, "
        f"top_p={request.top_p}, stream={request.stream}"
    )

    try:
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }
        content_type = content_types.get(request.response_format, "audio/wav")

        kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
            "max_new_tokens": request.max_new_tokens,
            "chunk_length": request.chunk_length,
        }

        if request.stream:
            return StreamingResponse(
                _stream_audio(provider, request, **kwargs),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Transfer-Encoding": "chunked",
                },
            )
        else:
            audio_bytes = await _generate_audio(provider, request, **kwargs)
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                },
            )

    except RuntimeError as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/v1/voices", response_model=VoicesResponse)
async def list_voices(provider: TTSProvider = Depends(get_provider)):
    """List available voices."""
    voices = await provider.get_voices()
    return VoicesResponse(
        voices=[
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "description": v.description,
                "preview_url": v.preview_url,
                "language": v.language,
            }
            for v in voices
        ]
    )


@router.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Voice Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@router.post("/v1/references/add")
async def add_reference(
    id: str = Form(...),
    text: str = Form(...),
    audio: UploadFile = File(...),
    storage: ReferenceStorage | None = Depends(get_storage),
):
    """Add a reference audio for zero-shot cloning."""
    if not storage:
        raise HTTPException(
            status_code=501,
            detail="Reference storage not configured",
        )

    audio_bytes = await audio.read()

    try:
        ref = await storage.add_reference(
            voice_id=id,
            audio_data=audio_bytes,
            text=text,
        )
        return {
            "success": True,
            "voice_id": ref.voice_id,
            "message": f"Reference '{id}' added successfully",
        }
    except Exception as e:
        logger.error(f"Failed to add reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/references")
async def list_references(
    storage: ReferenceStorage | None = Depends(get_storage),
):
    """List available reference voices."""
    if not storage:
        raise HTTPException(
            status_code=501,
            detail="Reference storage not configured",
        )

    refs = await storage.list_references()
    return {
        "references": [
            {
                "voice_id": r.voice_id,
                "text": r.transcript,
            }
            for r in refs
        ]
    }


@router.delete("/v1/references/{voice_id}")
async def delete_reference(
    voice_id: str,
    storage: ReferenceStorage | None = Depends(get_storage),
):
    """Delete a reference voice."""
    if not storage:
        raise HTTPException(
            status_code=501,
            detail="Reference storage not configured",
        )

    success = await storage.delete_reference(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Reference '{voice_id}' not found")

    return {"success": True, "message": f"Reference '{voice_id}' deleted"}


@router.websocket("/v1/audio/stream")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming.

    Protocol:
        Send: {"text": "...", "voice": "voice_id"}
        Receive: Binary audio chunks (PCM16), then {"type": "done"}
    """
    await websocket.accept()
    provider = websocket.app.state.provider

    if not provider or not provider.is_ready:
        await websocket.close(code=1003, reason="Provider not available")
        return

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice = data.get("voice")

            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            try:
                async for chunk in provider.synthesize(
                    text=text,
                    voice_id=voice if voice != "default" else None,
                    format="pcm",
                    streaming=True,
                ):
                    await websocket.send_bytes(chunk)

                await websocket.send_json({"type": "done"})
            except Exception as e:
                logger.error(f"WebSocket TTS error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")


@router.post("/v1/voices/encode")
async def encode_voice(
    voice_id: str = Form(...),
    transcript: str = Form(...),
    audio: UploadFile = File(...),
    request: Request = None,
):
    """Encode reference audio to pre-cached voice tokens.

    Creates a .npy file that can be used for fast inference without
    re-encoding the reference audio each time.
    """
    provider = request.app.state.provider
    if not provider or not provider.is_ready:
        raise HTTPException(status_code=503, detail="Provider not available")

    backend: TTSBackend | None = getattr(provider, "backend", None)
    if not backend:
        raise HTTPException(
            status_code=501,
            detail="Backend does not support voice encoding",
        )

    voice_cache: VoiceCache | None = getattr(provider, "_voice_cache", None)
    if not voice_cache:
        raise HTTPException(
            status_code=501,
            detail="Voice cache not configured",
        )

    audio_bytes = await audio.read()

    try:
        import asyncio

        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(
            None, lambda: backend.encode_reference(audio_bytes)
        )

        voice_cache.add(voice_id, tokens, transcript)

        return {
            "voice_id": voice_id,
            "tokens_shape": list(tokens.shape),
            "message": f"Voice '{voice_id}' encoded and cached",
        }
    except Exception as e:
        logger.error(f"Voice encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_audio(
    provider: TTSProvider,
    request: TTSRequest,
    **kwargs,
) -> bytes:
    """Generate complete audio from request."""
    voice_id = request.voice if request.voice != "default" else None

    audio_chunks = []
    async for chunk in provider.synthesize(
        text=request.input,
        voice_id=voice_id,
        format=request.response_format,
        streaming=False,
        **kwargs,
    ):
        audio_chunks.append(chunk)

    return b"".join(audio_chunks)


async def _stream_audio(
    provider: TTSProvider,
    request: TTSRequest,
    **kwargs,
) -> AsyncIterator[bytes]:
    """Stream audio chunks as they're generated."""
    voice_id = request.voice if request.voice != "default" else None

    async for chunk in provider.synthesize(
        text=request.input,
        voice_id=voice_id,
        format=request.response_format,
        streaming=True,
        **kwargs,
    ):
        yield chunk

