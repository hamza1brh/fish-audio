"""
S1-Mini Production Server
=========================

FastAPI-based production server for S1-Mini TTS inference.

Features:
---------
1. REST API for TTS generation
2. Streaming support for real-time audio
3. Health check endpoints for load balancers
4. Metrics endpoint for monitoring
5. Request timeout and cancellation
6. Graceful shutdown handling

Endpoints:
----------
    POST /v1/tts              - Generate TTS audio
    POST /v1/tts/stream       - Streaming TTS audio
    GET  /health              - Health check
    GET  /ready               - Readiness check
    GET  /metrics             - Prometheus metrics

Usage:
------
    # Start server programmatically
    from s1_mini.server import create_app, run_server
    from s1_mini import EngineConfig

    config = EngineConfig(checkpoint_path="checkpoints/openaudio-s1-mini")
    app = create_app(config)
    run_server(app, host="0.0.0.0", port=8080)

    # Or use CLI
    python -m s1_mini.server --checkpoint checkpoints/openaudio-s1-mini

Environment Variables:
----------------------
    S1_MINI_SERVER_HOST       Server bind address (default: 0.0.0.0)
    S1_MINI_SERVER_PORT       Server port (default: 8080)
    S1_MINI_CHECKPOINT_PATH   Model checkpoint path
    S1_MINI_DEVICE            Device (cuda, cpu)
    S1_MINI_API_KEY           Optional API key for authentication

Deployment Notes:
-----------------
- IMPORTANT: Use workers=1 with GPU inference
- Each worker loads its own model copy into VRAM
- For scaling, use multiple GPU instances behind a load balancer
"""

import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Import S1-Mini components
from s1_mini.config import EngineConfig, ServerConfig
from s1_mini.engine import ProductionTTSEngine
from s1_mini.exceptions import S1MiniError, TimeoutError


# =============================================================================
# Request/Response Models
# =============================================================================


class TTSRequest(BaseModel):
    """
    TTS generation request.

    Attributes:
        text: Text to synthesize (required)
        reference_audio: Base64-encoded audio for zero-shot voice cloning
        reference_text: Text spoken in reference audio (required with reference_audio)
        reference_id: Pre-registered reference voice ID (alternative to reference_audio)
        max_new_tokens: Maximum tokens to generate (default: 2048)
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        top_p: Nucleus sampling threshold 0.0-1.0 (default: 0.8)
        repetition_penalty: Repetition penalty 0.0-2.0 (default: 1.1)
        chunk_length: Chunk length for iterative generation
        seed: Random seed for reproducibility
        format: Output format (wav, mp3)
    """

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    reference_audio: Optional[str] = Field(None, description="Base64-encoded audio for zero-shot cloning")
    reference_text: Optional[str] = Field(None, description="Text spoken in reference audio")
    reference_id: Optional[str] = Field(None, description="Pre-registered reference ID")
    max_new_tokens: int = Field(2048, ge=1, le=4096, description="Max tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(1.1, ge=0.0, le=2.0, description="Repetition penalty")
    chunk_length: int = Field(200, ge=50, le=500, description="Chunk length")
    seed: Optional[int] = Field(None, description="Random seed")
    format: str = Field("wav", description="Output format (wav)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test of the text to speech system.",
                "reference_text": "This is the text spoken in the reference audio.",
                "temperature": 0.7,
                "top_p": 0.8,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    engine_ready: bool
    models_loaded: bool
    device: str
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None


class BatchTTSRequest(BaseModel):
    """
    Batch TTS generation request.

    Attributes:
        items: List of TTS requests to process
        max_new_tokens: Shared max tokens (can be overridden per item)
        temperature: Shared temperature
        top_p: Shared top-p sampling
        repetition_penalty: Shared repetition penalty
        chunk_length: Shared chunk length
    """
    items: List[TTSRequest] = Field(..., min_length=1, max_length=8, description="List of TTS requests")
    max_new_tokens: int = Field(2048, ge=1, le=4096, description="Max tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(1.1, ge=0.0, le=2.0, description="Repetition penalty")
    chunk_length: int = Field(200, ge=50, le=500, description="Chunk length")
    seed: Optional[int] = Field(None, description="Random seed")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {"text": "Hello, this is the first message."},
                    {"text": "And this is the second message."},
                ],
                "temperature": 0.7,
                "top_p": 0.8,
            }
        }


class BatchTTSResponseItem(BaseModel):
    """Single item in batch response."""
    success: bool
    audio_base64: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None


class BatchTTSResponse(BaseModel):
    """Batch TTS generation response."""
    success: bool
    results: List[BatchTTSResponseItem]
    total_time: float
    error: Optional[str] = None


# =============================================================================
# Global State
# =============================================================================


class AppState:
    """
    Application state container.

    Holds references to the TTS engine and configuration.
    """

    def __init__(self):
        self.engine: Optional[ProductionTTSEngine] = None
        self.config: Optional[EngineConfig] = None
        self.server_config: Optional[ServerConfig] = None
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.error_count: int = 0


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.

    Handles startup and shutdown of the TTS engine.
    """
    logger.info("Starting S1-Mini server...")

    # Initialize engine
    if app_state.config is None:
        # Load from environment
        app_state.config = EngineConfig.from_env()

    app_state.engine = ProductionTTSEngine(app_state.config)

    try:
        # Start engine (loads models, warmup)
        app_state.engine.start()
        logger.info("S1-Mini server ready!")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down S1-Mini server...")
        if app_state.engine:
            app_state.engine.stop()
        logger.info("S1-Mini server stopped")


# =============================================================================
# FastAPI App Factory
# =============================================================================


def create_app(
    engine_config: Optional[EngineConfig] = None,
    server_config: Optional[ServerConfig] = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        engine_config: Engine configuration
        server_config: Server configuration

    Returns:
        Configured FastAPI app
    """
    # Store config in app state
    app_state.config = engine_config
    app_state.server_config = server_config or ServerConfig()

    # Create app
    app = FastAPI(
        title="S1-Mini TTS API",
        description="Production-ready Text-to-Speech API using Fish-Speech S1-Mini",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_state.server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register API routes."""

    # =========================================================================
    # Health Endpoints
    # =========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns basic health status. Use for load balancer health checks.
        """
        if app_state.engine is None:
            return HealthResponse(
                status="unhealthy",
                engine_ready=False,
                models_loaded=False,
                device="unknown",
                vram_used_gb=None,
                vram_total_gb=None,
            )

        health = app_state.engine.health_check()

        return HealthResponse(
            status="healthy" if health["healthy"] else "unhealthy",
            engine_ready=health["healthy"],
            models_loaded=health["models_loaded"],
            device=health["device"],
            vram_used_gb=health["vram"]["used_gb"],
            vram_total_gb=health["vram"]["total_gb"],
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check endpoint.

        Returns 200 if ready to serve requests, 503 otherwise.
        """
        if app_state.engine is None or not app_state.engine.is_running:
            raise HTTPException(status_code=503, detail="Engine not ready")

        return {"status": "ready"}

    @app.get("/metrics", tags=["Health"])
    async def metrics():
        """
        Metrics endpoint.

        Returns Prometheus-compatible metrics.
        """
        if app_state.engine is None:
            return {}

        engine_metrics = app_state.engine.get_metrics()
        uptime = time.time() - app_state.start_time

        return {
            "uptime_seconds": uptime,
            "request_count": app_state.request_count,
            "error_count": app_state.error_count,
            "error_rate": app_state.error_count / max(app_state.request_count, 1),
            **engine_metrics,
        }

    # =========================================================================
    # TTS Endpoints
    # =========================================================================

    @app.post("/v1/tts", tags=["TTS"])
    async def generate_tts(request: TTSRequest):
        """
        Generate TTS audio.

        Returns complete audio as WAV file.

        Request Body:
            text: Text to synthesize
            reference_audio: Base64-encoded audio for zero-shot cloning (optional)
            reference_text: Text spoken in reference audio (required with reference_audio)
            reference_id: Pre-registered reference ID (alternative to reference_audio)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            ...

        Returns:
            Audio file (audio/wav)
        """
        if app_state.engine is None or not app_state.engine.is_running:
            raise HTTPException(status_code=503, detail="Engine not ready")

        app_state.request_count += 1

        try:
            # Handle reference audio for zero-shot cloning
            reference_audio_bytes = None
            if request.reference_audio is not None:
                # Validate reference_text is provided
                if request.reference_text is None:
                    raise HTTPException(
                        status_code=400,
                        detail="reference_text is required when using reference_audio"
                    )
                # Decode base64 audio
                try:
                    reference_audio_bytes = base64.b64decode(request.reference_audio)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid base64 encoding for reference_audio: {e}"
                    )

            # Generate audio
            response = app_state.engine.generate(
                text=request.text,
                reference_audio=reference_audio_bytes,
                reference_text=request.reference_text,
                reference_id=request.reference_id,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                chunk_length=request.chunk_length,
                seed=request.seed,
                return_bytes=True,
            )

            if not response.success:
                app_state.error_count += 1
                raise HTTPException(status_code=500, detail=response.error)

            # Return audio
            return Response(
                content=response.audio_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "X-Generation-Time": str(response.metrics.total_time_seconds)
                    if response.metrics
                    else "0",
                },
            )

        except HTTPException:
            raise
        except ValueError as e:
            app_state.error_count += 1
            raise HTTPException(status_code=400, detail=str(e))
        except TimeoutError as e:
            app_state.error_count += 1
            raise HTTPException(status_code=504, detail=str(e))
        except S1MiniError as e:
            app_state.error_count += 1
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            app_state.error_count += 1
            logger.error(f"TTS generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/tts/stream", tags=["TTS"])
    async def generate_tts_stream(request: TTSRequest):
        """
        Generate TTS audio with streaming.

        Returns audio as chunked stream for real-time playback.
        Supports zero-shot voice cloning via reference_audio + reference_text.
        """
        if app_state.engine is None or not app_state.engine.is_running:
            raise HTTPException(status_code=503, detail="Engine not ready")

        app_state.request_count += 1

        # Handle reference audio for zero-shot cloning
        reference_audio_bytes = None
        if request.reference_audio is not None:
            if request.reference_text is None:
                raise HTTPException(
                    status_code=400,
                    detail="reference_text is required when using reference_audio"
                )
            try:
                reference_audio_bytes = base64.b64decode(request.reference_audio)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 encoding for reference_audio: {e}"
                )

        async def audio_generator():
            """Async generator for streaming audio chunks."""
            try:
                for result in app_state.engine.generate_stream(
                    text=request.text,
                    reference_audio=reference_audio_bytes,
                    reference_text=request.reference_text,
                    reference_id=request.reference_id,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    chunk_length=request.chunk_length,
                    seed=request.seed,
                ):
                    if result.code == "error":
                        app_state.error_count += 1
                        logger.error(f"Streaming error: {result.error}")
                        break

                    if result.code in ("header", "segment"):
                        if result.audio is not None:
                            sample_rate, audio_data = result.audio
                            # Convert to bytes
                            audio_bytes = (audio_data * 32767).astype("int16").tobytes()
                            yield audio_bytes

                    # Small delay to prevent blocking
                    await asyncio.sleep(0)

            except Exception as e:
                app_state.error_count += 1
                logger.error(f"Streaming error: {e}")

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={"Transfer-Encoding": "chunked"},
        )

    @app.post("/v1/tts/batch", response_model=BatchTTSResponse, tags=["TTS"])
    async def generate_tts_batch(request: BatchTTSRequest):
        """
        Generate TTS audio for multiple texts in a batch.

        This endpoint processes multiple TTS requests together, which can
        be more efficient for bulk processing.

        Request Body:
            items: List of TTS requests (text + optional references)
            max_new_tokens: Shared maximum tokens
            temperature: Shared sampling temperature
            top_p: Shared nucleus sampling threshold
            ...

        Returns:
            BatchTTSResponse with results for each item
        """
        if app_state.engine is None or not app_state.engine.is_running:
            raise HTTPException(status_code=503, detail="Engine not ready")

        app_state.request_count += len(request.items)

        try:
            # Extract texts and reference info from items
            texts = []
            reference_audios = []
            reference_texts = []
            reference_ids = []

            for item in request.items:
                texts.append(item.text)

                # Decode reference audio if provided
                ref_audio = None
                if item.reference_audio is not None:
                    if item.reference_text is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"reference_text is required when using reference_audio for text: {item.text[:50]}..."
                        )
                    try:
                        ref_audio = base64.b64decode(item.reference_audio)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid base64 encoding for reference_audio: {e}"
                        )

                reference_audios.append(ref_audio)
                reference_texts.append(item.reference_text)
                reference_ids.append(item.reference_id)

            # Generate batch
            batch_response = app_state.engine.generate_batch(
                texts=texts,
                reference_audios=reference_audios,
                reference_texts=reference_texts,
                reference_ids=reference_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                chunk_length=request.chunk_length,
                seed=request.seed,
                return_bytes=True,
            )

            # Convert results to response format
            results = []
            for gen_response in batch_response.results:
                if gen_response.success and gen_response.audio_bytes:
                    results.append(BatchTTSResponseItem(
                        success=True,
                        audio_base64=base64.b64encode(gen_response.audio_bytes).decode(),
                        generation_time=gen_response.metrics.total_time_seconds if gen_response.metrics else None,
                    ))
                else:
                    app_state.error_count += 1
                    results.append(BatchTTSResponseItem(
                        success=False,
                        error=gen_response.error,
                    ))

            return BatchTTSResponse(
                success=batch_response.success,
                results=results,
                total_time=batch_response.total_time,
                error=batch_response.error,
            )

        except HTTPException:
            raise
        except Exception as e:
            app_state.error_count += len(request.items)
            logger.error(f"Batch TTS generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Info Endpoints
    # =========================================================================

    @app.get("/", tags=["Info"])
    async def root():
        """API information."""
        return {
            "name": "S1-Mini TTS API",
            "version": "1.0.0",
            "status": "running" if app_state.engine and app_state.engine.is_running else "starting",
            "docs": "/docs",
        }

    @app.get("/v1/info", tags=["Info"])
    async def info():
        """Detailed API information."""
        if app_state.engine is None:
            return {"status": "starting"}

        health = app_state.engine.health_check()

        return {
            "status": "ready" if health["healthy"] else "not_ready",
            "device": health["device"],
            "precision": health["precision"],
            "compile_enabled": health["compile_enabled"],
            "compile_backend": health["compile_backend"],
            "vram": health["vram"],
        }


# =============================================================================
# Server Runner
# =============================================================================


def run_server(
    app: Optional[FastAPI] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    workers: int = 1,
):
    """
    Run the production server.

    Args:
        app: FastAPI app (creates default if None)
        host: Bind address
        port: Port number
        reload: Enable auto-reload (development only)
        workers: Number of workers (MUST be 1 for GPU)
    """
    import uvicorn

    if app is None:
        app = create_app()

    if workers > 1:
        logger.warning(
            "Using workers > 1 with GPU inference will load multiple model copies. "
            "This may cause OOM errors. Use workers=1 unless you have multiple GPUs."
        )

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        access_log=True,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="S1-Mini TTS Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/openaudio-s1-mini",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development)",
    )

    args = parser.parse_args()

    # Create config
    config = EngineConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
        compile_model=not args.no_compile,
    )

    # Create and run app
    app = create_app(config)
    run_server(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
