"""Voice Service application entry point."""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.config import settings
from src.core.voice_cache import VoiceCache
from src.providers.base import TTSProvider
from src.providers.elevenlabs import ElevenLabsProvider
from src.providers.mock import MockProvider
from src.providers.registry import register_provider
from src.storage.huggingface import HuggingFaceStorage
from src.storage.local import LocalStorage
from src.storage.s3 import S3Storage

logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level.upper(),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)


def get_storage():
    """Create storage backend based on configuration."""
    if settings.reference_storage == "huggingface":
        return HuggingFaceStorage(
            repo_id=settings.hf_reference_repo,
            token=settings.hf_token,
        )
    elif settings.reference_storage == "s3":
        return S3Storage(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.aws_region,
        )
    else:
        return LocalStorage(settings.local_reference_path)


async def create_provider() -> TTSProvider:
    """Create TTS provider based on configuration."""
    provider_name = settings.tts_provider

    if provider_name == "s1_mini":
        try:
            settings.validate_s1_mini_config()
        except ValueError as e:
            logger.error(f"S1 Mini config error: {e}")
            logger.warning("Falling back to mock provider")
            provider = MockProvider()
            await provider.initialize()
            return provider

        try:
            from src.inference.engine import S1MiniBackend
            from src.providers.s1_mini import S1MiniProvider

            storage = get_storage()
            voice_cache = VoiceCache(settings.voices_path)

            backend = S1MiniBackend(
                checkpoint_path=settings.s1_checkpoint_path,
                codec_path=settings.codec_path,
                device=settings.s1_device,
                compile=settings.s1_compile,
                half=settings.s1_half,
            )

            provider = S1MiniProvider(
                backend=backend,
                voice_cache=voice_cache,
                storage=storage,
            )
        except ImportError as e:
            logger.warning(f"S1 Mini dependencies not available: {e}")
            logger.warning("Falling back to mock provider")
            provider = MockProvider()

    elif provider_name == "elevenlabs":
        if not settings.elevenlabs_api_key:
            logger.error("VOICE_ELEVENLABS_API_KEY not set")
            logger.warning("Falling back to mock provider")
            provider = MockProvider()
        else:
            provider = ElevenLabsProvider(
                api_key=settings.elevenlabs_api_key,
                model=settings.elevenlabs_model,
            )

    else:
        provider = MockProvider()

    await provider.initialize()
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("Voice Service starting up")
    logger.info(f"  Provider: {settings.tts_provider}")
    logger.info(f"  Storage: {settings.reference_storage}")
    logger.info(f"  Voices: {settings.voices_path}")
    logger.info(f"  Streaming: {settings.streaming_enabled}")
    logger.info("=" * 60)

    register_provider("mock", MockProvider)
    register_provider("elevenlabs", ElevenLabsProvider)

    try:
        from src.providers.s1_mini import S1MiniProvider

        register_provider("s1_mini", S1MiniProvider)
    except ImportError:
        logger.warning("S1 Mini provider not available (missing dependencies)")

    app.state.storage = get_storage()
    app.state.provider = await create_provider()

    # Warmup backend if available
    backend = getattr(app.state.provider, "backend", None)
    if backend and hasattr(backend, "warmup"):
        logger.info("Running backend warmup...")
        try:
            import asyncio

            await asyncio.get_event_loop().run_in_executor(None, backend.warmup)
            logger.info("Backend warmup complete")
        except Exception as e:
            logger.warning(f"Backend warmup failed: {e}")

    logger.info("Voice Service ready")
    yield

    logger.info("Voice Service shutting down")
    if app.state.provider:
        await app.state.provider.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Voice Service",
        description="Production TTS service with OpenAI-compatible API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=False,
    )
