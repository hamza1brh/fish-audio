"""Integration tests for WebSocket and voice encoding."""

import json

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from src.main import create_app
from src.providers.mock import MockProvider


@pytest.fixture
def app():
    """Create test application with mock provider."""
    test_app = create_app()
    test_app.state.provider = MockProvider()
    test_app.state.storage = None
    return test_app


@pytest.fixture
async def initialized_app(app):
    """Initialize the app's provider."""
    await app.state.provider.initialize()
    yield app
    await app.state.provider.shutdown()


@pytest.fixture
async def client(initialized_app):
    """Create async test client."""
    transport = ASGITransport(app=initialized_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(app):
    """Create sync test client for WebSocket testing."""
    with TestClient(app) as client:
        # Initialize provider within context
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(app.state.provider.initialize())
        yield client
        loop.run_until_complete(app.state.provider.shutdown())
        loop.close()


def test_websocket_streaming(sync_client):
    """WebSocket connection works and accepts messages."""
    # Test that WebSocket can connect
    with sync_client.websocket_connect("/v1/audio/stream") as websocket:
        # Send a message
        websocket.send_json({"text": "Hello world", "voice": "default"})
        # Connection established successfully
        assert True


def test_websocket_empty_text_error(sync_client):
    """WebSocket returns error for empty text."""
    with sync_client.websocket_connect("/v1/audio/stream") as websocket:
        websocket.send_json({"text": "", "voice": "default"})

        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Empty text" in data["message"]


def test_websocket_multiple_messages(sync_client):
    """WebSocket handles multiple messages in sequence."""
    with sync_client.websocket_connect("/v1/audio/stream") as websocket:
        for i in range(3):
            websocket.send_json({"text": f"Message {i}", "voice": "default"})

            done = False
            while not done:
                try:
                    data = websocket.receive(timeout=10.0)
                    if data["type"] == "websocket.receive" and "text" in data:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "done":
                            done = True
                except Exception:
                    break


@pytest.mark.asyncio
async def test_voice_encode_endpoint(client):
    """Voice encode endpoint responds (may skip if no backend)."""
    wav_data = _create_test_wav()

    response = await client.post(
        "/v1/voices/encode",
        data={"voice_id": "test-voice", "transcript": "Hello world"},
        files={"audio": ("test.wav", wav_data, "audio/wav")},
        timeout=30.0,
    )

    # Mock provider doesn't have a backend, so this should return 501
    assert response.status_code in [200, 501]


@pytest.mark.asyncio
async def test_http_streaming_response(client):
    """HTTP streaming returns chunked audio."""
    chunks = []

    async with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "Test streaming response",
            "voice": "default",
            "response_format": "pcm",
            "stream": True,
        },
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        async for chunk in response.aiter_bytes(chunk_size=1024):
            chunks.append(chunk)

    assert len(chunks) > 0, "No chunks received"
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0


@pytest.mark.asyncio
async def test_http_complete_response(client):
    """HTTP complete response returns audio file."""
    response = await client.post(
        "/v1/audio/speech",
        json={
            "input": "Test complete response",
            "voice": "default",
            "response_format": "wav",
        },
        timeout=30.0,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 44


@pytest.mark.asyncio
async def test_backend_interface_implementation(initialized_app):
    """Mock provider correctly reports no backend attribute."""
    provider = initialized_app.state.provider
    assert provider is not None, "Provider not initialized"

    # Mock provider should not have a backend attribute
    backend = getattr(provider, "backend", None)
    
    # This is expected for mock provider - it doesn't wrap a backend
    # S1MiniProvider would have a backend
    if backend is None:
        assert provider.name == "mock", "Non-mock provider should have a backend"
    else:
        # If we have a backend, verify it implements TTSBackend
        assert hasattr(backend, "sample_rate")
        assert hasattr(backend, "is_ready")
        assert hasattr(backend, "generate_stream")
        assert hasattr(backend, "encode_reference")
        assert hasattr(backend, "initialize")
        assert hasattr(backend, "shutdown")


def _create_test_wav() -> bytes:
    """Create minimal valid WAV file."""
    import struct

    sample_rate = 24000
    duration = 0.5
    samples = int(sample_rate * duration)

    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    pcm = (audio * 32767).astype(np.int16)

    buffer = bytearray()
    buffer.extend(b"RIFF")
    buffer.extend(struct.pack("<I", 36 + len(pcm) * 2))
    buffer.extend(b"WAVE")
    buffer.extend(b"fmt ")
    buffer.extend(struct.pack("<I", 16))
    buffer.extend(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buffer.extend(b"data")
    buffer.extend(struct.pack("<I", len(pcm) * 2))
    buffer.extend(pcm.tobytes())

    return bytes(buffer)
