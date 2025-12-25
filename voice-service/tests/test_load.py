"""Load and concurrency tests for voice service."""

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

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


@pytest.mark.asyncio
async def test_concurrent_requests_no_deadlock(client):
    """10 concurrent requests don't deadlock."""

    async def make_request(i: int):
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": f"Test message number {i}",
                "voice": "default",
                "response_format": "pcm",
            },
            timeout=30.0,
        )
        return response.status_code

    tasks = [make_request(i) for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if r == 200)
    assert success_count >= 5, f"Too many failures: {results}"


@pytest.mark.asyncio
async def test_request_isolation(client):
    """Request A's error doesn't affect Request B."""

    async def good_request():
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Valid text",
                "voice": "default",
                "response_format": "pcm",
            },
            timeout=30.0,
        )
        return response.status_code

    async def bad_request():
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "",
                "voice": "default",
                "response_format": "pcm",
            },
            timeout=30.0,
        )
        return response.status_code

    tasks = [good_request(), bad_request(), good_request()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    good_results = [results[0], results[2]]
    for r in good_results:
        assert r == 200 or isinstance(r, Exception), f"Unexpected result: {r}"


@pytest.mark.asyncio
async def test_streaming_ttfb(client):
    """Time to first byte under 5 seconds (mock provider)."""
    start = time.perf_counter()
    first_chunk_time = None

    async with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "input": "Hello world",
            "voice": "default",
            "response_format": "pcm",
            "stream": True,
        },
        timeout=30.0,
    ) as response:
        async for chunk in response.aiter_bytes():
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            break

    assert first_chunk_time is not None, "No chunks received"
    assert first_chunk_time < 5.0, f"TTFB too slow: {first_chunk_time:.2f}s"


@pytest.mark.asyncio
async def test_sequential_requests_performance(client):
    """Sequential requests complete in reasonable time."""
    start = time.perf_counter()
    count = 5

    for i in range(count):
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": f"Test message {i}",
                "voice": "default",
                "response_format": "pcm",
            },
            timeout=30.0,
        )
        assert response.status_code == 200

    elapsed = time.perf_counter() - start
    avg_time = elapsed / count
    assert avg_time < 3.0, f"Average request time too slow: {avg_time:.2f}s"


@pytest.mark.asyncio
async def test_health_under_load(client):
    """Health endpoint responds during load."""

    async def generate_load():
        for _ in range(5):
            await client.post(
                "/v1/audio/speech",
                json={"input": "Load test", "voice": "default"},
                timeout=30.0,
            )
            await asyncio.sleep(0.1)

    async def check_health():
        for _ in range(10):
            response = await client.get("/health", timeout=5.0)
            assert response.status_code == 200
            await asyncio.sleep(0.2)

    await asyncio.gather(generate_load(), check_health())
