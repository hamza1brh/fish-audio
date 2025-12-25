"""Request batcher tests. Run with: pytest --gpu"""

import asyncio

import numpy as np
import pytest

from src.inference.batcher import BatchRequest, RequestBatcher


@pytest.mark.gpu
class TestBatcherWithGPU:
    """Batcher tests requiring GPU."""

    @pytest.fixture
    def batcher(self, engine):
        """Create batcher with real engine."""
        return RequestBatcher(engine, max_batch_size=2, max_wait_ms=50)

    def test_single_request(self, batcher):
        """Single request through batcher."""

        async def run():
            await batcher.start()
            try:
                request = BatchRequest(text="Single request test.", max_new_tokens=256)
                audio = await batcher.submit(request)
                assert isinstance(audio, np.ndarray)
                assert len(audio) > 500
            finally:
                await batcher.stop()

        asyncio.run(run())

    def test_concurrent_requests(self, batcher):
        """Multiple concurrent requests get batched."""

        async def run():
            await batcher.start()
            try:
                requests = [
                    BatchRequest(text=f"Concurrent request {i}.", max_new_tokens=256)
                    for i in range(3)
                ]
                results = await asyncio.gather(*[batcher.submit(r) for r in requests])
                assert len(results) == 3
                for audio in results:
                    assert isinstance(audio, np.ndarray)
                    assert len(audio) > 500
            finally:
                await batcher.stop()

        asyncio.run(run())

    def test_batcher_stats(self, batcher):
        """Batcher reports stats."""

        async def run():
            await batcher.start()
            try:
                stats = batcher.stats
                assert "pending" in stats
                assert "max_batch_size" in stats
                assert stats["running"] is True
            finally:
                await batcher.stop()

        asyncio.run(run())


class TestBatcherUnit:
    """Unit tests without GPU."""

    def test_batch_request_defaults(self):
        """BatchRequest has sensible defaults."""
        req = BatchRequest(text="Test")
        assert req.temperature == 0.7
        assert req.top_p == 0.8
        assert req.max_new_tokens == 1024

    def test_batcher_not_started(self):
        """Batcher stats when not running."""

        class FakeEngine:
            pass

        batcher = RequestBatcher(FakeEngine(), max_batch_size=4)
        assert batcher.stats["running"] is False
        assert batcher.pending_count == 0

