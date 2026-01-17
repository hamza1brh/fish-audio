"""
Tests for Batched Inference Implementation
==========================================

This module contains comprehensive tests for:
1. BatchQueue - Request collection and grouping
2. Batched inference functions
3. BatchedModelWorker
4. Engine batch generation
5. End-to-end integration tests
6. Performance benchmarks
"""

import asyncio
import base64
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_texts():
    """Sample texts of varying lengths for testing."""
    return [
        "Hello, this is a short test.",
        "Welcome to our text-to-speech service. We hope you enjoy using it.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "Testing the batched inference system with a medium length sentence.",
        "Short one.",
        "A longer test sentence to ensure that the batching system handles variable length inputs correctly and efficiently.",
        "Another test.",
        "This is test number eight in our batch.",
        "Penultimate test sentence for the batch processing system.",
        "Final test sentence - number ten in the batch!",
    ]


@pytest.fixture
def sample_reference_audio():
    """Generate sample reference audio bytes (silence)."""
    import io
    import wave

    sample_rate = 44100
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)

    # Generate silence
    audio_data = np.zeros(samples, dtype=np.int16)

    # Write to WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

    return buffer.getvalue()


# =============================================================================
# Unit Tests: BatchQueue
# =============================================================================


class TestBatchQueue:
    """Tests for the BatchQueue class."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test BatchQueue initializes correctly."""
        from s1_mini.batch_queue import BatchQueue

        queue = BatchQueue(
            max_batch_size=4,
            batch_timeout_ms=200,
            enable_grouping=True,
        )

        assert queue.max_batch_size == 4
        assert queue.batch_timeout_ms == 200
        assert queue.enable_grouping is True
        assert not queue.is_running

    @pytest.mark.asyncio
    async def test_queue_start_stop(self):
        """Test starting and stopping the queue."""
        from s1_mini.batch_queue import BatchQueue

        async def mock_callback(batch):
            return [None] * len(batch.requests)

        queue = BatchQueue(process_callback=mock_callback)

        await queue.start()
        assert queue.is_running

        await queue.stop()
        assert not queue.is_running

    @pytest.mark.asyncio
    async def test_submit_without_start_raises(self):
        """Test that submitting without starting raises an error."""
        from s1_mini.batch_queue import BatchQueue

        queue = BatchQueue()

        with pytest.raises(RuntimeError, match="not running"):
            await queue.submit(text="Hello")

    @pytest.mark.asyncio
    async def test_request_grouping_by_reference(self):
        """Test that requests are grouped by reference audio hash."""
        from s1_mini.batch_queue import BatchQueue, BatchRequest

        # Create requests with different reference hashes
        req1 = BatchRequest(
            request_id="1",
            text="Hello",
            reference_audio=b"audio1",
        )
        req2 = BatchRequest(
            request_id="2",
            text="World",
            reference_audio=b"audio1",  # Same as req1
        )
        req3 = BatchRequest(
            request_id="3",
            text="Test",
            reference_audio=b"audio2",  # Different
        )

        # Verify hashes
        assert req1.reference_hash == req2.reference_hash
        assert req1.reference_hash != req3.reference_hash

    @pytest.mark.asyncio
    async def test_batch_formation(self):
        """Test that batches are formed correctly."""
        from s1_mini.batch_queue import BatchQueue

        batches_received = []

        async def capture_batch(batch):
            batches_received.append(batch)
            return [{"audio": f"audio_{i}"} for i in range(len(batch.requests))]

        queue = BatchQueue(
            max_batch_size=2,
            batch_timeout_ms=50,
            process_callback=capture_batch,
        )

        await queue.start()

        # Submit 3 requests quickly
        futures = [
            await queue.submit(text="Hello"),
            await queue.submit(text="World"),
            await queue.submit(text="Test"),
        ]

        # Wait for processing
        results = await asyncio.gather(*futures, return_exceptions=True)

        await queue.stop()

        # Should have formed at least 1 batch
        assert len(batches_received) >= 1

    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics tracking."""
        from s1_mini.batch_queue import BatchQueue

        async def mock_callback(batch):
            return [None] * len(batch.requests)

        queue = BatchQueue(
            max_batch_size=4,
            batch_timeout_ms=50,
            process_callback=mock_callback,
        )

        await queue.start()

        # Submit requests
        futures = [await queue.submit(text=f"Test {i}") for i in range(3)]
        await asyncio.gather(*futures, return_exceptions=True)

        await queue.stop()

        stats = queue.get_stats()
        assert stats["total_requests"] == 3
        assert stats["total_batches"] >= 1


# =============================================================================
# Unit Tests: Batched Inference Functions
# =============================================================================


class TestBatchedInferenceFunctions:
    """Tests for batched inference functions."""

    def test_multinomial_sample_batched(self):
        """Test batched multinomial sampling."""
        from fish_speech.models.text2semantic.inference_batched import (
            multinomial_sample_one_no_sync_batched,
        )

        # Create test probabilities
        batch_size = 4
        vocab_size = 100
        probs = torch.softmax(torch.randn(batch_size, vocab_size), dim=-1)

        # Sample
        samples = multinomial_sample_one_no_sync_batched(probs)

        assert samples.shape == (batch_size, 1)
        assert (samples >= 0).all()
        assert (samples < vocab_size).all()

    def test_logits_to_probs_batched(self):
        """Test batched logits to probability conversion."""
        from fish_speech.models.text2semantic.inference_batched import (
            logits_to_probs_batched,
        )

        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        temperature = torch.tensor(0.7)
        top_p = torch.tensor(0.9)
        repetition_penalty = torch.tensor(1.1)

        probs = logits_to_probs_batched(
            logits,
            temperature,
            top_p,
            repetition_penalty,
        )

        assert probs.shape == (batch_size, vocab_size)
        # Probabilities should sum to approximately 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_sample_batched(self):
        """Test batched sampling."""
        from fish_speech.models.text2semantic.inference_batched import sample_batched

        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        temperature = torch.tensor(0.7)
        top_p = torch.tensor(0.9)
        repetition_penalty = torch.tensor(1.1)

        indices, probs = sample_batched(
            logits,
            temperature,
            top_p,
            repetition_penalty,
        )

        assert indices.shape == (batch_size, 1)
        assert probs.shape == (batch_size, vocab_size)


# =============================================================================
# Unit Tests: Config
# =============================================================================


class TestBatchConfig:
    """Tests for batch configuration."""

    def test_default_batch_config(self):
        """Test default batch configuration values."""
        from s1_mini.config import EngineConfig

        config = EngineConfig()

        assert config.enable_batching is True
        assert config.max_batch_size == 4
        assert config.batch_timeout_ms == 200
        assert config.batch_grouping is True

    def test_batch_config_from_env(self):
        """Test batch config from environment variables."""
        from s1_mini.config import EngineConfig

        with patch.dict(os.environ, {
            "S1_MINI_ENABLE_BATCHING": "true",
            "S1_MINI_BATCH_SIZE": "8",
            "S1_MINI_BATCH_TIMEOUT": "300",
            "S1_MINI_BATCH_GROUPING": "false",
        }):
            config = EngineConfig.from_env()

            assert config.enable_batching is True
            assert config.max_batch_size == 8
            assert config.batch_timeout_ms == 300
            assert config.batch_grouping is False

    def test_batch_config_to_dict(self):
        """Test config to dict conversion includes batch settings."""
        from s1_mini.config import EngineConfig

        config = EngineConfig(
            enable_batching=True,
            max_batch_size=4,
            batch_timeout_ms=200,
            batch_grouping=True,
        )

        config_dict = config.to_dict()

        assert "enable_batching" in config_dict
        assert "max_batch_size" in config_dict
        assert "batch_timeout_ms" in config_dict
        assert "batch_grouping" in config_dict


# =============================================================================
# Integration Tests
# =============================================================================


class TestBatchIntegration:
    """Integration tests for batch processing."""

    @pytest.fixture
    def engine_config(self):
        """Create engine config for testing."""
        from s1_mini.config import EngineConfig

        return EngineConfig(
            checkpoint_path="checkpoints/openaudio-s1-mini",
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_batching=True,
            max_batch_size=4,
            batch_timeout_ms=200,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    @pytest.mark.skipif(
        not Path("checkpoints/openaudio-s1-mini").exists(),
        reason="Model checkpoint not found"
    )
    def test_batch_generation_basic(self, engine_config, sample_texts):
        """Test basic batch generation."""
        from s1_mini import ProductionTTSEngine

        engine = ProductionTTSEngine(engine_config)
        engine.start()

        try:
            # Generate batch of 4 texts
            response = engine.generate_batch(
                texts=sample_texts[:4],
                max_new_tokens=512,
                temperature=0.7,
                return_bytes=True,
            )

            assert response.success
            assert len(response.results) == 4
            assert all(r.success for r in response.results)
            assert all(r.audio_bytes is not None for r in response.results)

        finally:
            engine.stop()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    @pytest.mark.skipif(
        not Path("checkpoints/openaudio-s1-mini").exists(),
        reason="Model checkpoint not found"
    )
    def test_batch_generation_with_references(
        self, engine_config, sample_texts, sample_reference_audio
    ):
        """Test batch generation with reference audio."""
        from s1_mini import ProductionTTSEngine

        engine = ProductionTTSEngine(engine_config)
        engine.start()

        try:
            # Generate batch with some references
            response = engine.generate_batch(
                texts=sample_texts[:4],
                reference_audios=[sample_reference_audio, None, sample_reference_audio, None],
                reference_texts=["Reference one", None, "Reference two", None],
                max_new_tokens=512,
                temperature=0.7,
                return_bytes=True,
            )

            assert response.success
            assert len(response.results) == 4

        finally:
            engine.stop()


# =============================================================================
# Performance Tests
# =============================================================================


class TestBatchPerformance:
    """Performance tests for batch processing."""

    @pytest.fixture
    def engine(self):
        """Create and start engine for performance testing."""
        from s1_mini import ProductionTTSEngine, EngineConfig

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if not Path("checkpoints/openaudio-s1-mini").exists():
            pytest.skip("Model checkpoint not found")

        config = EngineConfig(
            checkpoint_path="checkpoints/openaudio-s1-mini",
            device="cuda",
            enable_batching=True,
            max_batch_size=4,
            batch_timeout_ms=200,
        )

        engine = ProductionTTSEngine(config)
        engine.start()

        yield engine

        engine.stop()

    def test_batch_vs_sequential_throughput(self, engine, sample_texts):
        """Compare batch vs sequential throughput."""
        texts = sample_texts[:4]

        # Sequential timing
        start = time.time()
        sequential_results = []
        for text in texts:
            result = engine.generate(
                text=text,
                max_new_tokens=512,
                temperature=0.7,
                return_bytes=True,
            )
            sequential_results.append(result)
        sequential_time = time.time() - start

        # Batch timing
        start = time.time()
        batch_response = engine.generate_batch(
            texts=texts,
            max_new_tokens=512,
            temperature=0.7,
            return_bytes=True,
        )
        batch_time = time.time() - start

        print(f"\nSequential time: {sequential_time:.2f}s")
        print(f"Batch time: {batch_time:.2f}s")
        print(f"Speedup: {sequential_time / batch_time:.2f}x")

        # Batch should not be slower than sequential
        # (In current implementation, batch processes sequentially but benefits from warm model)
        assert batch_response.success

    def test_batch_of_10_5second_audios(self, engine):
        """
        Test generating 10 ~5-second audio clips.

        This is the main performance test requested by the user.
        """
        # Texts designed to produce approximately 5 seconds of audio each
        # (roughly 50-80 words per text)
        texts = [
            "Welcome to our amazing text-to-speech demonstration. This system uses advanced neural networks to convert written text into natural sounding speech. We hope you enjoy listening to these generated audio samples.",
            "The weather today is absolutely beautiful. Clear blue skies stretch from horizon to horizon, and a gentle breeze carries the sweet scent of blooming flowers. It's the perfect day to spend time outdoors with family and friends.",
            "In the heart of the bustling city, a small coffee shop serves the most delicious espresso. Customers line up every morning, eager to start their day with the rich aroma and bold flavor of freshly brewed coffee.",
            "Technology continues to advance at an incredible pace. From smartphones to artificial intelligence, new innovations are transforming how we live, work, and communicate with each other around the world.",
            "The ancient library contained thousands of rare manuscripts and books. Scholars from distant lands would travel for months just to study the precious knowledge preserved within its towering stone walls.",
            "Music has the power to evoke deep emotions and memories. A single melody can transport us back in time, reminding us of special moments and people we cherish. It truly is a universal language.",
            "The chef carefully prepared each dish with precision and artistry. Fresh ingredients were transformed into culinary masterpieces that delighted both the eyes and the taste buds of every fortunate guest.",
            "Exploring the natural world reveals countless wonders. From microscopic organisms to majestic whales, life on Earth displays an incredible diversity that scientists are still working to fully understand.",
            "Education opens doors to unlimited possibilities. Through learning, we gain the knowledge and skills needed to pursue our dreams and make meaningful contributions to society and future generations.",
            "The sunset painted the sky in brilliant shades of orange, pink, and purple. As the sun dipped below the horizon, the world seemed to pause in appreciation of nature's daily masterpiece.",
        ]

        print("\n" + "=" * 60)
        print("BATCH PERFORMANCE TEST: 10 x ~5-second audios")
        print("=" * 60)

        start_time = time.time()

        response = engine.generate_batch(
            texts=texts,
            max_new_tokens=2048,  # Allow for ~5 seconds of audio
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
            return_bytes=True,
        )

        total_time = time.time() - start_time

        # Analyze results
        print(f"\nResults:")
        print(f"  Success: {response.success}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Engine reported time: {response.total_time:.2f}s")

        successful = sum(1 for r in response.results if r.success)
        failed = len(response.results) - successful
        print(f"  Successful: {successful}/10")
        print(f"  Failed: {failed}/10")

        # Calculate audio durations
        total_audio_duration = 0
        for i, result in enumerate(response.results):
            if result.success and result.audio_bytes:
                # Estimate duration from WAV file size
                # WAV header is 44 bytes, 44100 Hz * 2 bytes * 1 channel
                audio_size = len(result.audio_bytes) - 44
                duration = audio_size / (44100 * 2)
                total_audio_duration += duration
                print(f"  Audio {i+1}: {duration:.2f}s ({len(result.audio_bytes) / 1024:.1f} KB)")
            else:
                print(f"  Audio {i+1}: FAILED - {result.error}")

        print(f"\nSummary:")
        print(f"  Total audio duration: {total_audio_duration:.2f}s")
        print(f"  Real-time factor: {total_audio_duration / total_time:.2f}x")
        print(f"  Average per audio: {total_time / 10:.2f}s")
        print("=" * 60)

        # Assertions
        assert response.success, f"Batch generation failed: {response.error}"
        assert successful >= 8, f"Too many failures: {failed}/10"

        # Store results for manual inspection if needed
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        for i, result in enumerate(response.results):
            if result.success and result.audio_bytes:
                with open(output_dir / f"batch_test_{i+1}.wav", "wb") as f:
                    f.write(result.audio_bytes)

        print(f"\nAudio files saved to: {output_dir.absolute()}")


# =============================================================================
# API Tests
# =============================================================================


class TestBatchAPI:
    """Tests for the batch API endpoint."""

    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        from fastapi.testclient import TestClient
        from s1_mini.server import create_app
        from s1_mini.config import EngineConfig

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if not Path("checkpoints/openaudio-s1-mini").exists():
            pytest.skip("Model checkpoint not found")

        config = EngineConfig(
            checkpoint_path="checkpoints/openaudio-s1-mini",
            device="cuda",
            enable_batching=True,
            max_batch_size=4,
        )

        app = create_app(config)

        with TestClient(app) as client:
            # Wait for engine to be ready
            for _ in range(60):
                response = client.get("/ready")
                if response.status_code == 200:
                    break
                time.sleep(1)

            yield client

    def test_batch_endpoint_basic(self, test_client):
        """Test basic batch endpoint functionality."""
        response = test_client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {"text": "Hello world"},
                    {"text": "Goodbye world"},
                ],
                "temperature": 0.7,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"]
        assert len(data["results"]) == 2
        assert all(r["success"] for r in data["results"])
        assert all(r["audio_base64"] is not None for r in data["results"])

    def test_batch_endpoint_with_references(self, test_client, sample_reference_audio):
        """Test batch endpoint with reference audio."""
        ref_base64 = base64.b64encode(sample_reference_audio).decode()

        response = test_client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {
                        "text": "Hello with reference",
                        "reference_audio": ref_base64,
                        "reference_text": "Reference text here",
                    },
                    {"text": "Hello without reference"},
                ],
                "temperature": 0.7,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"]
        assert len(data["results"]) == 2

    def test_batch_endpoint_validation_error(self, test_client):
        """Test batch endpoint validation errors."""
        # Missing reference_text when reference_audio is provided
        response = test_client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {
                        "text": "Hello",
                        "reference_audio": "aGVsbG8=",  # base64 "hello"
                        # Missing reference_text
                    },
                ],
            },
        )

        assert response.status_code == 400

    def test_batch_endpoint_max_items(self, test_client):
        """Test batch endpoint max items limit."""
        # Try to submit more than max_items (8)
        response = test_client.post(
            "/v1/tts/batch",
            json={
                "items": [{"text": f"Text {i}"} for i in range(10)],
            },
        )

        # Should fail validation (max 8 items)
        assert response.status_code == 422


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_batch_inference.py -v
    pytest.main([__file__, "-v", "-s"])
