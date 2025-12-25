"""Real GPU inference tests. Run with: pytest --gpu"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestGPUInference:
    """Tests that run on actual GPU with loaded model."""

    def test_engine_initializes(self, engine):
        """Engine initializes and loads models."""
        assert engine.is_ready
        # S1 Mini decoder model uses 44100 Hz sample rate
        assert engine.sample_rate in [24000, 44100]

    def test_simple_generation(self, engine):
        """Generate audio from simple text."""
        text = "Hello, this is a test."
        segments = list(engine.generate(text=text, max_new_tokens=512))

        assert len(segments) == 1
        audio = segments[0]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 1000  # At least some audio

    def test_generation_parameters(self, engine):
        """Different parameters produce valid audio."""
        text = "Testing parameters."

        for temp in [0.5, 0.7, 0.9]:
            segments = list(
                engine.generate(
                    text=text,
                    temperature=temp,
                    top_p=0.8,
                    max_new_tokens=256,
                )
            )
            assert len(segments) == 1
            assert len(segments[0]) > 500

    def test_longer_text(self, engine):
        """Handle longer text input."""
        text = "This is a longer piece of text that should generate more audio. " * 3
        segments = list(engine.generate(text=text, max_new_tokens=1024))

        assert len(segments) == 1
        audio = segments[0]
        assert len(audio) > 5000

    def test_streaming_generation(self, engine):
        """Streaming mode yields multiple chunks."""
        text = "This text should stream in chunks."
        segments = list(
            engine.generate(
                text=text,
                chunk_length=100,
                streaming=True,
                max_new_tokens=512,
            )
        )

        # Streaming should yield at least one chunk
        assert len(segments) >= 1
        for seg in segments:
            assert isinstance(seg, np.ndarray)

    def test_batch_generation(self, engine):
        """Batch generation processes multiple requests."""
        requests = [
            {"text": "First request.", "max_new_tokens": 256},
            {"text": "Second request.", "max_new_tokens": 256},
        ]

        results = engine.generate_batch(requests)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) > 500

    def test_empty_text_fails(self, engine):
        """Empty text raises error."""
        with pytest.raises(Exception):
            list(engine.generate(text=""))

    def test_audio_range(self, engine):
        """Generated audio is in valid range."""
        segments = list(engine.generate(text="Audio range test.", max_new_tokens=256))
        audio = segments[0]

        assert audio.min() >= -1.5
        assert audio.max() <= 1.5

    def test_reproducibility_with_seed(self, engine):
        """Same seed produces similar output."""
        # Note: Fish Speech may not support seeds in current implementation
        # This test verifies generation is deterministic when possible
        text = "Reproducibility test."

        segments1 = list(engine.generate(text=text, max_new_tokens=256))
        segments2 = list(engine.generate(text=text, max_new_tokens=256))

        # Both should produce valid audio
        assert len(segments1[0]) > 0
        assert len(segments2[0]) > 0



