"""Performance benchmark tests. Run with: pytest --gpu -v"""

import time

import numpy as np
import pytest


@pytest.mark.gpu
class TestBenchmarks:
    """Performance benchmarks for production sizing."""

    def test_single_request_latency(self, engine):
        """Measure single request latency."""
        text = "Measure single request latency."
        
        # Warmup
        list(engine.generate(text=text, max_new_tokens=256))

        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            list(engine.generate(text=text, max_new_tokens=256))
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000

        print(f"\nSingle request latency: {avg_ms:.0f}ms (+/- {std_ms:.0f}ms)")
        assert avg_ms < 10000  # Should complete within 10s

    def test_throughput_batch_1(self, engine):
        """Measure throughput with batch size 1."""
        text = "Throughput test request."
        num_requests = 10

        # Warmup
        list(engine.generate(text=text, max_new_tokens=256))

        start = time.perf_counter()
        for _ in range(num_requests):
            list(engine.generate(text=text, max_new_tokens=256))
        elapsed = time.perf_counter() - start

        rps = num_requests / elapsed
        print(f"\nBatch=1 throughput: {rps:.2f} req/sec ({rps * 60:.0f} req/min)")

    def test_throughput_batch_4(self, engine):
        """Measure throughput with batch size 4."""
        requests = [
            {"text": f"Batch request {i}.", "max_new_tokens": 256}
            for i in range(4)
        ]
        num_batches = 5

        # Warmup
        engine.generate_batch(requests[:1])

        start = time.perf_counter()
        for _ in range(num_batches):
            engine.generate_batch(requests)
        elapsed = time.perf_counter() - start

        total_requests = num_batches * len(requests)
        rps = total_requests / elapsed
        print(f"\nBatch=4 throughput: {rps:.2f} req/sec ({rps * 60:.0f} req/min)")

    def test_audio_generation_speed(self, engine):
        """Measure audio seconds generated per wall-clock second."""
        text = "This is a longer piece of text to measure audio generation speed."

        # Warmup
        list(engine.generate(text=text, max_new_tokens=512))

        samples = []
        for _ in range(3):
            start = time.perf_counter()
            segments = list(engine.generate(text=text, max_new_tokens=512))
            elapsed = time.perf_counter() - start

            audio = segments[0]
            audio_duration = len(audio) / engine.sample_rate
            samples.append(audio_duration / elapsed)

        rtf = np.mean(samples)  # Real-time factor
        print(f"\nReal-time factor: {rtf:.2f}x (>{1.0} = faster than realtime)")
        assert rtf > 0.1  # Should be at least 0.1x realtime

    def test_memory_usage(self, engine, gpu_available):
        """Report GPU memory usage."""
        if not gpu_available:
            pytest.skip("GPU not available")

        import torch

        torch.cuda.reset_peak_memory_stats()

        # Run a generation
        list(engine.generate(text="Memory test.", max_new_tokens=512))

        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3

        print(f"\nGPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def test_concurrent_capacity_estimate(self, engine):
        """Estimate concurrent request capacity."""
        text = "Capacity estimation request."
        num_samples = 5

        times = []
        for _ in range(num_samples):
            start = time.perf_counter()
            list(engine.generate(text=text, max_new_tokens=256))
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        requests_per_min = 60 / avg_time

        print(f"\nEstimated capacity:")
        print(f"  Single GPU: {requests_per_min:.0f} req/min")
        print(f"  3 GPUs: {requests_per_min * 3:.0f} req/min")
        print(f"  6 GPUs: {requests_per_min * 6:.0f} req/min")
        print(f"  12 GPUs: {requests_per_min * 12:.0f} req/min")








