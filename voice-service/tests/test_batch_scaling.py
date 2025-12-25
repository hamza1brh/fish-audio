"""Test batch size scaling on GPU."""
import time

import pytest
import torch


@pytest.mark.gpu
def test_batch_size_scaling(engine):
    """Measure throughput at different batch sizes."""
    print(f"\nBaseline VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-" * 60)

    for batch_size in [1, 2, 4, 8]:
        requests = [
            {"text": f"Request number {i}.", "max_new_tokens": 256}
            for i in range(batch_size)
        ]

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        results = engine.generate_batch(requests)
        elapsed = time.perf_counter() - start

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        throughput = batch_size / elapsed

        print(
            f"Batch={batch_size}: {throughput:.2f} req/sec "
            f"({throughput*60:.0f} req/min), Peak VRAM: {peak_mem:.2f} GB"
        )

        assert len(results) == batch_size
        for r in results:
            assert len(r) > 0


