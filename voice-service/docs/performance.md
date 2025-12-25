# Performance

Tuning and benchmarks.

## Capacity

Estimates for A10G (24GB):

| Config | Throughput |
|--------|------------|
| batch=1 | ~15 req/min |
| batch=4 | ~45 req/min |
| batch=8 | ~70 req/min |

## GPU Sizing

| Concurrent Users | Avg Load | GPUs Needed |
|------------------|----------|-------------|
| 1,000 | 200 req/min | 3 |
| 10,000 | 2,000 req/min | 30 |
| 50,000 | 10,000 req/min | 150 |

Assumes 1 request per user per 5 minutes at peak.

## Batching

Enable batching for higher throughput:

```python
from src.inference.batcher import RequestBatcher

batcher = RequestBatcher(
    engine=engine,
    max_batch_size=4,   # Requests per batch
    max_wait_ms=100,    # Max wait before processing
)
await batcher.start()
```

Trade-off:
- Higher batch size = better GPU utilization
- Higher wait time = more latency

## torch.compile

Enable for ~20% faster inference:

```bash
VOICE_S1_COMPILE=true
```

First request is slow (compilation). Best for long-running servers.

## Memory

S1-Mini memory usage:

| Component | VRAM |
|-----------|------|
| LLaMA model | ~1.5GB |
| DAC decoder | ~0.3GB |
| KV cache | ~0.5GB |
| Total | ~2.5GB |

Leaves ~20GB free on A10G for batching.

## Benchmarks

Run benchmarks:

```bash
cd voice-service
pytest tests/test_benchmark.py --gpu -v -s
```

Output:

```
Single request latency: 2500ms (+/- 200ms)
Batch=1 throughput: 0.40 req/sec (24 req/min)
Batch=4 throughput: 1.20 req/sec (72 req/min)
Real-time factor: 3.5x
GPU memory: 2.5GB allocated
```

## Optimization Checklist

1. Enable `VOICE_S1_COMPILE=true` for production
2. Use batching for high-throughput workloads
3. Monitor GPU utilization with `nvidia-smi`
4. Use streaming for faster time-to-first-byte
5. Pre-warm model with health check on startup

## Monitoring

Key metrics:

| Metric | Alert Threshold |
|--------|-----------------|
| Request latency p99 | > 10s |
| GPU utilization | < 50% (underutilized) |
| Queue depth | > 100 |
| Error rate | > 1% |

Prometheus metrics (if enabled):

```
voice_requests_total
voice_request_duration_seconds
voice_gpu_memory_bytes
```








