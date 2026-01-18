# GPU Optimization & Maximum Throughput Guide

This document explains the current architecture and how to maximize throughput on a single GPU.

## Current Architecture Analysis

### Batching: Static, Not Dynamic

```
Current Implementation:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Request    │───►│ Input Queue  │───►│   Worker    │
│  Queue      │    │ (Python)     │    │  (Thread)   │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ LLaMA Model │
                                       │ batch_size=1│
                                       │ CUDA Graphs │
                                       └─────────────┘
```

**Key Code** (`fish_speech/models/text2semantic/inference.py`):
```python
# Fixed batch size of 1
model.setup_caches(
    max_batch_size=1,  # Cannot be changed without breaking CUDA graphs
    max_seq_len=model.config.max_seq_len,
)

# CUDA graphs with reduce-overhead mode
decode_one_token = torch.compile(
    decode_one_token_ar,
    mode="reduce-overhead",  # Uses CUDA graphs
    fullgraph=True,
)
```

### Why batch_size=1?

1. **CUDA Graphs Requirement**: `mode="reduce-overhead"` captures the entire execution as a CUDA graph
2. **Static Shapes**: CUDA graphs require fixed tensor shapes at capture time
3. **Dynamic Batching Conflict**: Changing batch size would invalidate the captured graph

### Trade-off

| Approach | Latency | Throughput | Implementation |
|----------|---------|------------|----------------|
| batch_size=1 + CUDA graphs | Low | Limited | Current |
| Dynamic batching | Higher | Higher | Requires disable CUDA graphs |
| Continuous batching | Low | High | Complex (vLLM-style) |

---

## GPU Utilization vs VRAM

### Typical RTX 4090 Metrics

```
VRAM Usage:
  Model weights (BF16):     ~1.8 GB
  DAC decoder:              ~1.9 GB
  KV cache (batch=1):       ~1.5 GB
  Activations:              ~1 GB
  CUDA graph workspace:     ~0.5 GB
  ─────────────────────────────────
  Peak during inference:    ~7 GB

GPU Utilization:
  During token generation:  60-80%
  Between requests:         ~0%
  Average sustained:        40-60%
```

### Why GPU is Underutilized

1. **Memory-bound operations**: Attention is memory-bandwidth limited
2. **Sequential token generation**: Each token depends on previous
3. **Python overhead**: Queue management, data transfer
4. **Between-request gaps**: Model waits for next request

---

## Optimization Options

### Option 1: Multiple Python Processes (Recommended)

Run multiple independent Python processes, each with its own model.

```bash
# Terminal 1
python tools/api_server.py --port 8001

# Terminal 2
python tools/api_server.py --port 8002

# Load balancer (nginx/HAProxy)
upstream fish_speech {
    server localhost:8001;
    server localhost:8002;
}
```

**Pros**:
- Each process has independent CUDA context
- True parallelism
- Simple to implement

**Cons**:
- Requires 2x VRAM (~14 GB for 2 instances)
- Each process recompiles kernels on first run

**VRAM Calculation**:
```
24 GB total (RTX 4090)
- 7 GB per instance
= 3 instances maximum (21 GB)
= Leaves 3 GB headroom
```

### Option 2: Disable CUDA Graphs

Change compilation mode to allow dynamic batching:

```python
# In inference.py, change:
decode_one_token = torch.compile(
    decode_one_token_ar,
    mode="default",  # Instead of "reduce-overhead"
    fullgraph=True,
)
```

**Pros**:
- Enables concurrent threading
- Single model instance
- Lower VRAM

**Cons**:
- 30-40% slower per request
- May not provide net throughput gain

### Option 3: Request Batching (Pre-collection)

Collect multiple requests, process together:

```python
# Pseudo-code for request batching
pending_requests = []
batch_timeout = 0.1  # 100ms

while True:
    # Collect requests for batch_timeout
    while len(pending_requests) < max_batch and time_elapsed < batch_timeout:
        if request_available:
            pending_requests.append(request)

    # Process batch
    if pending_requests:
        results = process_batch(pending_requests)
        for req, result in zip(pending_requests, results):
            req.respond(result)
        pending_requests.clear()
```

**Pros**:
- Better GPU utilization
- Works with single model

**Cons**:
- Adds latency (waiting for batch)
- Complex to implement correctly
- Requires disabling CUDA graphs

### Option 4: Speculative Decoding (Advanced)

Use a smaller model to predict multiple tokens, verify with main model.

**Pros**:
- 1.5-2x speedup potential
- Works with CUDA graphs

**Cons**:
- Requires draft model
- Complex implementation
- May affect quality

---

## Practical Recommendations

### For RTX 4090 (24 GB)

**Best Configuration**:
```
2-3 Python processes (multiprocessing)
Each process: BF16 + torch.compile
Load balancer distributes requests
```

**Expected Throughput**:
```
Single instance:   ~45 requests/minute
2 instances:       ~80-90 requests/minute
3 instances:       ~100-120 requests/minute (if VRAM allows)
```

### For RTX 3090/4080 (24 GB)

Same as RTX 4090, but slightly lower per-instance throughput.

### For GPUs with 12 GB VRAM

```
1 instance only (BF16)
OR
2 instances with INT8 quantization
```

---

## Stress Test Script

Run the comprehensive GPU stress test:

```bash
# 60-second stress test
python tools/runpod/gpu_stress_test.py

# 5-minute stress test
python tools/runpod/gpu_stress_test.py --duration 300

# Test multi-instance feasibility
python tools/runpod/gpu_stress_test.py --test-multi-instance

# Test with specific text lengths
python tools/runpod/gpu_stress_test.py --text-length short
```

### Output Metrics

```
THROUGHPUT:
  Requests/minute:    45.2
  Requests/second:    0.75
  Throughput:         5.5x realtime

LATENCY:
  Average:            1.33s
  P50:                1.21s
  P95:                1.89s
  P99:                2.15s

GPU METRICS:
  Avg GPU util:       62%
  Peak VRAM:          6,824 MB
  Avg Power:          285 W
```

---

## Multi-Process Deployment

### Using Python multiprocessing

```python
# multi_worker_server.py
import multiprocessing
import subprocess

def start_worker(port):
    subprocess.run([
        "python", "tools/api_server.py",
        "--listen", f"0.0.0.0:{port}",
        "--device", "cuda",
    ])

if __name__ == "__main__":
    ports = [8001, 8002, 8003]  # 3 workers
    processes = []

    for port in ports:
        p = multiprocessing.Process(target=start_worker, args=(port,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### Using Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  fish-speech-1:
    image: fish-speech
    ports:
      - "8001:8080"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
    environment:
      - CUDA_VISIBLE_DEVICES=0

  fish-speech-2:
    image: fish-speech
    ports:
      - "8002:8080"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### Using nginx Load Balancer

```nginx
# nginx.conf
upstream fish_speech {
    least_conn;  # Use least connections algorithm
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;

    location / {
        proxy_pass http://fish_speech;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

---

## Monitoring

### Real-time GPU Monitoring

```bash
# Watch GPU utilization
watch -n 0.5 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet -d 1
```

### Key Metrics to Monitor

1. **GPU Utilization**: Should be 60-80% during inference
2. **Memory Utilization**: Should stay below 90% of total
3. **Power Draw**: Near TDP indicates good utilization
4. **SM Clock**: Should be at boost frequency

---

## Summary

| Configuration | VRAM | Requests/min | Best For |
|---------------|------|--------------|----------|
| Single instance | 7 GB | ~45 | Low traffic |
| 2 processes | 14 GB | ~80-90 | Medium traffic |
| 3 processes | 21 GB | ~100-120 | High traffic |
| INT8 + 2 proc | 10 GB | ~75-85 | Limited VRAM |

**Key Insight**: The GPU is underutilized with a single instance because:
1. Autoregressive generation is inherently sequential
2. CUDA graphs prevent dynamic batching
3. Memory bandwidth limits attention computation

**Best Solution**: Multiple Python processes with load balancing.
