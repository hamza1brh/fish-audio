# GPU Optimization Analysis: Report vs Fish Speech

## Executive Summary

After analyzing the GPU optimization report and comparing it to Fish Speech's current architecture, here's what's applicable.

**Key Finding:** Fish Speech uses `batch_size=1` with CUDA graphs, which prevents dynamic batching. The report recommends techniques that require architectural changes, but some optimizations can be applied with minimal effort.

---

## Current Fish Speech Architecture

```
fish_speech/models/text2semantic/inference.py:641-645
+--------------------------------------------------------------+
|  model.setup_caches(                                         |
|      max_batch_size=1,           <-- FIXED at 1              |
|      max_seq_len=model.config.max_seq_len,                   |
|      dtype=next(model.parameters()).dtype,                   |
|  )                                                           |
+--------------------------------------------------------------+

+---------------+    +--------------+    +---------------------+
| Request Queue |--->| Single Thread|--->| CUDA Graphs         |
| (Python)      |    | Worker       |    | (reduce-overhead)   |
+---------------+    +--------------+    +---------------------+
                                                   |
                                                   v
                                         +---------------------+
                                         | batch_size=1 ONLY   |
                                         | Static shapes       |
                                         +---------------------+
```

**Current Optimizations Already Applied:**
- `torch.compile(mode="reduce-overhead")` with CUDA graphs
- Persistent kernel caching (`~/.cache/fish_speech/torch_compile`)
- INT8/INT4 runtime quantization support
- Pre-created fixed parameter tensors

---

## Batching Terminology Clarification

| Type | Description | Provider | Throughput Gain |
|------|-------------|----------|-----------------|
| **Static Batching** | Fixed batch size, all requests must finish together | Default | Baseline |
| **Dynamic Batching** | Collects N requests, processes as a batch | Triton | 2-4x |
| **Continuous Batching** | Adds/removes requests at EACH token step | vLLM/SGLang | Up to 23x |

**Important:** Triton provides **dynamic batching**, NOT **continuous batching**.

With Fish Speech's CUDA graphs (batch_size=1), Triton would queue requests but process them one-by-one - no actual batching benefit unless CUDA graphs are disabled (losing 30-40% performance).

---

## Optimization Applicability Matrix

| Optimization | Report Section | Effort | Impact | Applicable? |
|--------------|----------------|--------|--------|-------------|
| **Multi-process workers** | Async Pipeline | Low | High | YES |
| **KV Cache FP8/INT8** | KV Cache Quantization | Low | Medium | YES |
| **FlashAttention** | Memory-efficient Attention | Medium | Medium | PARTIAL |
| **Async Pipeline** | Throughput Techniques | Medium | Medium | YES |
| **Triton Server** | Production Deployment | Medium | High | LATER |
| **Piecewise CUDA Graphs** | CUDA Graphs Trade-off | High | High | COMPLEX |
| **Continuous Batching** | Dynamic Batching | Very High | Very High | MAJOR REWRITE |
| **PagedAttention** | KV Cache Optimization | Very High | High | REQUIRES vLLM |

---

## Detailed Analysis

### 1. PagedAttention - NOT DIRECTLY APPLICABLE

**What the report says:**
> Standard KV cache implementations waste 60-80% of memory due to pre-allocation for maximum sequence length. PagedAttention solves this through block-based memory management.

**Fish Speech reality:**
- Uses standard pre-allocated KV cache (`inference.py:641`)
- Implementation requires custom CUDA kernels or vLLM integration
- vLLM's LLM class expects HuggingFace-compatible models
- Fish Speech's `DualARTransformer` is custom, not HF-compatible

**Verdict:** Would require either:
1. Porting model to HuggingFace format + using vLLM (major effort)
2. Implementing custom PagedAttention kernels (very major effort)

**Recommendation:** Skip for now. Multi-process workers achieve similar memory efficiency by running multiple independent instances.

---

### 2. KV Cache Quantization - DEFERRED

**What the report says:**
> FP8 KV cache provides 2x memory savings with <1% accuracy loss on RTX 4090

**Fish Speech reality:**
After investigation, FP8 KV cache is NOT easily applicable because:

1. **PyTorch's SDPA doesn't support FP8 inputs** - The `scaled_dot_product_attention` function requires FP16/BF16/FP32 tensors for K and V
2. **Requires custom CUDA kernels** - vLLM's FP8 KV cache uses custom kernels, not native PyTorch
3. **Attention computation changes needed** - Would need to convert FP8 -> FP16 before attention, negating some benefits

```python
# Current implementation (llama.py:192-210)
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16):
        # dtype is used for k_cache and v_cache tensors
        # Changing to FP8 would require attention layer modifications
```

**Alternative approaches:**
1. INT8 quantization of model weights (already supported via `--runtime-int8`)
2. Run more worker processes (uses more VRAM but maximizes throughput)

**Recommendation:** Deferred - Multi-process workers provide better ROI with less complexity

---

### 3. FlashAttention - PARTIAL BENEFIT

**What the report says:**
> FlashAttention reduces memory from O(N^2) to O(N), FlashDecoding provides up to 8x speedup for long sequences

**Fish Speech current state:**
```python
# inference.py:49-76 - Already has SDPA backend selection
from torch.nn.attention import SDPBackend, sdpa_kernel

# Uses MATH backend when compile=True for Inductor compatibility
if compile:
    with sdpa_kernel(SDPBackend.MATH):
        next_token = decode_one_token(...)
```

**The catch:** When `torch.compile` is enabled with `reduce-overhead`, it uses MATH backend (not FlashAttention) for Inductor codegen compatibility.

**Options:**
1. Keep current approach (compile + MATH) - already getting good perf
2. Switch to FlashAttention when compile=False
3. Test if FlashAttention works with torch.compile (may require `mode="default"`)

**Recommendation:** Test FlashAttention with `mode="default"` to compare throughput vs CUDA graphs approach

---

### 4. Triton Inference Server - LATER

**What the report says:**
> Triton Inference Server provides dynamic batching out of the box

**Config example from report:**
```protobuf
dynamic_batching {
    preferred_batch_size: [8, 16, 32]
    max_queue_delay_microseconds: 50000  # 50ms
}
```

**For Fish Speech:**
1. **Python Backend** (easier) - Wrap existing inference code
2. **ONNX Backend** (harder) - Export model to ONNX

**Key limitation:** Triton's dynamic batching collects requests, but Fish Speech's CUDA graphs require batch_size=1. So Triton would queue and process one-by-one.

**Recommendation:** Defer to later. Multi-process workers provide similar throughput gains with less complexity.

---

### 5. Piecewise CUDA Graphs - COMPLEX BUT POSSIBLE

**What the report says:**
```python
# Pre-compile graphs for discrete batch sizes
compile_sizes = [1, 2, 4, 8, 16, 32, 64]
graphs = {}
for size in compile_sizes:
    graphs[size] = capture_cuda_graph(model, batch_size=size)

# At runtime, round up to nearest compiled size
def inference(batch):
    size = next(s for s in compile_sizes if s >= len(batch))
    return graphs[size].replay(batch)
```

**For Fish Speech:**
- Would need to modify `model.setup_caches()` to support multiple batch sizes
- Need to capture CUDA graphs for each batch size
- Runtime logic to select appropriate graph

**Challenges:**
1. `DualARTransformer` cache setup is batch-size specific
2. Would need to pre-warm all batch sizes (long startup)
3. Memory overhead for multiple graph captures

**Recommendation:** Medium-term optimization. Multi-process is simpler for now.

---

### 6. Continuous Batching - MAJOR ARCHITECTURAL CHANGE

**What the report says:**
> vLLM's continuous batching achieves up to 23x throughput improvement

**Why it's hard for Fish Speech:**
1. CUDA graphs require static tensor shapes
2. Current architecture: single-threaded queue processor
3. Would need to rewrite the entire inference loop:
   - Implement iteration-level scheduling
   - Track per-request KV cache positions
   - Handle variable-length sequences in same batch

**Recommendation:** Not practical without major rewrite. Consider vLLM/SGLang integration for a future "v2" architecture.

---

## Recommended Implementation Plan

### Tier 1: Quick Wins (Implement Now)

| Optimization | Effort | Impact | Status |
|--------------|--------|--------|--------|
| Multi-process workers | Low | 2-3x throughput | **DONE** |
| KV Cache FP8 | High | ~50% KV cache VRAM | Deferred (requires custom kernels) |

**Multi-process workers:**
- Run 2-3 independent Python processes
- Load balancer distributes requests
- Each process: 7GB VRAM x 3 = 21GB (fits RTX 4090)

**KV Cache FP8:**
- Modify `setup_caches()` to use FP8 dtype
- May require attention layer adjustments

### Tier 2: Medium Effort (Optional)

| Optimization | Effort | Impact |
|--------------|--------|--------|
| Async Pipeline with Overlapping | Medium | 20-40% additional |
| FlashAttention testing | Medium | Unknown |

### Tier 3: Deferred

| Optimization | Effort | Why Deferred |
|--------------|--------|--------------|
| Triton Server | Medium | Multi-process simpler |
| Piecewise CUDA Graphs | High | Memory overhead |
| PagedAttention | Very High | Requires vLLM |
| Continuous Batching | Very High | Major rewrite |

---

## Key Insight from Report

> "The critical bottleneck: memory bandwidth, not compute"

Fish Speech at batch_size=1 is **memory-bandwidth bound**. Options to improve:

1. **Increase batch size** -> Shifts to compute-bound (needs continuous batching)
2. **Run multiple instances** -> Multiplies throughput linearly
3. **Reduce memory transfers** -> KV cache quantization, FlashAttention

**Multi-process workers (Option 2) is the simplest path** given current architecture constraints.

---

## Summary Table

| What | Do It? | Why |
|------|--------|-----|
| Multi-process workers | YES | Simple, 2-3x throughput |
| KV Cache FP8 | YES | Easy VRAM savings |
| Triton Server | LATER | Multi-process is simpler for now |
| Piecewise CUDA Graphs | LATER | Complex, multi-process is simpler |
| PagedAttention | NO | Requires vLLM or custom kernels |
| Continuous Batching | NO | Major rewrite, would need vLLM |

---

## Implemented Tools

### Multi-Worker Server (`tools/multi_worker_server.py`)

Launches multiple independent API server processes for parallel request handling.

```bash
# Start 2 workers on ports 8001-8002
python tools/multi_worker_server.py --workers 2 --base-port 8001

# Start 3 workers with load balancer on port 8000
python tools/multi_worker_server.py --workers 3 --base-port 8001 --with-balancer

# Stagger startup for kernel cache sharing
python tools/multi_worker_server.py --workers 2 --stagger-start 10
```

### Load Balancer (`tools/load_balancer.py`)

Distributes requests across workers using round-robin or least-connections.

```bash
# Start load balancer pointing to workers
python tools/load_balancer.py --port 8000 --workers 8001,8002,8003

# Use least-connections strategy
python tools/load_balancer.py --port 8000 --workers 8001,8002 --strategy least-connections
```

**Endpoints:**
- `POST /v1/tts` - TTS requests (proxied to workers)
- `GET /v1/health` - Health check all workers
- `GET /v1/workers` - Worker statistics

### Multi-Worker Benchmark (`tools/benchmark_multi_worker.py`)

Benchmark and compare single vs multi-worker throughput.

```bash
# Benchmark single endpoint
python tools/benchmark_multi_worker.py --endpoint http://localhost:8080 --num-requests 20

# Benchmark with concurrent requests
python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --concurrent 4

# Compare single vs multi-worker
python tools/benchmark_multi_worker.py --compare --single-endpoint http://localhost:8080 --multi-endpoint http://localhost:8000
```

---

## Files Created/Modified

| File | Changes |
|------|---------|
| `tools/multi_worker_server.py` | NEW - Multi-process launcher |
| `tools/load_balancer.py` | NEW - Load balancer with least-connections |
| `tools/benchmark_multi_worker.py` | NEW - Multi-worker benchmark |
| `docs/GPU_OPTIMIZATION_ANALYSIS.md` | NEW - This document |

---

## Verification Plan

1. **Multi-process baseline**: Run 2 workers, measure throughput vs single
2. **KV Cache FP8**: Compare VRAM and quality with FP8 vs BF16
3. **Load test**: Concurrent requests to load balancer
