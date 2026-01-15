# Fish-Speech S1-Mini Production Inference Engine Plan

## Overview

This document outlines the plan to transform the Fish-Speech S1-Mini inference engine
from a research/local implementation to a production-ready, scalable inference system.

**Target Deployment**: AWS SageMaker (Linux with Triton support)
**Development/Testing**: Windows (with fallback for Triton)

---

## Current Architecture Analysis

### Pipeline Overview
```
Text Input
    │
    ▼
┌─────────────────────────────┐
│  LLAMA/DualARTransformer    │  (text2semantic model)
│  - 32 main transformer layers│
│  - 4 fast layers for codebooks│
└─────────────────────────────┘
    │
    ▼
Semantic VQ Token Codes (8 codebooks)
    │
    ▼
┌─────────────────────────────┐
│  DAC Decoder                │  (audio reconstruction)
│  - Modified Descript Audio  │
│    Codec                    │
└─────────────────────────────┘
    │
    ▼
Audio Output (44.1kHz WAV)
```

### Key Source Files (Original)
| File | Purpose |
|------|---------|
| `fish_speech/inference_engine/__init__.py` | Main TTSInferenceEngine class |
| `fish_speech/models/text2semantic/inference.py` | LLAMA generation & worker thread |
| `fish_speech/models/text2semantic/llama.py` | DualARTransformer model |
| `fish_speech/inference_engine/vq_manager.py` | VQ encoding/decoding |
| `fish_speech/inference_engine/reference_loader.py` | Reference audio caching |
| `fish_speech/models/dac/modded_dac.py` | DAC encoder/decoder |

---

## Identified Problems

### Problem 1: Aggressive VRAM Cache Clearing
**Location**: `inference.py:568-569`, `__init__.py:122-124`

```python
# Called after EVERY generation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

**Impact**:
- Forces VRAM reallocation on every request
- Causes memory fragmentation
- Adds 100-500ms overhead per request

**Solution**: Only clear cache on OOM errors or when explicitly requested

---

### Problem 2: Triton Not Available on Windows
**Location**: `inference.py:26-31`

```python
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True  # FAILS on Windows
```

**Impact**:
- `torch.compile()` fails or falls back to slow eager mode
- 2-5x slower inference on Windows

**Solution**: Platform detection with graceful fallback
```python
import platform
if platform.system() == "Linux":
    # Use Triton-backed inductor
    backend = "inductor"
else:
    # Windows fallback - no Triton
    backend = "eager"
```

---

### Problem 3: Single-Threaded Request Processing
**Location**: `inference.py:551-565`

```python
while True:
    item = input_queue.get()  # BLOCKS - sequential processing
    for chunk in generate_long(...):
        response_queue.put(chunk)
```

**Impact**:
- Only 1 request processed at a time
- N concurrent users = N× latency for last user
- No GPU parallelism utilization

**Solution**:
- Implement request batching
- Use async processing with multiple worker threads
- Dynamic batch assembly with timeout

---

### Problem 4: No KV Cache Persistence
**Location**: `inference.py:132-136`

KV cache is cleared between codebook predictions but not reused across requests.

**Impact**:
- Identical prompts recompute entire KV cache
- Reference audio KV states recalculated every time

**Solution**:
- Implement prefix caching for reference audio
- Cache KV states for common prompt patterns
- LRU cache with configurable size

---

### Problem 5: Hardcoded Batch Size
**Location**: `inference.py:545`

```python
model.setup_caches(max_batch_size=1, ...)  # Hardcoded
```

**Impact**:
- Cannot process multiple requests in parallel
- Wastes GPU compute capacity

**Solution**: Configurable batch size with dynamic batching

---

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: Critical)

#### 1.1 Platform-Aware Compilation Module
**File**: `s1_mini/compilation.py`

- Detect OS and available backends
- Configure torch.compile appropriately
- Provide fallback for Windows development
- Support explicit backend override via environment variable

#### 1.2 Optimized Model Manager
**File**: `s1_mini/model_manager.py`

- Load models once at startup
- Warmup with dummy inference
- Proper VRAM management (no aggressive clearing)
- Health check endpoints
- Graceful model reload capability

#### 1.3 Inference Engine Core
**File**: `s1_mini/engine.py`

- Rewritten TTSInferenceEngine
- Persistent model state
- Optimized memory management
- Comprehensive logging and metrics

---

### Phase 2: Performance Optimization (Priority: High)

#### 2.1 Request Batching
**File**: `s1_mini/batching.py`

- Dynamic batch assembly
- Configurable batch timeout (default: 50ms)
- Batch size limits (default: 4)
- Request padding for variable lengths

#### 2.2 KV Cache Optimization
**File**: `s1_mini/kv_cache.py`

- Prefix caching implementation
- LRU eviction policy
- Cache size configuration
- Cache statistics/monitoring

#### 2.3 Reference Audio Caching
**File**: `s1_mini/reference_cache.py`

- Pre-computed VQ tokens for references
- Memory-mapped cache for large reference sets
- Async cache warming

---

### Phase 3: Production Features (Priority: Medium)

#### 3.1 Production Server
**File**: `s1_mini/server.py`

- FastAPI-based REST API
- Health check endpoints (/health, /ready)
- Metrics endpoint (/metrics)
- Request timeout and cancellation
- Graceful shutdown

#### 3.2 Monitoring and Observability
**File**: `s1_mini/metrics.py`

- Request latency histograms
- Queue depth monitoring
- VRAM usage tracking
- Error rate tracking
- Prometheus-compatible metrics

#### 3.3 Configuration Management
**File**: `s1_mini/config.py`

- Environment-based configuration
- Validation with Pydantic
- Sensible defaults for production

---

### Phase 4: AWS SageMaker Integration (Priority: Medium)

#### 4.1 SageMaker Endpoint
**File**: `s1_mini/sagemaker/`

- Model packaging for SageMaker
- Inference handler implementation
- Multi-model endpoint support
- Auto-scaling configuration

---

## File Structure

```
s1_mini/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── compilation.py           # Platform-aware torch.compile
├── model_manager.py         # Model lifecycle management
├── engine.py                # Core inference engine
├── batching.py              # Request batching logic
├── kv_cache.py              # KV cache optimization
├── reference_cache.py       # Reference audio caching
├── vq_manager.py            # VQ encoding/decoding (optimized)
├── server.py                # Production FastAPI server
├── metrics.py               # Observability and monitoring
├── utils.py                 # Utility functions
├── exceptions.py            # Custom exceptions
└── sagemaker/               # AWS SageMaker integration
    ├── __init__.py
    ├── handler.py           # SageMaker inference handler
    └── package.py           # Model packaging utilities
```

---

## Expected Performance Improvements

| Metric | Current | After Optimization | Improvement |
|--------|---------|-------------------|-------------|
| Cold start | 10-15s | 10-15s | - |
| Warm inference (single) | 5-15s | 1-5s | 3-5x |
| Concurrent throughput | 1 req/time | 4-8 req/batch | 4-8x |
| VRAM usage | 12-16GB | 8-12GB | 25-35% reduction |
| Windows dev speed | Very slow | Reasonable | 2-3x |
| Request timeout | None | Configurable | Safety |

---

## Configuration Options

```python
# Environment variables for configuration
S1_MINI_DEVICE = "cuda"                    # Device to use
S1_MINI_PRECISION = "float16"              # Model precision
S1_MINI_COMPILE = "auto"                   # auto, true, false
S1_MINI_COMPILE_BACKEND = "auto"           # auto, inductor, eager
S1_MINI_BATCH_SIZE = 4                     # Max batch size
S1_MINI_BATCH_TIMEOUT_MS = 50              # Batch assembly timeout
S1_MINI_KV_CACHE_SIZE = 100                # KV cache entries
S1_MINI_REFERENCE_CACHE_SIZE = 1000        # Reference cache entries
S1_MINI_REQUEST_TIMEOUT_S = 60             # Request timeout
S1_MINI_VRAM_CLEAR_ON_OOM = "true"         # Only clear on OOM
S1_MINI_METRICS_ENABLED = "true"           # Enable metrics
```

---

## Rollout Plan

### Stage 1: Development (Local Windows)
1. Implement core engine with Windows fallback
2. Test with single requests
3. Validate output quality matches original

### Stage 2: Testing (Local with batching)
1. Enable batching
2. Load test with concurrent requests
3. Monitor VRAM and latency

### Stage 3: Staging (AWS SageMaker)
1. Deploy to SageMaker staging endpoint
2. Enable Triton compilation
3. Run integration tests
4. Performance benchmarking

### Stage 4: Production
1. Deploy to production endpoint
2. Enable auto-scaling
3. Monitor metrics
4. Gradual traffic migration

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Output quality degradation | A/B testing against original engine |
| VRAM OOM in production | Conservative batch sizes, monitoring |
| Triton compilation failures | Graceful fallback to eager mode |
| Request queue buildup | Timeout and rejection policies |
| Model loading failures | Health checks, automatic restart |

---

## Success Criteria

1. **Latency**: P95 latency < 5s for typical requests
2. **Throughput**: Support 10+ concurrent users per GPU
3. **Reliability**: 99.9% success rate for valid requests
4. **Compatibility**: Output quality matches original engine
5. **Observability**: Full metrics and logging coverage

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2024-XX-XX | 1.0 | Claude | Initial plan |

