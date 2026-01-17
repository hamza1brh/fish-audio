# Fish-Speech Production Deployment Guide

Comprehensive guide covering torch.compile optimization, quantization, batch inference, and cloud deployment (RunPod & SageMaker).

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [torch.compile Optimization](#torchcompile-optimization)
3. [Persistent Kernel Caching](#persistent-kernel-caching)
4. [Quantization](#quantization)
5. [Batch Inference](#batch-inference)
6. [Benchmark System](#benchmark-system)
7. [RunPod Deployment](#runpod-deployment)
8. [SageMaker Deployment](#sagemaker-deployment)
9. [VRAM Optimization](#vram-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Performance Overview

### Before vs After Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tokens/sec | ~12 | 116-172 | **10-14x** |
| RTF (Real-Time Factor) | 1.77 | 0.18 | **10x faster** |
| Throughput | 0.56x realtime | 5.55x realtime | **10x** |
| Audio/hour | ~30 min | ~330 min | **11x** |

### Key Optimizations Applied

1. **torch.compile** with `reduce-overhead` mode (CUDA graphs)
2. **Persistent kernel caching** - no recompilation on restart
3. **SDPA backend optimization** - MATH backend for compiled code
4. **INT8 quantization** - 50% VRAM reduction
5. **Model persistence** - keep models loaded in VRAM

---

## torch.compile Optimization

### How It Works

torch.compile transforms Python code into optimized CUDA kernels using the Inductor backend. With `reduce-overhead` mode, it uses CUDA graphs for near-zero kernel launch overhead.

```python
# In fish_speech/models/text2semantic/inference.py
decode_one_token = torch.compile(
    decode_one_token_ar,
    mode="reduce-overhead",
    fullgraph=True,
)
```

### Key Configuration

```python
# Enable for maximum performance
compile=True  # Default in all entry points

# SDPA backend selection - critical for performance
# Use MATH backend only when compile is enabled (better for Inductor codegen)
# Otherwise use default (Flash Attention) for better performance
if compile:
    with sdpa_kernel(SDPBackend.MATH):
        next_token = decode_one_token(...)
else:
    next_token = decode_one_token(...)  # Uses Flash Attention
```

### Why MATH Backend for Compiled Code?

The Inductor compiler generates better fused kernels with the MATH attention backend. Flash Attention, while fast, is a pre-compiled library that can't be fused with surrounding operations.

### Entry Points with Compile Support

| File | Flag | Default |
|------|------|---------|
| `tools/run_webui.py` | `--compile` / `--no-compile` | Enabled |
| `tools/api_server.py` | `--compile` / `--no-compile` | Enabled |
| `tools/runpod/benchmark.py` | Always enabled | Enabled |

```bash
# Run with compilation (default)
python tools/run_webui.py

# Disable for debugging
python tools/run_webui.py --no-compile
```

---

## Persistent Kernel Caching

### The Problem

First-run torch.compile takes 30-60 seconds to compile kernels. Without caching, this happens on every restart.

### The Solution

We persist compiled kernels to disk so subsequent runs start instantly.

```python
# In fish_speech/models/text2semantic/inference.py

# Cache directory for compiled kernels
CACHE_DIR = Path.home() / ".cache" / "fish_speech" / "torch_compile"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)

# Enable all caching features
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True

if hasattr(torch._inductor.config, "kernel_cache"):
    torch._inductor.config.kernel_cache = True
```

### Cache Location

```
~/.cache/fish_speech/torch_compile/
├── triton/                  # Triton kernels
├── inductor/                # Inductor-generated code
└── fx_graph_cache/          # FX graph cache
```

### Cache Behavior

| Scenario | Warmup Time |
|----------|-------------|
| First run (empty cache) | 30-60s |
| Subsequent runs (cached) | 2-5s |
| After PyTorch upgrade | 30-60s (recompile) |

### Clearing the Cache

```bash
rm -rf ~/.cache/fish_speech/torch_compile/
```

---

## Quantization

### Overview

We support two quantization strategies using TorchAO:

| Component | Method | VRAM Reduction | Quality Impact |
|-----------|--------|----------------|----------------|
| LLaMA (semantic) | INT8 weight-only | ~50% | None |
| DAC (decoder) | INT8 weight-only | ~48% | None |
| LLaMA | INT4 weight-only | ~75% | Slight degradation |

### Runtime Quantization (Recommended)

Quantize at load time for maximum compatibility:

```python
# In fish_speech/models/text2semantic/inference.py
from torchao.quantization import quantize_, Int8WeightOnlyConfig

def init_model(checkpoint_path, device, precision, compile=False,
               runtime_int4=False, runtime_int8=False):
    model = DualARTransformer.from_pretrained(checkpoint_path)
    model.to(device=device, dtype=precision)

    if runtime_int8 and device != "cpu":
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())

    return model
```

### DAC Decoder Quantization

```python
# In fish_speech/models/dac/inference.py
def load_model(config_name, checkpoint_path, device="cuda", quantize_int8=False):
    model = instantiate(cfg)
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.to(device)

    if quantize_int8 and device != "cpu":
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())

    return model
```

### VRAM Usage by Configuration

| Configuration | LLaMA | DAC | Total Baseline | Peak |
|---------------|-------|-----|----------------|------|
| BF16 | 1.8 GB | 1.9 GB | 3.7 GB | 6-8 GB |
| BF16 + DAC INT8 | 1.8 GB | 1.0 GB | 2.8 GB | 5-7 GB |
| INT8 + DAC INT8 | 1.1 GB | 1.0 GB | 2.1 GB | 4-6 GB |

### Pre-quantized Checkpoints

For faster startup, you can pre-quantize and save:

```bash
# Quantize with TorchAO (modern approach)
python tools/llama/quantize_torchao.py --mode int8

# Creates: checkpoints/openaudio-s1-mini-int8/
```

### Requirements

```bash
pip install torchao>=0.5.0
```

**Note**: TorchAO requires compatible PyTorch versions. Check compatibility:

```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig
# If this imports without error, you're good
```

---

## Batch Inference

### The Challenge

torch.compile with `reduce-overhead` mode uses CUDA graphs, which are incompatible with Python threading. Concurrent requests cause TLS assertion errors.

### Solution: Sequential Batch Processing

Process requests one-by-one in a single thread, leveraging compiled kernels:

```python
def benchmark_batch_sequential(
    checkpoint_path: str,
    num_requests: int = 10,
    runtime_int8: bool = False,
    dac_int8: bool = False,
    compile_mode: bool = True,
) -> BatchMetrics:
    """
    Benchmark batch/sequential inference.
    Works with torch.compile and CUDA graphs.
    """
    # Load models once
    engine = TTSInferenceEngine(...)

    # Process requests sequentially
    for i in range(num_requests):
        for result in engine.inference(req):
            if result.code == "final":
                audio_result = result.audio
```

### Why Not Concurrent?

```python
# This FAILS with CUDA graphs:
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(generate, text) for text in texts]

# Error: "CUDA graphs does not support changing TLS state"
```

### Batch Performance

With compiled kernels, sequential processing is fast enough for most use cases:

| Metric | Value |
|--------|-------|
| Requests/sec | 1.0-1.5 |
| Throughput | 5x realtime |
| Latency | 0.8-1.2s per request |

### Running Batch Benchmark

```bash
# Default: 10 sequential requests
python tools/runpod/benchmark.py

# More requests
python tools/runpod/benchmark.py --batch-size 20

# Disable batch test
python tools/runpod/benchmark.py --batch-size 0
```

---

## Benchmark System

### Overview

The benchmark system (`tools/runpod/benchmark.py`) provides comprehensive performance testing:

### Features

- **Automatic torch.compile** - always enabled for optimal performance
- **Multiple configurations** - BF16, INT8, INT8+DAC INT8
- **Comprehensive metrics** - RTF, tokens/sec, latency percentiles, VRAM
- **Batch testing** - sequential request processing
- **Visual reports** - HTML with matplotlib charts
- **JSON export** - machine-readable results

### CLI Options

```bash
python tools/runpod/benchmark.py \
  --num-samples 10 \        # Samples per test text
  --batch-size 20 \         # Sequential batch requests
  --warmup-runs 3 \         # Warmup iterations
  --output report.html \    # HTML report
  --json results.json       # JSON export
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| RTF | Real-time factor (gen_time / audio_duration) |
| Tokens/sec | Semantic tokens generated per second |
| Throughput | Audio seconds per compute second |
| VRAM baseline | Memory after model load |
| VRAM peak | Maximum during generation |
| Warmup time | First inference time (includes compile) |
| P50/P95/P99 | Latency percentiles |
| Audio/hour | Production capacity estimate |

### Generated Plots

1. **VRAM Comparison** - Bar chart of baseline/peak VRAM
2. **RTF Comparison** - Bar chart of real-time factors
3. **Latency Distribution** - Box plot of generation times
4. **Warmup Time** - Bar chart (shows compile overhead)
5. **Audio Production** - Capacity estimates

### Sample Output

```
======================================================================
Fish-Speech Benchmark Suite
======================================================================

GPU Information:
  name: NVIDIA GeForce RTX 4090
  total_memory_gb: 24.0
  pytorch_version: 2.4.0+cu124
  torch.compile: Enabled (required for optimal performance)

Testing 3 configurations...

[1/3] BF16 (compiled)
  LLaMA VRAM: 1824.5 MB
  DAC VRAM: 1870.3 MB
  Warmup time: 45.2s (includes compilation)

Results:
  VRAM: 3695 MB baseline → 5890 MB peak
  Latency: 0.89s ± 0.08s (P95: 1.02s)
  RTF: 0.180 | Throughput: 5.55x realtime
  Tokens/sec: 172.3

Batch Sequential (10 requests):
  Total time: 8.50s
  Total audio: 45.00s
  Throughput: 5.29x realtime
  Requests/sec: 1.18
```

---

## RunPod Deployment

### Recommended Setup

- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU**: RTX 4090 (24GB) or A100/H100
- **Volume**: 50GB+ for models and cache

### Quick Start

```bash
# 1. Setup
curl -sSL https://raw.githubusercontent.com/fishaudio/fish-speech/main/tools/runpod/setup.sh | bash

# 2. Run benchmark
cd fish-speech
python tools/runpod/benchmark.py

# 3. Start API server
python tools/api_server.py --listen 0.0.0.0:8080
```

### Manual Setup

```bash
# Install system deps
apt-get update && apt-get install -y portaudio19-dev ffmpeg

# Clone and install
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e .
pip install torchao streamlit

# Download model
huggingface-cli download fishaudio/openaudio-s1-mini \
  --local-dir checkpoints/openaudio-s1-mini
```

### Production Configuration

```bash
# API server with compile enabled (default)
python tools/api_server.py \
  --listen 0.0.0.0:8080 \
  --workers 1 \
  --device cuda \
  --half

# With API key authentication
python tools/api_server.py \
  --listen 0.0.0.0:8080 \
  --api-key YOUR_SECRET_KEY
```

### Syncing Files to RunPod

```bash
# From local machine
rsync -avz --exclude '.git' --exclude '__pycache__' \
  ./fish-speech/ runpod:/workspace/fish-speech/
```

---

## SageMaker Deployment

### PyTorch 2.2.x Compatibility

SageMaker uses older PyTorch versions (2.2.x). We added compatibility shims:

```python
# In fish_speech/models/text2semantic/inference.py

# PyTorch version compatibility for attention backend
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    # Fallback for PyTorch < 2.5 (e.g., 2.2.x on SageMaker)
    from contextlib import contextmanager

    class SDPBackend:
        FLASH_ATTENTION = "flash"
        EFFICIENT_ATTENTION = "efficient"
        MATH = "math"

    @contextmanager
    def sdpa_kernel(backend):
        """Compatibility shim for PyTorch < 2.5"""
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            enable_flash = backend == SDPBackend.FLASH_ATTENTION
            enable_efficient = backend == SDPBackend.EFFICIENT_ATTENTION
            enable_math = backend == SDPBackend.MATH
            with torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_math=enable_math,
                enable_mem_efficient=enable_efficient,
            ):
                yield
        else:
            yield
```

### SageMaker Endpoint Configuration

```python
# inference.py for SageMaker
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

def model_fn(model_dir):
    """Load model for SageMaker."""
    engine = TTSInferenceEngine.from_pretrained(
        model_dir,
        compile=True,  # Enable torch.compile
        precision=torch.bfloat16,
    )
    return engine

def predict_fn(input_data, model):
    """Generate audio from text."""
    request = ServeTTSRequest(**input_data)
    for result in model.inference(request):
        if result.code == "final":
            return result.audio
```

### Instance Recommendations

| Instance | GPU | VRAM | Use Case |
|----------|-----|------|----------|
| ml.g5.xlarge | A10G | 24GB | Development |
| ml.g5.2xlarge | A10G | 24GB | Production |
| ml.p4d.24xlarge | A100 | 40GB | High throughput |

---

## VRAM Optimization

### Keeping Models in VRAM

The key to low latency is keeping models loaded. Our architecture:

```python
# In tools/server/model_manager.py
class ModelManager:
    """Keeps models loaded in VRAM across requests."""

    def __init__(self, ...):
        # Load once, keep forever
        self.llama_queue = launch_thread_safe_queue(...)
        self.decoder_model = load_dac_model(...)
        self.engine = TTSInferenceEngine(...)
```

### Cache Setup Optimization

KV cache is allocated once and reused:

```python
# In fish_speech/models/text2semantic/inference.py
def generate(...):
    # Only set up cache on first run
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True
```

### Memory Management

```python
# Avoid frequent VRAM clearing
# Only delete references, let Python GC handle cleanup
del logits, hidden_states, forward_result

# For cleanup between configurations (benchmarking only)
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
```

### Pre-warming

Always warm up before serving traffic:

```python
# Dry run to allocate caches and compile kernels
list(engine.inference(
    ServeTTSRequest(text="Hello world.", max_new_tokens=256)
))
```

---

## Troubleshooting

### Slow Inference (~12 tokens/sec instead of 100+)

**Cause**: SDPA backend misconfiguration or compile disabled.

**Fix**: Ensure compile is enabled and check backend:

```python
# Verify compile is enabled
python tools/run_webui.py  # Default: compiled

# Check logs for:
# "torch.compile cache directory: ~/.cache/fish_speech/torch_compile"
```

### TLS Assertion Error in Concurrent Code

**Cause**: CUDA graphs incompatible with threading.

**Error**:
```
RuntimeError: CUDA graphs does not support changing TLS state
```

**Fix**: Use sequential processing instead of threading.

### torchao Import Error

**Cause**: PyTorch/torchao version mismatch.

**Fix**:
```bash
pip install torchao>=0.5.0
# Or check compatibility
python -c "from torchao.quantization import quantize_, Int8WeightOnlyConfig"
```

### Slow First Run (30-60s warmup)

**Expected**: First run compiles kernels. Subsequent runs use cache.

**Verify cache**:
```bash
ls -la ~/.cache/fish_speech/torch_compile/
```

### Out of VRAM

**Solutions**:
1. Enable INT8 quantization (50% reduction)
2. Reduce max_new_tokens
3. Use smaller batch sizes

```bash
# Check VRAM usage
nvidia-smi

# Clear VRAM
python -c "import torch; torch.cuda.empty_cache()"
```

### Model Not Found

```bash
# Download model
huggingface-cli download fishaudio/openaudio-s1-mini \
  --local-dir checkpoints/openaudio-s1-mini
```

---

## Files Reference

### Core Inference

| File | Purpose |
|------|---------|
| `fish_speech/models/text2semantic/inference.py` | LLaMA inference, torch.compile, kernel caching |
| `fish_speech/models/dac/inference.py` | DAC decoder loading, INT8 quantization |
| `fish_speech/inference_engine.py` | High-level TTS engine |

### Deployment

| File | Purpose |
|------|---------|
| `tools/api_server.py` | Production API server |
| `tools/run_webui.py` | Gradio web interface |
| `tools/server/model_manager.py` | VRAM-persistent model loading |

### Benchmarking

| File | Purpose |
|------|---------|
| `tools/runpod/benchmark.py` | Comprehensive benchmark suite |
| `tools/runpod/load_test.py` | Production load simulation |
| `tools/runpod/setup.sh` | RunPod quick setup |

### Configuration

| File | Purpose |
|------|---------|
| `tools/server/api_utils.py` | CLI argument parsing |
| `fish_speech/configs/modded_dac_vq.yaml` | DAC model config |

---

## Summary

### Production Checklist

- [ ] torch.compile enabled (default)
- [ ] Kernel cache directory configured
- [ ] Models pre-warmed before serving
- [ ] INT8 quantization if VRAM-constrained
- [ ] Sequential processing (no threading)
- [ ] API authentication configured
- [ ] Monitoring VRAM usage

### Performance Targets

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| Tokens/sec | 100+ | torch.compile enabled |
| RTF | < 0.3 | Compiled + INT8 |
| Warmup | < 5s | Kernel cache warm |
| VRAM | < 4GB baseline | INT8 + DAC INT8 |
