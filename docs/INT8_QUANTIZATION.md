# INT8 Quantization Guide

This guide explains how to quantize Fish-Speech models to INT8 for reduced VRAM usage, and how to benchmark the performance difference.

## Table of Contents

1. [Why Quantize?](#why-quantize)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Benchmark Comparison](#benchmark-comparison)
5. [Why Subprocesses for Benchmarking?](#why-subprocesses-for-benchmarking)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## Why Quantize?

### Benefits

| Aspect | BF16 | INT8 | Improvement |
|--------|------|------|-------------|
| VRAM Baseline | ~4.6 GB | ~2.5 GB | **46% reduction** |
| VRAM Peak | ~6.4 GB | ~4.2 GB | **34% reduction** |
| Model Size | ~1.8 GB | ~0.9 GB | **50% smaller** |
| Speed | Baseline | Similar | **No degradation** |
| Quality | Full precision | Negligible loss | **Imperceptible** |

### When to Use INT8

- **Small GPUs**: 8GB or less VRAM
- **Multi-model inference**: Run multiple models on one GPU
- **Cost optimization**: Use smaller/cheaper GPU instances
- **Edge deployment**: Reduced memory footprint

### When to Use BF16

- **Maximum quality**: Critical audio quality requirements
- **Plenty of VRAM**: RTX 4090 (24GB) or better
- **Development**: Debugging and experimentation

---

## Quick Start

### On RunPod (One Command)

```bash
cd /workspace/fish-audio
python tools/runpod/run_comparison.py --num-samples 10
```

This will:
1. Quantize the model to INT8 (if not already done)
2. Benchmark BF16 model
3. Clear VRAM (via subprocess isolation)
4. Benchmark INT8 model
5. Generate comparison report

### Output Files

```
comparison_report.html    # Visual comparison report
bf16_results.json        # BF16 benchmark data
int8_results.json        # INT8 benchmark data
```

---

## Detailed Workflow

### Step 1: Quantize the Model

```bash
# Creates: checkpoints/openaudio-s1-mini-int8-torchao-YYYYMMDD_HHMMSS/
python tools/llama/quantize.py \
    --checkpoint-path checkpoints/openaudio-s1-mini \
    --mode int8
```

Output:
```
Quantization Summary
============================================================
  Mode:              int8
  Original size:     1824.50 MB
  Quantized size:    912.25 MB
  Compression ratio: 2.0x
  Output path:       checkpoints/openaudio-s1-mini-int8-torchao-20260118_143052
```

### Step 2: Benchmark BF16

```bash
python tools/runpod/benchmark.py \
    --checkpoint checkpoints/openaudio-s1-mini \
    --num-samples 10
```

### Step 3: Clear VRAM

```python
# VRAM isn't fully freed in the same process!
# Must use subprocess or restart Python
import torch
torch.cuda.empty_cache()  # Not enough!
```

**Solution**: Run each benchmark in a separate subprocess (see below).

### Step 4: Benchmark INT8

```bash
python tools/runpod/benchmark.py \
    --checkpoint checkpoints/openaudio-s1-mini-int8-torchao-*/ \
    --num-samples 10
```

---

## Benchmark Comparison

### Automated Comparison Script

The `run_comparison.py` script handles everything:

```bash
# Standard test (10 samples per text = 40 total)
python tools/runpod/run_comparison.py

# Heavy test (20 samples per text = 80 total)
python tools/runpod/run_comparison.py --num-samples 20

# Skip quantization (use existing INT8 checkpoint)
python tools/runpod/run_comparison.py --skip-quantize
```

### From Local Machine (Deploy and Test)

If you have SSH access to RunPod configured:

```bash
# Setup SSH config first (~/.ssh/config):
# Host runpod
#     HostName <your-ip>
#     User root
#     Port <your-port>

# Run the deploy script
./tools/runpod/deploy_and_test.sh

# Heavy test
./tools/runpod/deploy_and_test.sh --heavy

# Custom samples
./tools/runpod/deploy_and_test.sh --samples 15
```

---

## Why Subprocesses for Benchmarking?

### The Problem

CUDA memory is **not fully released** when you delete tensors and call `torch.cuda.empty_cache()`:

```python
# This doesn't fully free VRAM!
del model
torch.cuda.empty_cache()
gc.collect()

# VRAM still shows ~2GB allocated
print(torch.cuda.memory_allocated())  # > 0
```

### The Reason

1. **CUDA context**: PyTorch maintains a CUDA context with cached allocations
2. **Compiled kernels**: torch.compile caches kernels in memory
3. **cuDNN workspace**: Allocated for convolution operations
4. **CUDA graphs**: Capture static memory allocations

### The Solution

Run each benchmark in a **separate subprocess**:

```python
import subprocess
import sys

# Subprocess exits → CUDA memory fully released
subprocess.run([sys.executable, "benchmark_bf16.py"])

# Now VRAM is truly at 0
subprocess.run([sys.executable, "benchmark_int8.py"])
```

The `run_comparison.py` script does this automatically by:
1. Writing a temporary benchmark script
2. Running it in a subprocess
3. Reading results from JSON file
4. Subprocess exits → VRAM cleared
5. Running next benchmark in new subprocess

---

## Expected Results

### RTX 4090 (24GB)

| Metric | BF16 | INT8 | Difference |
|--------|------|------|------------|
| VRAM Baseline | 4,602 MB | 2,500 MB | -46% |
| VRAM Peak | 6,440 MB | 4,200 MB | -35% |
| Tokens/sec | 120 | 115-125 | ±5% |
| RTF | 0.178 | 0.17-0.19 | Similar |
| Throughput | 5.6x | 5.3-5.8x | Similar |

### RTX 3090 (24GB)

| Metric | BF16 | INT8 |
|--------|------|------|
| VRAM Baseline | ~4.6 GB | ~2.5 GB |
| Tokens/sec | ~80-100 | ~80-100 |

### RTX 4070 Ti (12GB)

| Metric | BF16 | INT8 |
|--------|------|------|
| VRAM Baseline | ~4.6 GB | ~2.5 GB |
| Can run? | Yes (tight) | Yes (comfortable) |

### RTX 3060 (12GB)

| Metric | BF16 | INT8 |
|--------|------|------|
| VRAM Baseline | ~4.6 GB | ~2.5 GB |
| Can run? | Risky | Yes |

---

## Troubleshooting

### "CUDA out of memory" during quantization

```bash
# Clear VRAM first
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Run quantization
python tools/llama/quantize.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int8
```

### Quantized model runs slower than BF16

This can happen if:
1. **Kernel compilation**: First run compiles new kernels for INT8
2. **Cache miss**: Clear cache and re-run

```bash
rm -rf ~/.cache/fish_speech/torch_compile/
python tools/runpod/run_comparison.py  # Will recompile
```

### torchao not available

```bash
pip install torchao>=0.5.0
```

Check compatibility:
```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig
# If this imports, you're good
```

### Benchmark shows VRAM not cleared

This happens if running multiple benchmarks in the same Python process. Use:

```bash
# Correct: Use subprocess isolation
python tools/runpod/run_comparison.py

# Incorrect: Running benchmark.py twice in same shell session
python tools/runpod/benchmark.py --checkpoint bf16_model
python tools/runpod/benchmark.py --checkpoint int8_model  # VRAM not cleared!
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `tools/llama/quantize.py` | Main quantization entry point |
| `tools/llama/quantize_torchao.py` | TorchAO quantization implementation |
| `tools/runpod/run_comparison.py` | Automated INT8 vs BF16 comparison |
| `tools/runpod/deploy_and_test.sh` | Sync to RunPod and run comparison |
| `tools/runpod/benchmark.py` | General benchmark (accepts --checkpoint) |

---

## Summary

1. **INT8 reduces VRAM by ~46%** with negligible quality loss
2. **Speed is similar** to BF16 (within ±5%)
3. **Use subprocesses** to ensure clean VRAM between benchmarks
4. **Run `run_comparison.py`** for automated comparison
5. **Check `comparison_report.html`** for visual results
