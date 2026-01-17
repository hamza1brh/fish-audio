# Fish-Speech RunPod Deployment Guide

Deploy Fish-Speech TTS on RunPod for production inference testing.

> **See Also**: [Full Production Deployment Guide](../../docs/PRODUCTION_DEPLOYMENT.md) - Comprehensive documentation covering torch.compile, quantization, batch inference, kernel caching, and SageMaker deployment.

## Target Environment

- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU**: RTX 4090 (24GB VRAM) - recommended
- **PyTorch**: 2.4.0 (fully compatible)

## Quick Start (5 Commands)

```bash
# 1. Clone repo
git clone https://github.com/fishaudio/fish-speech.git && cd fish-speech

# 2. Install dependencies
pip install -e . && pip install torchao streamlit

# 3. Download model
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# 4. Run benchmark
python tools/runpod/benchmark.py

# 5. (Optional) Start Streamlit UI
streamlit run tools/testing/streamlit_quant_test.py --server.port 8501 --server.address 0.0.0.0
```

Or use the setup script:
```bash
curl -sSL https://raw.githubusercontent.com/fishaudio/fish-speech/main/tools/runpod/setup.sh | bash
```

## Available Tools

### 1. Benchmark (`benchmark.py`)

Comprehensive performance testing across different quantization configurations.

```bash
python tools/runpod/benchmark.py
```

**What it tests:**
- BF16 (baseline)
- BF16 + DAC INT8
- INT8 Runtime
- INT8 Runtime + DAC INT8

**Output:**
- VRAM usage (baseline and peak)
- Generation latency
- Real-Time Factor (RTF)
- Throughput (x real-time)
- `benchmark_results.json` with all metrics

### 2. Load Test (`load_test.py`)

Simulates production load with sustained traffic.

```bash
# Basic test (60 seconds)
python tools/runpod/load_test.py --duration 60

# With concurrency
python tools/runpod/load_test.py --duration 120 --concurrency 2

# Rate-limited
python tools/runpod/load_test.py --duration 60 --rps 0.5
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | 60 | Test duration in seconds |
| `--concurrency` | 1 | Number of worker threads |
| `--rps` | None | Target requests per second |
| `--no-int8` | False | Disable LLaMA INT8 |
| `--no-dac-int8` | False | Disable DAC INT8 |
| `--output` | load_test_results.json | Output file |

**Output metrics:**
- Requests/minute
- Audio minutes/minute
- Latency percentiles (p50, p95, p99)
- Success/failure rates
- Peak VRAM usage

### 3. Streamlit UI

Interactive web interface for testing.

```bash
# Local access only
streamlit run tools/testing/streamlit_quant_test.py

# External access (RunPod)
streamlit run tools/testing/streamlit_quant_test.py --server.port 8501 --server.address 0.0.0.0
```

**Features:**
- Select quantization level (BF16, INT8, INT4)
- DAC INT8 toggle
- Voice cloning (zero-shot)
- Real-time VRAM monitoring
- Audio download

## Expected Performance (RTX 4090)

### VRAM Usage

| Configuration | Baseline | Peak (during generation) |
|---------------|----------|--------------------------|
| BF16 | ~3.7 GB | ~6-8 GB |
| BF16 + DAC INT8 | ~2.5 GB | ~5-7 GB |
| INT8 + DAC INT8 | ~1.9 GB | ~4-6 GB |

### Throughput

| Metric | Expected Value |
|--------|----------------|
| RTF | 0.3-0.5 (faster than real-time) |
| Throughput | 2-3x real-time |
| Requests/minute | 30-60 |
| Audio minutes/minute | 15-30 |

### Latency

| Percentile | Expected |
|------------|----------|
| p50 | 1-2s |
| p95 | 2-4s |
| p99 | 3-5s |

*Values depend on text length and model configuration.*

## Quantization Guide

### Recommended: INT8 + DAC INT8

Best balance of quality and VRAM efficiency:
- LLaMA: ~1.1 GB (vs 1.8 GB BF16)
- DAC: ~0.66 GB (vs 1.87 GB BF16)
- Total: ~1.9 GB baseline
- No perceptible quality loss

### Maximum Quality: BF16

Use when quality is paramount:
- Full precision weights
- ~3.7 GB baseline VRAM
- RTX 4090 has plenty of headroom

### Maximum VRAM Savings: INT4 + DAC INT8

Use for smaller GPUs (< 8GB):
- ~1.5 GB baseline
- Slight quality degradation
- Not recommended for production

## Troubleshooting

### "CUDA out of memory"

1. Enable INT8 quantization:
   ```bash
   python tools/runpod/benchmark.py  # Uses INT8 by default
   ```

2. Reduce max_new_tokens in generation settings

3. Clear VRAM between tests:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### "Module not found: torchao"

```bash
pip install torchao>=0.15.0
```

### "Checkpoint not found"

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

### Slow first inference

First inference includes:
- Model compilation (if enabled)
- KV cache allocation
- Warmup

Subsequent inferences will be faster. The benchmark script handles warmup automatically.

### Streamlit not accessible externally

Use `--server.address 0.0.0.0`:
```bash
streamlit run tools/testing/streamlit_quant_test.py --server.port 8501 --server.address 0.0.0.0
```

Then access via RunPod's HTTP port forwarding.

## Files Reference

```
tools/runpod/
├── README.md           # This file
├── setup.sh            # One-command setup script
├── benchmark.py        # Performance benchmarking
└── load_test.py        # Production load simulation

tools/testing/
├── streamlit_quant_test.py    # Interactive Streamlit UI
├── vram_breakdown.py          # Detailed VRAM analysis
└── test_combined_quantization.py  # Quantization testing
```

## Production Deployment Tips

1. **Use INT8 + DAC INT8** for production - best quality/VRAM ratio

2. **Pre-warm the model** before serving traffic:
   ```python
   # Run a dummy request after loading
   for _ in engine.inference(ServeTTSRequest(text="warmup")):
       pass
   ```

3. **Monitor VRAM** during load testing to ensure headroom

4. **Set reasonable timeouts** - generation can take 2-5s for longer texts

5. **Queue requests** rather than parallel processing - the model processes one request at a time internally

## License

Same as Fish-Speech main repository.
