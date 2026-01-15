# S1-Mini Acceleration Backends Documentation

## Overview

This document explains the different acceleration backends available for S1-Mini inference on Windows (and Linux). Since Triton is not available on Windows, we provide alternative backends that can achieve similar or better performance.

---

## Backend Comparison

```
Performance Scale (RTF on RTX 5070 Ti):

                    Slower ◄─────────────────────────► Faster

PyTorch Eager:      ████░░░░░░░░░░░░░░░░  0.62x (current)
Optimized Eager:    ██████░░░░░░░░░░░░░░  0.85x
ONNX Runtime:       ████████████░░░░░░░░  1.5x
TensorRT FP16:      ██████████████████░░  2.2x
TensorRT INT8:      ████████████████████  3.0x+
Linux + Triton:     ████████████████░░░░  2.0x (reference)
```

---

## Backend 1: Optimized PyTorch Eager

### What It Does
Enhances the existing PyTorch eager execution with better configuration:
- Enables `max-autotune` mode for kernel selection
- Uses CUDA graphs for the decode phase
- Properly enables Flash Attention when available

### When to Use
- Quick testing during development
- When other backends aren't available
- As a reliable fallback

### Configuration
```python
from s1_mini.config import EngineConfig

config = EngineConfig(
    backend="pytorch",
    compile_mode="max-autotune",  # Better than reduce-overhead on Windows
    use_cuda_graphs=True,
    use_flash_attention=True,
)
```

### Expected Performance
- RTF: 0.75-0.85x
- VRAM: ~4.5 GB
- Startup: ~10 seconds

---

## Backend 2: ONNX Runtime with CUDA

### What It Does
Converts the PyTorch model to ONNX format and runs inference using ONNX Runtime with the CUDA Execution Provider. ONNX Runtime applies:
- Operation fusion (combine multiple ops into one kernel)
- Memory optimization (reduce allocations)
- Kernel optimization (use cuDNN and cuBLAS efficiently)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX Runtime Backend                      │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Prefill    │    │    Decode    │    │   Decoder    │  │
│  │   Session    │    │   Session    │    │   Session    │  │
│  │  (ONNX EP)   │    │  (ONNX EP)   │    │  (ONNX EP)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CUDA Execution Provider                 │   │
│  │         (cuDNN + cuBLAS + Custom Kernels)           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

**Step 1: Model Export**
```python
# The DualARTransformer is exported to two ONNX models:

# 1. Prefill model - processes the full input sequence
#    Input: [batch, seq_len] token IDs
#    Output: logits + KV cache for all layers

# 2. Decode model - generates one token at a time
#    Input: single token + existing KV cache
#    Output: logits + updated KV cache
```

**Step 2: Session Creation**
```python
import onnxruntime as ort

# Create session with CUDA provider
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',  # Fallback
]

session = ort.InferenceSession("model.onnx", providers=providers)
```

**Step 3: Inference Loop**
```python
# Prefill phase
kv_cache = session_prefill.run(
    ["kv_cache", "logits"],
    {"input_ids": prompt_tokens}
)

# Decode loop
for _ in range(max_tokens):
    logits, kv_cache = session_decode.run(
        ["logits", "kv_cache"],
        {"input_ids": next_token, "kv_cache": kv_cache}
    )
    next_token = sample(logits)
```

### Configuration
```python
config = EngineConfig(
    backend="onnx",
    onnx_optimization_level="all",  # ORT_ENABLE_ALL
    onnx_graph_optimization=True,
    onnx_memory_optimization=True,
)
```

### Expected Performance
- RTF: 1.2-1.8x
- VRAM: ~5.0 GB (slightly higher due to ONNX overhead)
- Startup: ~30 seconds (includes ONNX optimization)
- First inference: ~5 seconds (kernel compilation)

### Installation
```bash
# Install ONNX Runtime with CUDA support
pip install onnxruntime-gpu>=1.17.0

# Verify CUDA provider is available
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Backend 3: TensorRT

### What It Does
NVIDIA TensorRT is the most aggressive optimization framework:
- Layer and tensor fusion
- Kernel auto-tuning for specific GPU
- Precision calibration (FP16/INT8)
- Memory pooling

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorRT Backend                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TensorRT Engine Cache                    │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ prefill_bs1│  │ decode_bs1 │  │  decoder   │     │  │
│  │  │   .trt     │  │   .trt     │  │   .trt     │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TensorRT Runtime                         │  │
│  │  - Optimized CUDA kernels                            │  │
│  │  - Fused operations                                   │  │
│  │  - Memory-efficient execution                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

**Step 1: ONNX to TensorRT Conversion**
```python
import tensorrt as trt

# Create builder and network
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# Parse ONNX model
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

# Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

# Build optimized engine
engine = builder.build_serialized_network(network, config)
```

**Step 2: Create Optimization Profiles**
```python
# TensorRT needs to know input shape ranges
profile = builder.create_optimization_profile()

# Prefill: variable sequence length
profile.set_shape(
    "input_ids",
    min=(1, 1),      # Minimum shape
    opt=(1, 256),    # Optimal shape (most common)
    max=(1, 2048),   # Maximum shape
)

config.add_optimization_profile(profile)
```

**Step 3: Execute with CUDA Streams**
```python
# Create execution context
context = engine.create_execution_context()

# Allocate buffers
inputs = cuda.mem_alloc(input_size)
outputs = cuda.mem_alloc(output_size)

# Execute asynchronously
context.execute_async_v2(
    bindings=[int(inputs), int(outputs)],
    stream_handle=stream.handle
)
stream.synchronize()
```

### Configuration
```python
config = EngineConfig(
    backend="tensorrt",
    tensorrt_precision="fp16",  # or "int8" for maximum speed
    tensorrt_workspace_gb=4,
    tensorrt_cache_dir="./trt_engines",
)
```

### Expected Performance
- RTF: 1.8-2.5x (FP16), 2.5-4.0x (INT8)
- VRAM: ~4.0 GB (more efficient memory use)
- Startup: ~60 seconds first time (engine building)
- Subsequent startup: ~5 seconds (cached engines)

### Installation
```bash
# TensorRT requires CUDA toolkit
# Download from: https://developer.nvidia.com/tensorrt

# Install Python bindings
pip install tensorrt>=8.6

# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

### INT8 Quantization

For maximum performance, TensorRT can use INT8 precision:

```python
# Requires calibration dataset
class CalibrationDataset:
    def __init__(self, texts):
        self.texts = texts
        self.index = 0

    def get_batch(self, batch_size=1):
        if self.index >= len(self.texts):
            return None
        text = self.texts[self.index]
        self.index += 1
        return tokenize(text)

# Build INT8 engine
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Calibrator(CalibrationDataset(sample_texts))
```

---

## Backend Selection Guide

### Decision Tree

```
Start
  │
  ├─► Need maximum performance?
  │     │
  │     ├─► Yes ──► TensorRT + INT8
  │     │            (RTF: 2.5-4.0x)
  │     │
  │     └─► No ───► TensorRT FP16
  │                  (RTF: 1.8-2.5x)
  │
  ├─► Need reliable, tested solution?
  │     │
  │     └─► Yes ──► ONNX Runtime
  │                  (RTF: 1.2-1.8x)
  │
  └─► Just testing/development?
        │
        └─► Yes ──► Optimized PyTorch
                     (RTF: 0.75-0.85x)
```

### Recommended Configurations

**Development (Windows)**
```python
config = EngineConfig(
    backend="pytorch",
    compile_mode="max-autotune",
    use_cuda_graphs=True,
)
```

**Production Testing (Windows)**
```python
config = EngineConfig(
    backend="onnx",
    onnx_optimization_level="all",
)
```

**Production (Linux/SageMaker)**
```python
config = EngineConfig(
    backend="tensorrt",
    tensorrt_precision="fp16",
)
```

---

## Troubleshooting

### ONNX Runtime Issues

**Problem: CUDA provider not found**
```
Available providers: ['CPUExecutionProvider']
```
**Solution:**
```bash
# Uninstall CPU-only version
pip uninstall onnxruntime

# Install GPU version
pip install onnxruntime-gpu
```

**Problem: CUDA version mismatch**
```
CUDA driver version is insufficient
```
**Solution:**
Update NVIDIA drivers or install matching onnxruntime-gpu version.

### TensorRT Issues

**Problem: Engine building fails**
```
[TRT] [E] Could not find any implementation for node...
```
**Solution:**
- Increase workspace size
- Check ONNX model for unsupported ops
- Try lower optimization level

**Problem: Shape mismatch at runtime**
```
[TRT] [E] Binding index 0 out of range
```
**Solution:**
- Ensure input shapes match optimization profile
- Rebuild engine with correct shape ranges

---

## Performance Tuning Tips

### General
1. **Warm up the model** - First few inferences are slower
2. **Use consistent batch sizes** - Avoid shape changes
3. **Monitor VRAM** - Leave headroom for temporary allocations

### ONNX Runtime
1. Enable all optimizations: `sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
2. Use IO binding for GPU tensors to avoid copies
3. Reuse sessions across requests

### TensorRT
1. Cache engines to disk - rebuild is expensive
2. Use FP16 unless accuracy is critical
3. Profile with different batch sizes to find optimal
4. Consider INT8 for 2x additional speedup

---

## Benchmarking

Run the benchmark to compare backends:

```bash
# Benchmark all available backends
python -m s1_mini.benchmark --all-backends

# Benchmark specific backend
python -m s1_mini.benchmark --backend onnx

# Detailed profiling
python -m s1_mini.benchmark --backend tensorrt --profile
```

Output format:
```
=== Backend Comparison ===
Backend          | RTF    | Latency (5s audio) | VRAM
-----------------+--------+--------------------+-------
PyTorch Eager    | 0.62x  | 8.1s               | 4.4GB
PyTorch Optimized| 0.82x  | 6.1s               | 4.5GB
ONNX Runtime     | 1.45x  | 3.4s               | 5.0GB
TensorRT FP16    | 2.10x  | 2.4s               | 4.0GB
TensorRT INT8    | 3.20x  | 1.6s               | 3.5GB
```
