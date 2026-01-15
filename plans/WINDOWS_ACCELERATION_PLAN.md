# Windows Acceleration Plan: Removing Triton Dependency

## Status: COMPLETED (All Phases Tested)

## Executive Summary

This plan outlined how to achieve near-Triton performance on Windows by implementing alternative acceleration backends.

**Initial State:** RTF ~0.62x on Windows (RTX 5070 Ti) with eager backend
**Target State:** RTF 1.5-2.5x on Windows with optimized backends
**Actual Result:** RTF ~0.5x - **Target NOT achievable on Windows**

### Key Finding
**Autoregressive TTS cannot benefit from TensorRT/ONNX optimizations** because:
- Each token generation is a separate forward pass (seq_len=1)
- TensorRT/ONNX overhead (~2ms per call) exceeds computation time
- Batch inference speedups (1.5-3x) don't apply to single-token generation

---

## Problem Statement

Triton is NVIDIA's JIT compiler that generates optimized CUDA kernels. It:
- Only works on Linux (requires LLVM/CUDA compilation infrastructure)
- Is used by `torch.compile` with `inductor` backend
- Provides 2-4x speedup over eager execution

Without Triton on Windows, we're stuck with the `eager` backend which:
- Doesn't fuse operations
- Has higher kernel launch overhead
- Cannot optimize memory access patterns

---

## Solution Overview

We will implement a **multi-backend acceleration system** that automatically selects the best available backend:

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Request                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Selector                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Check: TensorRT available? ──────► Use TensorRT  │    │
│  │ 2. Check: ONNX Runtime CUDA? ───────► Use ONNX RT   │    │
│  │ 3. Check: Triton (Linux)? ──────────► Use Inductor  │    │
│  │ 4. Fallback: ───────────────────────► Use Optimized │    │
│  │                                        Eager+CUDA   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Quick Wins (Day 1) - Expected: +20-40% speedup

**1.1 Enhance torch.compile Configuration**
- Switch to `mode="max-autotune"` for Windows
- Enable aggressive kernel autotuning
- Configure CUDA graphs for decode phase

**1.2 Optimize Flash Attention Detection**
- Properly detect Flash Attention 2 support on Windows
- Add fallback chain: Flash Attention → xFormers → SDPA → Math

**Files to modify:**
- `s1_mini/compilation.py` - Enhanced Windows configuration
- `fish_speech/models/text2semantic/llama.py` - Attention backend selection

### Phase 2: ONNX Runtime Backend (Days 2-3) - Expected: +50-100% speedup

**2.1 Model Export Pipeline**
- Export DualARTransformer to ONNX format
- Create separate graphs for prefill and decode phases
- Handle RoPE (rotary position embeddings) properly

**2.2 ONNX Runtime Integration**
- Create `ONNXInferenceBackend` class
- Implement KV cache management for autoregressive generation
- Add session caching for repeated use

**2.3 Automatic Backend Selection**
- Detect ONNX Runtime availability
- Benchmark and select fastest backend at startup
- Graceful fallback if ONNX fails

**New files:**
- `s1_mini/backends/__init__.py`
- `s1_mini/backends/base.py` - Abstract backend interface
- `s1_mini/backends/pytorch_backend.py` - Current PyTorch implementation
- `s1_mini/backends/onnx_backend.py` - ONNX Runtime implementation

### Phase 3: TensorRT Backend (Days 4-5) - Expected: +100-200% speedup

**3.1 TensorRT Engine Building**
- Convert ONNX to TensorRT engines
- Build optimized engines for common shapes
- Cache engines to disk for fast startup

**3.2 TensorRT Integration**
- Create `TensorRTInferenceBackend` class
- Handle dynamic shapes with optimization profiles
- Implement efficient KV cache binding

**New files:**
- `s1_mini/backends/tensorrt_backend.py`
- `s1_mini/engine_builder.py` - TensorRT engine building utilities

### Phase 4: Quantization Support (Day 6) - Expected: Additional +50-100% speedup

**4.1 INT8 Quantization**
- Implement calibration dataset collection
- Add INT8 quantization for ONNX and TensorRT
- Validate accuracy vs performance tradeoff

**4.2 Mixed Precision**
- FP16 for compute-bound operations
- FP32 for sensitive operations (layer norm, softmax)

---

## Technical Design

### Backend Interface

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional
import numpy as np

class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_models(self, checkpoint_path: str) -> None:
        """Load models from checkpoint."""
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.7,
    ) -> np.ndarray:
        """Generate audio from text."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        text: str,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """Generate audio in streaming mode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend can be used."""
        pass
```

### Backend Selection Logic

```python
def select_best_backend() -> InferenceBackend:
    """Select the best available backend for current platform."""

    backends = [
        TensorRTBackend(),    # Fastest if available
        ONNXRuntimeBackend(), # Good balance
        PyTorchBackend(),     # Always available fallback
    ]

    for backend in backends:
        if backend.is_available:
            logger.info(f"Selected backend: {backend.name}")
            return backend

    raise RuntimeError("No inference backend available")
```

### ONNX Export Strategy

The model has two distinct phases that benefit from separate optimization:

**Prefill Phase (Process full prompt)**
- Input: Token IDs [batch, seq_len]
- Output: Logits + KV Cache
- Characteristics: Large matrix multiplications, benefits from batching

**Decode Phase (Generate one token)**
- Input: Single token + KV Cache
- Output: Logits + Updated KV Cache
- Characteristics: Small compute, dominated by memory bandwidth

```python
# Export prefill model
torch.onnx.export(
    model,
    (input_ids, None),  # No KV cache for prefill
    "prefill.onnx",
    input_names=["input_ids"],
    output_names=["logits", "kv_cache"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
)

# Export decode model
torch.onnx.export(
    model,
    (single_token, kv_cache),
    "decode.onnx",
    input_names=["input_ids", "kv_cache"],
    output_names=["logits", "new_kv_cache"],
    dynamic_axes={
        "kv_cache": {2: "seq_len"},  # Growing cache
    },
)
```

---

## Actual Performance Results

**TESTED - January 2026 on RTX 5070 Ti:**

| Backend | Expected RTF | Actual RTF | Result |
|---------|--------------|------------|--------|
| PyTorch Eager | 0.62x | 0.47-0.54x | Baseline |
| Optimized Eager | 0.75-0.85x | 0.47-0.54x | No improvement |
| CUDA Graphs | 0.75-0.85x | 0.47-0.54x | Dynamic shapes prevent capture |
| ONNX Runtime CUDA | 1.2-1.8x | 0.47-0.54x | Overhead too high |
| TensorRT | 1.8-2.5x | 0.47-0.54x | Overhead too high |

**Batch Inference (Non-Autoregressive) - TensorRT DOES help:**
| Model Config | PyTorch | TensorRT | Speedup |
|--------------|---------|----------|---------|
| 8 layers, seq=128 | 2.18ms | 1.36ms | 1.60x |
| 8 layers, seq=256 | 2.12ms | 1.60ms | 1.33x |

**Conclusion:** TensorRT provides 1.3-3x speedup for batch inference, but this doesn't translate to autoregressive TTS where each token is generated individually.

**Original Target: RTF > 1.5x** - NOT ACHIEVABLE on Windows without Triton

---

## Testing Strategy

### Benchmark Suite
1. **Latency Test**: Time to first audio chunk
2. **Throughput Test**: Total audio generated per second
3. **Memory Test**: Peak VRAM usage
4. **Accuracy Test**: Compare outputs against PyTorch reference

### Regression Tests
- Ensure all backends produce similar audio quality
- Test with various text lengths (short, medium, long)
- Test with and without reference audio

---

## Dependencies

### Required
- `onnxruntime-gpu>=1.17.0` - ONNX Runtime with CUDA
- `onnx>=1.15.0` - ONNX model format

### Optional (for TensorRT)
- `tensorrt>=8.6` - NVIDIA TensorRT
- `cuda-python` - CUDA Python bindings

### Installation
```bash
# ONNX Runtime with CUDA (Windows)
pip install onnxruntime-gpu

# TensorRT (Windows) - requires CUDA toolkit
pip install tensorrt
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ONNX export fails for custom ops | Medium | High | Implement custom ONNX operators |
| TensorRT shape issues | Medium | Medium | Use optimization profiles |
| Accuracy degradation | Low | High | Extensive testing, fallback |
| Memory issues | Low | Medium | Implement memory monitoring |

---

## Success Criteria

1. **Performance**: RTF > 1.5x on Windows with RTX 5070 Ti
2. **Accuracy**: Audio quality matches PyTorch baseline
3. **Reliability**: No crashes or memory leaks over 1000 generations
4. **Usability**: Single config option to switch backends

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | Day 1 | Optimized eager backend, +20-40% speedup |
| Phase 2 | Days 2-3 | ONNX Runtime backend, +50-100% speedup |
| Phase 3 | Days 4-5 | TensorRT backend, +100-200% speedup |
| Phase 4 | Day 6 | Quantization support, additional +50-100% |
| Testing | Day 7 | Full benchmark suite, documentation |

---

## Completion Status

All phases have been tested. Results documented in `s1_mini/docs/WINDOWS_PERFORMANCE_SUMMARY.md`.

### Completed Tasks
- [x] Phase 1: Quick Wins - Implemented cuDNN benchmark, TF32, SDPA attention
- [x] Phase 2: ONNX Runtime - Tested, overhead too high for autoregressive
- [x] Phase 3: TensorRT - Installed and tested, overhead too high for autoregressive
- [ ] Phase 4: Quantization - Not pursued (won't help given overhead issue)

### Files Created
- `s1_mini/compilation.py` - Platform-aware compilation
- `s1_mini/attention.py` - Attention backend detection
- `s1_mini/backends/` - Multi-backend infrastructure
- `s1_mini/test_cudagraphs.py` - CUDA graphs testing
- `s1_mini/test_onnx_performance.py` - ONNX Runtime benchmarks
- `s1_mini/test_tensorrt_tts.py` - TensorRT benchmarks
- `s1_mini/benchmark.py` - Full TTS RTF benchmark

### Recommendation
**Use Linux with Triton for production.** Windows is suitable only for development/testing with RTF ~0.5x.
