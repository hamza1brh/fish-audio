"""
Test TensorRT performance for TTS model components.

This script benchmarks TensorRT execution provider for transformer
blocks similar to those used in the S1-Mini TTS model.
"""

import sys
import time
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ["S1_MINI_LOG_PLATFORM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


class DualARTransformerBlock(nn.Module):
    """
    Simplified transformer block matching S1-Mini architecture.

    The actual S1-Mini uses:
    - dim=1536, heads=24 for slow transformer
    - 32 layers (slow) + 4 layers (fast)
    """

    def __init__(self, dim=1536, heads=24, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Self attention (matching S1-Mini)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Feed forward (SwiGLU style)
        hidden_dim = int(2 * dim * ff_mult / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round to multiple of 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # RMSNorm
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        B, T, D = x.shape

        # Self attention with residual
        h = self.norm1(x)
        qkv = self.wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(out)

        # FFN with SwiGLU and residual
        h = self.norm2(x)
        x = x + self.w2(nn.functional.silu(self.w1(h)) * self.w3(h))

        return x


class S1MiniLikeModel(nn.Module):
    """
    Model matching S1-Mini architecture for benchmarking.

    S1-Mini specs:
    - 32 slow layers (dim=1536, heads=24)
    - 4 fast layers
    - ~500M parameters
    """

    def __init__(self, dim=1536, heads=24, layers=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DualARTransformerBlock(dim, heads) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def export_to_onnx(model, sample_input, output_path):
    """Export model to ONNX format."""
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch', 1: 'seq_len'},
                'output': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    return output_path


def create_trt_session(onnx_path):
    """Create ONNX Runtime session with TensorRT provider."""
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(Path(onnx_path).parent),
        }),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider',
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(onnx_path, sess_options, providers=providers)


def benchmark_pytorch(model, input_tensor, num_iterations=50, warmup=10):
    """Benchmark PyTorch model."""
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / num_iterations


def benchmark_onnx(session, input_numpy, num_iterations=50, warmup=10):
    """Benchmark ONNX Runtime session."""
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        _ = session.run(None, {input_name: input_numpy})

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: input_numpy})
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations


def main():
    print("=" * 70)
    print("TensorRT Performance Test for S1-Mini-like Model")
    print("=" * 70)

    print(f"\nONNX Runtime: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test configurations matching S1-Mini architecture
    configs = [
        # Single layer test
        {"dim": 1536, "heads": 24, "layers": 1, "seq_len": 128, "name": "1 layer"},
        # 4 layers (fast transformer)
        {"dim": 1536, "heads": 24, "layers": 4, "seq_len": 128, "name": "4 layers (fast)"},
        # 8 layers (subset of slow transformer)
        {"dim": 1536, "heads": 24, "layers": 8, "seq_len": 128, "name": "8 layers"},
        # 8 layers with longer sequence
        {"dim": 1536, "heads": 24, "layers": 8, "seq_len": 256, "name": "8 layers (seq=256)"},
    ]

    results = []

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Config: {config['name']} (dim={config['dim']}, layers={config['layers']}, seq={config['seq_len']})")
        print("=" * 70)

        # Create model
        model = S1MiniLikeModel(
            dim=config['dim'],
            heads=config['heads'],
            layers=config['layers'],
        ).cuda().half()

        # Create input
        input_tensor = torch.randn(
            1, config['seq_len'], config['dim'],
            device='cuda', dtype=torch.float16
        )

        # Benchmark PyTorch
        print("\n1. PyTorch (CUDA)...")
        pytorch_time = benchmark_pytorch(model, input_tensor)
        print(f"   Average time: {pytorch_time*1000:.3f}ms")

        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "model.onnx")

            print("\n2. Exporting to ONNX...")
            try:
                export_to_onnx(model.cpu().float(), input_tensor.cpu().float(), onnx_path)
                print(f"   Exported successfully")
            except Exception as e:
                print(f"   ERROR: Export failed - {e}")
                continue

            # Prepare input
            input_numpy = input_tensor.cpu().float().numpy()

            # Benchmark TensorRT
            print("\n3. ONNX Runtime (TensorRT)...")
            print("   Building TensorRT engine (this may take a moment)...")
            try:
                session_trt = create_trt_session(onnx_path)

                # Check which provider is actually being used
                providers_used = session_trt.get_providers()
                print(f"   Providers used: {providers_used}")

                trt_time = benchmark_onnx(session_trt, input_numpy)
                print(f"   Average time: {trt_time*1000:.3f}ms")
                trt_speedup = pytorch_time / trt_time
                print(f"   Speedup vs PyTorch: {trt_speedup:.2f}x")

                results.append({
                    "config": config['name'],
                    "pytorch_ms": pytorch_time * 1000,
                    "tensorrt_ms": trt_time * 1000,
                    "speedup": trt_speedup,
                })

            except Exception as e:
                print(f"   ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n{'Config':<25} | {'PyTorch':>12} | {'TensorRT':>12} | {'Speedup':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<25} | {r['pytorch_ms']:>10.2f}ms | {r['tensorrt_ms']:>10.2f}ms | {r['speedup']:>9.2f}x")

        avg_speedup = np.mean([r['speedup'] for r in results])
        print("-" * 70)
        print(f"{'Average Speedup':<25} | {'':<12} | {'':<12} | {avg_speedup:>9.2f}x")

    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR S1-MINI TTS")
    print("=" * 70)

    if results:
        # Estimate RTF improvement
        # Current RTF without TensorRT: ~0.60x
        # With TensorRT speedup applied to transformer forward passes
        current_rtf = 0.60
        # Transformer forward is ~70% of total generation time
        transformer_fraction = 0.7
        estimated_new_rtf = current_rtf / (1 - transformer_fraction + transformer_fraction / avg_speedup)

        print(f"\n  Current RTF (PyTorch):       ~{current_rtf:.2f}x")
        print(f"  Average TensorRT speedup:    {avg_speedup:.2f}x")
        print(f"  Estimated new RTF:           ~{estimated_new_rtf:.2f}x")

        if estimated_new_rtf >= 1.0:
            print(f"\n  RESULT: TensorRT should achieve REAL-TIME performance!")
        else:
            print(f"\n  RESULT: TensorRT improves but may not reach real-time.")
            print(f"          Need ~{1.0/current_rtf:.2f}x overall speedup for RTF=1.0")

    print("=" * 70)


if __name__ == "__main__":
    main()
