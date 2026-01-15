"""
Test ONNX Runtime performance with CUDA provider.

This script tests if ONNX Runtime can provide speedup over PyTorch
for transformer-like models on Windows.
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
import onnx
import onnxruntime as ort


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing ONNX export."""

    def __init__(self, dim=1024, heads=16, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Self attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

        # Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape

        # Self attention
        normed = self.norm1(x)
        q = self.q_proj(normed).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot product attention (using manual computation for ONNX compatibility)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(out)

        # Feed forward
        x = x + self.ff(self.norm2(x))

        return x


class StackedTransformer(nn.Module):
    """Stack of transformer blocks (similar to LLM architecture)."""

    def __init__(self, dim=1024, heads=16, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(dim, heads) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


def export_to_onnx(model, sample_input, output_path):
    """Export PyTorch model to ONNX using legacy exporter."""
    model.eval()

    # Use legacy exporter (dynamo=False) to avoid encoding issues
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
            dynamo=False,  # Use legacy exporter
        )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path


def create_onnx_session(onnx_path, provider='cuda'):
    """Create ONNX Runtime session with specified provider."""
    if provider == 'cuda':
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
    elif provider == 'tensorrt':
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
            }),
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider',
        ]
    else:
        providers = ['CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(onnx_path, sess_options, providers=providers)


def benchmark_pytorch(model, input_tensor, num_iterations=100, warmup=10):
    """Benchmark PyTorch model."""
    model.eval()

    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / num_iterations


def benchmark_onnx(session, input_numpy, num_iterations=100, warmup=10):
    """Benchmark ONNX Runtime session."""
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_numpy})

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: input_numpy})
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations


def main():
    print("=" * 70)
    print("ONNX Runtime Performance Test")
    print("=" * 70)

    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test configurations
    configs = [
        {"dim": 512, "heads": 8, "layers": 4, "seq_len": 64, "batch": 1},
        {"dim": 1024, "heads": 16, "layers": 4, "seq_len": 128, "batch": 1},
        {"dim": 1024, "heads": 16, "layers": 8, "seq_len": 256, "batch": 1},
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Config: dim={config['dim']}, heads={config['heads']}, "
              f"layers={config['layers']}, seq={config['seq_len']}")
        print("=" * 70)

        # Create model
        model = StackedTransformer(
            dim=config['dim'],
            heads=config['heads'],
            layers=config['layers'],
        ).cuda().half()

        # Create input
        input_tensor = torch.randn(
            config['batch'], config['seq_len'], config['dim'],
            device='cuda', dtype=torch.float16
        )

        # Benchmark PyTorch
        print("\n1. PyTorch (CUDA)...")
        pytorch_time = benchmark_pytorch(model, input_tensor)
        print(f"   Average time: {pytorch_time*1000:.3f}ms")

        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        print("\n2. Exporting to ONNX...")
        try:
            export_to_onnx(model.cpu().float(), input_tensor.cpu().float(), onnx_path)
            print(f"   Exported to {onnx_path}")
        except Exception as e:
            print(f"   ERROR: Export failed - {e}")
            continue

        # Prepare ONNX input
        input_numpy = input_tensor.cpu().float().numpy()

        # Benchmark ONNX CUDA
        print("\n3. ONNX Runtime (CUDA)...")
        try:
            session_cuda = create_onnx_session(onnx_path, 'cuda')
            onnx_cuda_time = benchmark_onnx(session_cuda, input_numpy)
            print(f"   Average time: {onnx_cuda_time*1000:.3f}ms")
            cuda_speedup = pytorch_time / onnx_cuda_time
            print(f"   Speedup vs PyTorch: {cuda_speedup:.2f}x")
        except Exception as e:
            print(f"   ERROR: {e}")
            onnx_cuda_time = None

        # Benchmark ONNX TensorRT
        print("\n4. ONNX Runtime (TensorRT)...")
        try:
            session_trt = create_onnx_session(onnx_path, 'tensorrt')
            onnx_trt_time = benchmark_onnx(session_trt, input_numpy)
            print(f"   Average time: {onnx_trt_time*1000:.3f}ms")
            trt_speedup = pytorch_time / onnx_trt_time
            print(f"   Speedup vs PyTorch: {trt_speedup:.2f}x")
        except Exception as e:
            print(f"   ERROR: {e}")
            onnx_trt_time = None

        # Summary for this config
        print(f"\n   SUMMARY:")
        print(f"   PyTorch:      {pytorch_time*1000:.3f}ms (baseline)")
        if onnx_cuda_time:
            print(f"   ONNX CUDA:    {onnx_cuda_time*1000:.3f}ms ({cuda_speedup:.2f}x)")
        if onnx_trt_time:
            print(f"   ONNX TRT:     {onnx_trt_time*1000:.3f}ms ({trt_speedup:.2f}x)")

        # Cleanup
        os.unlink(onnx_path)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
