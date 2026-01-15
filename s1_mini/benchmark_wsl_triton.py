"""
WSL2 Triton Benchmark for S1-Mini-like Model.

This script tests torch.compile with inductor backend (Triton)
on WSL2 to measure the expected performance improvement.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


class DualARTransformerBlock(nn.Module):
    """Simplified transformer block matching S1-Mini architecture."""

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
        hidden_dim = ((hidden_dim + 255) // 256) * 256
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
    """Model matching S1-Mini architecture for benchmarking."""

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


def benchmark_model(model, input_tensor, num_iterations=100, warmup=20, name="Model"):
    """Benchmark a model."""
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

    avg_time = elapsed / num_iterations
    print(f"  {name}: {avg_time*1000:.3f}ms per iteration")
    return avg_time


def main():
    print("=" * 70)
    print("WSL2 Triton Benchmark for S1-Mini-like Model")
    print("=" * 70)

    print(f"\nPyTorch: {torch.__version__}")

    # Check Triton
    try:
        import triton
        print(f"Triton: {triton.__version__}")
        triton_available = True
    except ImportError:
        print("Triton: NOT AVAILABLE")
        triton_available = False

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Test configurations
    configs = [
        {"dim": 1536, "heads": 24, "layers": 8, "seq_len": 128, "name": "8 layers, seq=128"},
        {"dim": 1536, "heads": 24, "layers": 8, "seq_len": 256, "name": "8 layers, seq=256"},
        {"dim": 1536, "heads": 24, "layers": 16, "seq_len": 128, "name": "16 layers, seq=128"},
    ]

    results = []

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Config: {config['name']}")
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

        # Benchmark eager mode
        print("\n1. PyTorch Eager Mode:")
        eager_time = benchmark_model(model, input_tensor, name="Eager")

        # Benchmark with torch.compile (inductor backend using Triton)
        if triton_available:
            print("\n2. torch.compile with Inductor (Triton):")
            try:
                model_compiled = torch.compile(model, backend="inductor", mode="max-autotune")

                # Extra warmup for compilation
                print("  Compiling (first few runs trigger JIT)...")
                with torch.inference_mode():
                    for _ in range(5):
                        _ = model_compiled(input_tensor)
                    torch.cuda.synchronize()

                compiled_time = benchmark_model(model_compiled, input_tensor, name="Compiled")

                speedup = eager_time / compiled_time
                print(f"\n  Speedup: {speedup:.2f}x")

                results.append({
                    "config": config['name'],
                    "eager_ms": eager_time * 1000,
                    "compiled_ms": compiled_time * 1000,
                    "speedup": speedup,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print("\n2. torch.compile: Skipped (Triton not available)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n{'Config':<25} | {'Eager':>12} | {'Compiled':>12} | {'Speedup':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<25} | {r['eager_ms']:>10.2f}ms | {r['compiled_ms']:>10.2f}ms | {r['speedup']:>9.2f}x")

        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print("-" * 70)
        print(f"{'Average Speedup':<25} | {'':<12} | {'':<12} | {avg_speedup:>9.2f}x")

        print("\n" + "=" * 70)
        print("PROJECTED TTS PERFORMANCE")
        print("=" * 70)

        # Windows baseline RTF
        windows_rtf = 0.51

        # Project new RTF with Triton speedup
        # Assuming transformer is ~70% of total time
        transformer_fraction = 0.7
        projected_rtf = windows_rtf / (1 - transformer_fraction + transformer_fraction / avg_speedup)

        print(f"\n  Windows RTF (without Triton): {windows_rtf:.2f}x")
        print(f"  Triton speedup on transformer: {avg_speedup:.2f}x")
        print(f"  Projected WSL2 RTF:            {projected_rtf:.2f}x")

        if projected_rtf >= 1.0:
            print(f"\n  RESULT: WSL2 with Triton should achieve REAL-TIME performance!")
        else:
            print(f"\n  RESULT: WSL2 with Triton improves performance significantly.")

    print("=" * 70)


if __name__ == "__main__":
    main()
