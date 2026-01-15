"""
Test CUDA Graphs backend for Windows acceleration.

CUDA Graphs can significantly reduce kernel launch overhead by
capturing and replaying entire execution graphs.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ["S1_MINI_LOG_PLATFORM"] = "false"

import torch


def test_cudagraphs_backend():
    """Test if cudagraphs backend works on this system."""
    print("=" * 70)
    print("Testing CUDA Graphs Backend")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 1024)
            self.linear2 = torch.nn.Linear(1024, 512)
            self.norm = torch.nn.LayerNorm(512)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return self.norm(x)

    model = SimpleModel().cuda().half()
    x = torch.randn(1, 64, 512, device="cuda", dtype=torch.float16)

    # Test without compile
    print("\n1. Testing WITHOUT torch.compile...")
    model.eval()
    with torch.inference_mode():
        # Warmup
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        torch.cuda.synchronize()
        no_compile_time = time.perf_counter() - start
        print(f"   100 iterations: {no_compile_time*1000:.2f}ms")

    # Test with cudagraphs backend
    print("\n2. Testing WITH torch.compile(backend='cudagraphs')...")
    try:
        model_compiled = torch.compile(model, backend="cudagraphs")
        model_compiled.eval()

        with torch.inference_mode():
            # Warmup (this triggers compilation)
            print("   Compiling...")
            for _ in range(5):
                _ = model_compiled(x)
            torch.cuda.synchronize()
            print("   Compilation done!")

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                _ = model_compiled(x)
            torch.cuda.synchronize()
            cudagraphs_time = time.perf_counter() - start
            print(f"   100 iterations: {cudagraphs_time*1000:.2f}ms")

        speedup = no_compile_time / cudagraphs_time
        print(f"\n   Speedup: {speedup:.2f}x")

        if speedup > 1.1:
            print("   CUDA Graphs provides meaningful speedup!")
            return True
        else:
            print("   CUDA Graphs does not provide significant speedup")
            return False

    except Exception as e:
        print(f"   ERROR: cudagraphs backend failed: {e}")
        return False


def test_eager_with_optimizations():
    """Test eager backend with aggressive optimizations."""
    print("\n" + "=" * 70)
    print("Testing Eager Backend with Optimizations")
    print("=" * 70)

    # Enable all optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Optimizations enabled:")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")

    # Test with a larger model (more representative)
    class TransformerBlock(torch.nn.Module):
        def __init__(self, dim=1024, heads=16):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 4),
                torch.nn.GELU(),
                torch.nn.Linear(dim * 4, dim),
            )
            self.norm1 = torch.nn.LayerNorm(dim)
            self.norm2 = torch.nn.LayerNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.ff(self.norm2(x))
            return x

    model = TransformerBlock().cuda().half()
    x = torch.randn(1, 128, 1024, device="cuda", dtype=torch.float16)

    model.eval()
    with torch.inference_mode():
        # Warmup
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            _ = model(x)
        torch.cuda.synchronize()
        base_time = time.perf_counter() - start
        print(f"\nBaseline (eager): {base_time*1000:.2f}ms for 50 iterations")

    # Test with torch.compile eager
    print("\nTesting torch.compile(backend='eager', mode='max-autotune')...")
    try:
        model_compiled = torch.compile(
            model,
            backend="eager",
            mode="max-autotune",
        )
        model_compiled.eval()

        with torch.inference_mode():
            # Warmup
            print("  Compiling...")
            for _ in range(10):
                _ = model_compiled(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                _ = model_compiled(x)
            torch.cuda.synchronize()
            compiled_time = time.perf_counter() - start
            print(f"  Compiled (eager): {compiled_time*1000:.2f}ms for 50 iterations")

        speedup = base_time / compiled_time
        print(f"  Speedup: {speedup:.2f}x")

    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    cudagraphs_works = test_cudagraphs_backend()
    test_eager_with_optimizations()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"CUDA Graphs backend: {'WORKING' if cudagraphs_works else 'NOT WORKING'}")
    print("\nTo install ONNX Runtime with CUDA support:")
    print("  pip uninstall onnxruntime")
    print("  pip install onnxruntime-gpu")
    print("=" * 70)
