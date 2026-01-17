"""
Quantization Comparison Test
============================

This script compares different precision/quantization approaches for the
Fish-Speech S1-Mini model:

1. float16 (baseline)
2. bfloat16
3. int8 (weight-only quantization)
4. int4 (weight-only quantization, group size 128)

Metrics collected:
- VRAM usage
- Inference speed (tokens/second)
- Audio quality (subjective + waveform comparison)
- Real-time factor

Usage:
    python tests/test_quantization_comparison.py

Requirements:
    - Quantized model checkpoints (run tools/llama/quantize.py first)
    - Or run with --quantize-first to create them
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    precision: str
    vram_before_load: float
    vram_after_load: float
    vram_peak: float
    model_size_gb: float
    load_time: float
    warmup_time: float
    inference_times: List[float]
    tokens_per_second: List[float]
    audio_durations: List[float]
    real_time_factors: List[float]
    errors: List[str]

    @property
    def avg_inference_time(self) -> float:
        return np.mean(self.inference_times) if self.inference_times else 0

    @property
    def avg_tokens_per_second(self) -> float:
        return np.mean(self.tokens_per_second) if self.tokens_per_second else 0

    @property
    def avg_rtf(self) -> float:
        return np.mean(self.real_time_factors) if self.real_time_factors else 0


def get_vram_gb() -> float:
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_peak_vram_gb() -> float:
    """Get peak VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_vram_stats():
    """Reset VRAM statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def check_quantized_checkpoints(base_path: str) -> Dict[str, Optional[str]]:
    """Check which quantized checkpoints exist."""
    base = Path(base_path)
    parent = base.parent

    checkpoints = {
        "float16": str(base) if base.exists() else None,
        "bfloat16": str(base) if base.exists() else None,  # Same checkpoint, different precision
    }

    # Look for quantized versions
    for item in parent.iterdir():
        if item.is_dir():
            name = item.name.lower()
            if "int8" in name:
                checkpoints["int8"] = str(item)
            elif "int4" in name:
                checkpoints["int4"] = str(item)

    return checkpoints


def create_quantized_checkpoint(
    source_path: str,
    quantization: str,
    group_size: int = 128,
) -> str:
    """Create a quantized checkpoint if it doesn't exist."""
    from tools.llama.quantize import quantize as quantize_model

    source = Path(source_path)
    parent = source.parent

    if quantization == "int8":
        output_name = f"{source.name}-int8"
    else:  # int4
        output_name = f"{source.name}-int4-g{group_size}"

    output_path = parent / output_name

    if output_path.exists():
        print(f"Quantized checkpoint already exists: {output_path}")
        return str(output_path)

    print(f"Creating {quantization} quantized checkpoint...")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")

    # Run quantization
    quantize_model(
        checkpoint_path=source_path,
        mode=quantization,
        group_size=group_size if quantization == "int4" else None,
    )

    return str(output_path)


def benchmark_precision(
    checkpoint_path: str,
    precision: str,
    test_texts: List[str],
    num_warmup: int = 2,
) -> BenchmarkResult:
    """
    Benchmark a specific precision/quantization configuration.

    Args:
        checkpoint_path: Path to model checkpoint
        precision: Precision to use (float16, bfloat16, int8, int4)
        test_texts: Texts to generate for benchmarking
        num_warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with timing and memory metrics
    """
    from s1_mini import ProductionTTSEngine, EngineConfig

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {precision}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}")

    reset_vram_stats()
    vram_before = get_vram_gb()

    # Determine actual precision to use
    if precision in ("int8", "int4"):
        # Quantized models use float16 for activations
        model_precision = "float16"
    else:
        model_precision = precision

    # Create engine
    config = EngineConfig(
        checkpoint_path=checkpoint_path,
        device="cuda",
        precision=model_precision,
        compile_model=True,
        enable_batching=False,
    )

    load_start = time.time()
    try:
        engine = ProductionTTSEngine(config)
        engine.start()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return BenchmarkResult(
            precision=precision,
            vram_before_load=vram_before,
            vram_after_load=0,
            vram_peak=0,
            model_size_gb=0,
            load_time=0,
            warmup_time=0,
            inference_times=[],
            tokens_per_second=[],
            audio_durations=[],
            real_time_factors=[],
            errors=[str(e)],
        )
    load_time = time.time() - load_start

    vram_after = get_vram_gb()
    model_size = vram_after - vram_before

    print(f"  Load time: {load_time:.2f}s")
    print(f"  Model VRAM: {model_size:.2f} GB")

    # Warmup
    warmup_start = time.time()
    for i in range(num_warmup):
        try:
            _ = engine.generate(
                text="Warmup text for model initialization.",
                max_new_tokens=128,
            )
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
    warmup_time = time.time() - warmup_start

    print(f"  Warmup time: {warmup_time:.2f}s")

    # Benchmark inference
    inference_times = []
    tokens_per_second = []
    audio_durations = []
    real_time_factors = []
    errors = []

    for i, text in enumerate(test_texts):
        print(f"\n  Test {i+1}/{len(test_texts)}: {text[:50]}...")

        try:
            start = time.time()
            response = engine.generate(
                text=text,
                max_new_tokens=2048,
                temperature=0.7,
                return_bytes=True,
            )
            elapsed = time.time() - start

            if response.success and response.audio:
                sample_rate, audio = response.audio
                duration = len(audio) / sample_rate
                rtf = duration / elapsed

                inference_times.append(elapsed)
                audio_durations.append(duration)
                real_time_factors.append(rtf)

                if response.metrics:
                    tokens_per_second.append(
                        response.metrics.tokens_generated / response.metrics.total_time_seconds
                    )

                print(f"    Time: {elapsed:.2f}s, Audio: {duration:.2f}s, RTF: {rtf:.2f}x")
            else:
                errors.append(response.error or "Unknown error")
                print(f"    FAILED: {response.error}")

        except Exception as e:
            errors.append(str(e))
            print(f"    ERROR: {e}")

    # Get peak VRAM
    vram_peak = get_peak_vram_gb()

    # Cleanup
    engine.stop()
    del engine
    reset_vram_stats()

    return BenchmarkResult(
        precision=precision,
        vram_before_load=vram_before,
        vram_after_load=vram_after,
        vram_peak=vram_peak,
        model_size_gb=model_size,
        load_time=load_time,
        warmup_time=warmup_time,
        inference_times=inference_times,
        tokens_per_second=tokens_per_second,
        audio_durations=audio_durations,
        real_time_factors=real_time_factors,
        errors=errors,
    )


def print_comparison_table(results: Dict[str, BenchmarkResult]):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Precision':<12} {'VRAM (GB)':<12} {'Load (s)':<10} {'Avg Time':<10} "
          f"{'Tokens/s':<10} {'RTF':<8} {'Errors':<8}")
    print("-" * 80)

    # Results
    baseline_rtf = None
    for precision, result in results.items():
        if result.errors and not result.inference_times:
            print(f"{precision:<12} FAILED: {result.errors[0][:50]}")
            continue

        rtf = result.avg_rtf
        if baseline_rtf is None:
            baseline_rtf = rtf
            speedup = "baseline"
        else:
            speedup = f"{rtf / baseline_rtf:.2f}x" if baseline_rtf > 0 else "N/A"

        print(f"{precision:<12} {result.model_size_gb:<12.2f} {result.load_time:<10.2f} "
              f"{result.avg_inference_time:<10.2f} {result.avg_tokens_per_second:<10.1f} "
              f"{rtf:<8.2f} {len(result.errors):<8}")

    print("=" * 80)

    # Summary
    print("\nKey Insights:")
    for precision, result in results.items():
        if result.model_size_gb > 0:
            print(f"  {precision}: {result.model_size_gb:.2f} GB VRAM, "
                  f"{result.avg_rtf:.2f}x real-time")


def main():
    parser = argparse.ArgumentParser(description="Compare quantization approaches")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/openaudio-s1-mini",
        help="Base checkpoint path",
    )
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        default=["float16", "bfloat16"],
        choices=["float16", "bfloat16", "int8", "int4"],
        help="Precisions to test",
    )
    parser.add_argument(
        "--quantize-first",
        action="store_true",
        help="Create quantized checkpoints if they don't exist",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of test texts to generate",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    # Check/create checkpoints
    checkpoints = check_quantized_checkpoints(args.checkpoint)
    print("\nAvailable checkpoints:")
    for precision, path in checkpoints.items():
        status = "FOUND" if path else "NOT FOUND"
        print(f"  {precision}: {status} - {path}")

    # Create quantized checkpoints if requested
    if args.quantize_first:
        for precision in args.precisions:
            if precision in ("int8", "int4") and checkpoints.get(precision) is None:
                checkpoints[precision] = create_quantized_checkpoint(
                    args.checkpoint,
                    precision,
                )

    # Test texts
    test_texts = [
        "Welcome to our text-to-speech demonstration. This system converts written text into natural sounding speech.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
        "Technology continues to advance at an incredible pace, transforming how we live and work.",
        "Music has the power to evoke deep emotions and memories. It truly is a universal language.",
        "Education opens doors to unlimited possibilities and meaningful contributions to society.",
    ][:args.num_tests]

    # Run benchmarks
    results = {}
    for precision in args.precisions:
        checkpoint = checkpoints.get(precision)
        if checkpoint is None:
            print(f"\nSkipping {precision}: checkpoint not found")
            print(f"  Run with --quantize-first to create quantized checkpoints")
            continue

        results[precision] = benchmark_precision(
            checkpoint_path=checkpoint,
            precision=precision,
            test_texts=test_texts,
        )

    # Print comparison
    if results:
        print_comparison_table(results)

    # Save results
    output_path = Path("test_outputs") / "quantization_comparison.json"
    output_path.parent.mkdir(exist_ok=True)

    import json
    with open(output_path, "w") as f:
        json.dump(
            {
                precision: {
                    "model_size_gb": r.model_size_gb,
                    "load_time": r.load_time,
                    "avg_inference_time": r.avg_inference_time,
                    "avg_tokens_per_second": r.avg_tokens_per_second,
                    "avg_rtf": r.avg_rtf,
                    "errors": r.errors,
                }
                for precision, r in results.items()
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
