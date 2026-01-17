"""
Comprehensive Fish-Speech Benchmark for RunPod
Tests: BF16 vs INT8, single vs batch inference, VRAM usage, throughput

Run: python tools/runpod/benchmark.py
"""

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np


@dataclass
class BenchmarkResult:
    config_name: str
    llama_vram_mb: float
    dac_vram_mb: float
    total_baseline_vram_mb: float
    peak_vram_mb: float
    avg_generation_time_s: float
    avg_audio_duration_s: float
    avg_rtf: float  # Real-time factor (lower is better)
    avg_tokens_per_sec: float
    throughput_audio_sec_per_sec: float  # How many seconds of audio per second of compute
    samples_tested: int
    errors: int = 0
    notes: str = ""


@dataclass
class BatchBenchmarkResult:
    config_name: str
    batch_size: int
    total_generation_time_s: float
    total_audio_duration_s: float
    throughput_audio_sec_per_sec: float
    peak_vram_mb: float
    samples_tested: int


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def clear_vram():
    """Clear VRAM and reset stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_vram_mb():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0


def get_peak_vram_mb():
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0


TEST_TEXTS = [
    # Short
    "Hello, this is a test of the text to speech system.",
    # Medium
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and is commonly used for testing.",
    # Long
    "Artificial intelligence has transformed the way we interact with technology. From voice assistants to autonomous vehicles, AI systems are becoming increasingly integrated into our daily lives. The development of large language models has particularly revolutionized natural language processing, enabling more natural and intuitive human-computer interactions.",
    # Multi-language
    "今天天气真好，阳光明媚。I hope you're having a wonderful day!",
]


def benchmark_config(
    config_name: str,
    checkpoint_path: str,
    runtime_int8: bool = False,
    runtime_int4: bool = False,
    dac_int8: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 1024,
) -> BenchmarkResult:
    """Benchmark a specific configuration."""
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")

    clear_vram()
    vram_start = get_vram_mb()

    # Load LLaMA model
    print("Loading LLaMA model...")
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device="cuda",
        precision=precision,
        compile=False,
        runtime_int4=runtime_int4,
        runtime_int8=runtime_int8,
    )
    torch.cuda.synchronize()
    vram_after_llama = get_vram_mb()
    llama_vram = vram_after_llama - vram_start
    print(f"  LLaMA VRAM: {llama_vram:.1f} MB")

    # Load DAC decoder
    print("Loading DAC decoder...")
    dac = load_dac_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(Path(checkpoint_path) / "codec.pth"),
        device="cuda",
        quantize_int8=dac_int8,
    )
    torch.cuda.synchronize()
    vram_after_dac = get_vram_mb()
    dac_vram = vram_after_dac - vram_after_llama
    print(f"  DAC VRAM: {dac_vram:.1f} MB")

    total_baseline = vram_after_dac - vram_start
    print(f"  Total baseline VRAM: {total_baseline:.1f} MB")

    # Create engine
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=dac,
        precision=precision,
        compile=False,
    )

    # Warmup
    print("Warming up...")
    req = ServeTTSRequest(text="Warmup test.", max_new_tokens=256, streaming=False)
    for result in engine.inference(req):
        pass

    # Reset peak VRAM after warmup
    torch.cuda.reset_peak_memory_stats()

    # Run benchmarks
    print(f"Running {num_samples} samples per text ({len(TEST_TEXTS)} texts)...")
    generation_times = []
    audio_durations = []
    tokens_generated = []
    errors = 0

    for text in TEST_TEXTS:
        for i in range(num_samples):
            try:
                req = ServeTTSRequest(
                    text=text,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    streaming=False,
                )

                start_time = time.perf_counter()
                audio_result = None

                for result in engine.inference(req):
                    if result.code == "final":
                        audio_result = result.audio

                gen_time = time.perf_counter() - start_time

                if audio_result:
                    sample_rate, audio_data = audio_result
                    audio_duration = len(audio_data) / sample_rate
                    generation_times.append(gen_time)
                    audio_durations.append(audio_duration)
                    # Rough token estimate based on audio length
                    tokens_generated.append(int(audio_duration * 21.5))  # ~21.5 tokens/sec audio
                else:
                    errors += 1

            except Exception as e:
                print(f"  Error: {e}")
                errors += 1

    peak_vram = get_peak_vram_mb()

    # Calculate metrics
    if generation_times:
        avg_gen_time = np.mean(generation_times)
        avg_audio_dur = np.mean(audio_durations)
        avg_rtf = avg_gen_time / avg_audio_dur if avg_audio_dur > 0 else float('inf')
        avg_tokens_sec = np.mean(tokens_generated) / avg_gen_time if avg_gen_time > 0 else 0
        throughput = avg_audio_dur / avg_gen_time if avg_gen_time > 0 else 0
    else:
        avg_gen_time = avg_audio_dur = avg_rtf = avg_tokens_sec = throughput = 0

    result = BenchmarkResult(
        config_name=config_name,
        llama_vram_mb=llama_vram,
        dac_vram_mb=dac_vram,
        total_baseline_vram_mb=total_baseline,
        peak_vram_mb=peak_vram,
        avg_generation_time_s=avg_gen_time,
        avg_audio_duration_s=avg_audio_dur,
        avg_rtf=avg_rtf,
        avg_tokens_per_sec=avg_tokens_sec,
        throughput_audio_sec_per_sec=throughput,
        samples_tested=len(generation_times),
        errors=errors,
    )

    print(f"\nResults:")
    print(f"  Peak VRAM: {peak_vram:.1f} MB")
    print(f"  Avg generation time: {avg_gen_time:.2f}s")
    print(f"  Avg audio duration: {avg_audio_dur:.2f}s")
    print(f"  RTF: {avg_rtf:.3f} (lower is better)")
    print(f"  Throughput: {throughput:.2f}x real-time")
    print(f"  Tokens/sec: {avg_tokens_sec:.1f}")
    print(f"  Errors: {errors}")

    # Cleanup
    del engine, dac, llama_queue
    clear_vram()

    return result


def benchmark_batch_inference(
    config_name: str,
    checkpoint_path: str,
    batch_sizes: list[int] = [1, 2, 4, 8],
    runtime_int8: bool = True,
    dac_int8: bool = True,
) -> list[BatchBenchmarkResult]:
    """Benchmark batch inference with different batch sizes."""
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    print(f"\n{'='*60}")
    print(f"Batch Inference Benchmark: {config_name}")
    print(f"{'='*60}")

    clear_vram()

    # Load models
    print("Loading models...")
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device="cuda",
        precision=precision,
        compile=False,
        runtime_int8=runtime_int8,
    )

    dac = load_dac_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(Path(checkpoint_path) / "codec.pth"),
        device="cuda",
        quantize_int8=dac_int8,
    )

    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=dac,
        precision=precision,
        compile=False,
    )

    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        torch.cuda.reset_peak_memory_stats()

        # Generate batch_size requests sequentially (simulating concurrent requests)
        texts = [TEST_TEXTS[i % len(TEST_TEXTS)] for i in range(batch_size)]

        start_time = time.perf_counter()
        total_audio_duration = 0
        samples_completed = 0

        for text in texts:
            try:
                req = ServeTTSRequest(
                    text=text,
                    max_new_tokens=1024,
                    streaming=False,
                )

                for result in engine.inference(req):
                    if result.code == "final" and result.audio:
                        sample_rate, audio_data = result.audio
                        total_audio_duration += len(audio_data) / sample_rate
                        samples_completed += 1
            except Exception as e:
                print(f"  Error: {e}")

        total_time = time.perf_counter() - start_time
        peak_vram = get_peak_vram_mb()
        throughput = total_audio_duration / total_time if total_time > 0 else 0

        result = BatchBenchmarkResult(
            config_name=config_name,
            batch_size=batch_size,
            total_generation_time_s=total_time,
            total_audio_duration_s=total_audio_duration,
            throughput_audio_sec_per_sec=throughput,
            peak_vram_mb=peak_vram,
            samples_tested=samples_completed,
        )
        results.append(result)

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total audio: {total_audio_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f}x real-time")
        print(f"  Peak VRAM: {peak_vram:.1f} MB")

    # Cleanup
    del engine, dac, llama_queue
    clear_vram()

    return results


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("Fish-Speech Full Benchmark Suite")
    print("=" * 70)

    # GPU Info
    gpu_info = get_gpu_info()
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")

    checkpoint_path = "checkpoints/openaudio-s1-mini"

    if not Path(checkpoint_path).exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
        return

    # Configurations to test
    configs = [
        ("BF16 (baseline)", False, False, False),
        ("BF16 + DAC INT8", False, False, True),
        ("INT8 Runtime", False, True, False),
        ("INT8 Runtime + DAC INT8", False, True, True),
    ]

    results = []

    for config_name, runtime_int4, runtime_int8, dac_int8 in configs:
        try:
            result = benchmark_config(
                config_name=config_name,
                checkpoint_path=checkpoint_path,
                runtime_int8=runtime_int8,
                runtime_int4=runtime_int4,
                dac_int8=dac_int8,
                num_samples=3,  # 3 samples per text
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Batch inference benchmark
    print("\n" + "=" * 70)
    print("Batch Inference Benchmark")
    print("=" * 70)

    batch_results = benchmark_batch_inference(
        config_name="INT8 + DAC INT8",
        checkpoint_path=checkpoint_path,
        batch_sizes=[1, 2, 4],
        runtime_int8=True,
        dac_int8=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n## Single Request Performance\n")
    print(f"{'Config':<30} {'VRAM':<12} {'Peak VRAM':<12} {'RTF':<10} {'Throughput':<12}")
    print("-" * 76)

    for r in results:
        print(f"{r.config_name:<30} {r.total_baseline_vram_mb:>8.0f} MB {r.peak_vram_mb:>8.0f} MB {r.avg_rtf:>8.3f} {r.throughput_audio_sec_per_sec:>8.2f}x")

    print("\n## Batch Inference (INT8 + DAC INT8)\n")
    print(f"{'Batch Size':<15} {'Total Time':<15} {'Throughput':<15} {'Peak VRAM':<15}")
    print("-" * 60)

    for r in batch_results:
        print(f"{r.batch_size:<15} {r.total_generation_time_s:>10.2f}s {r.throughput_audio_sec_per_sec:>11.2f}x {r.peak_vram_mb:>11.0f} MB")

    # Save results
    output = {
        "gpu_info": gpu_info,
        "single_request_results": [
            {
                "config": r.config_name,
                "llama_vram_mb": r.llama_vram_mb,
                "dac_vram_mb": r.dac_vram_mb,
                "total_baseline_vram_mb": r.total_baseline_vram_mb,
                "peak_vram_mb": r.peak_vram_mb,
                "avg_generation_time_s": r.avg_generation_time_s,
                "avg_audio_duration_s": r.avg_audio_duration_s,
                "avg_rtf": r.avg_rtf,
                "throughput_x_realtime": r.throughput_audio_sec_per_sec,
                "samples_tested": r.samples_tested,
            }
            for r in results
        ],
        "batch_results": [
            {
                "batch_size": r.batch_size,
                "total_generation_time_s": r.total_generation_time_s,
                "total_audio_duration_s": r.total_audio_duration_s,
                "throughput_x_realtime": r.throughput_audio_sec_per_sec,
                "peak_vram_mb": r.peak_vram_mb,
            }
            for r in batch_results
        ],
    }

    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Production recommendations
    print("\n" + "=" * 70)
    print("PRODUCTION RECOMMENDATIONS")
    print("=" * 70)

    if results:
        best_throughput = max(results, key=lambda x: x.throughput_audio_sec_per_sec)
        best_vram = min(results, key=lambda x: x.total_baseline_vram_mb)

        print(f"\nBest throughput: {best_throughput.config_name}")
        print(f"  {best_throughput.throughput_audio_sec_per_sec:.2f}x real-time")

        print(f"\nLowest VRAM: {best_vram.config_name}")
        print(f"  {best_vram.total_baseline_vram_mb:.0f} MB baseline")

        # Estimate concurrent users
        if best_throughput.avg_generation_time_s > 0:
            avg_request_time = best_throughput.avg_generation_time_s
            requests_per_min = 60 / avg_request_time
            print(f"\nEstimated capacity (single GPU):")
            print(f"  Requests/minute: ~{requests_per_min:.0f}")
            print(f"  Audio minutes/minute: ~{requests_per_min * best_throughput.avg_audio_duration_s / 60:.1f}")


if __name__ == "__main__":
    run_full_benchmark()
