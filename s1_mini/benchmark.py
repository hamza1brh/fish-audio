"""
S1-Mini Real-Time Factor Benchmark
==================================

Measures how fast your GPU can generate audio compared to real-time playback.

Real-Time Factor (RTF) = Audio Duration / Generation Time
- RTF > 1.0 = Faster than real-time (good for production)
- RTF = 1.0 = Exactly real-time
- RTF < 1.0 = Slower than real-time

Example: RTF of 2.0 means generating 2 seconds of audio per 1 second of compute
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ["S1_MINI_LOG_PLATFORM"] = "false"

import numpy as np

# Configure optimal attention BEFORE importing models
from s1_mini.attention import configure_optimal_attention, log_attention_info
from s1_mini.compilation import configure_windows_fallback, get_platform_info


@dataclass
class BenchmarkResult:
    text: str
    text_length: int
    generation_time_s: float
    audio_duration_s: float
    rtf: float  # Real-time factor
    tokens_generated: int
    tokens_per_second: float
    sample_rate: int


def run_benchmark():
    print("=" * 70)
    print("S1-Mini Real-Time Factor Benchmark (Optimized)")
    print("=" * 70)

    import torch
    from s1_mini.config import EngineConfig
    from s1_mini.model_manager import ModelManager
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU")

    # Apply optimizations
    print("\nApplying optimizations...")
    platform_info = get_platform_info()
    print(f"  Platform: {platform_info.system}")
    print(f"  Triton: {'Available' if platform_info.triton_available else 'Not Available'}")

    # Configure Windows fallback optimizations
    if platform_info.system == "Windows":
        configure_windows_fallback()
        print("  Applied: cuDNN benchmark, TF32, max-autotune")

    # Configure optimal attention
    attention_backend = configure_optimal_attention()
    print(f"  Attention: {attention_backend.value}")

    # Log attention details
    log_attention_info()

    print("\nLoading models...")

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
        compile_model=True,  # Use torch.compile with optimizations
    )

    manager = ModelManager(config)
    manager.load_models()

    tts_engine = TTSInferenceEngine(
        llama_queue=manager.llama_queue,
        decoder_model=manager.decoder_model,
        precision=config.torch_dtype,
        compile=False,
    )

    # Benchmark texts of varying lengths
    benchmark_texts = [
        # Short
        "Hello world.",
        "This is a test.",

        # Medium
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to the S1 Mini text to speech benchmark test.",
        "Artificial intelligence is transforming how we interact with technology.",

        # Long
        "In a world where technology continues to advance at an unprecedented pace, "
        "the ability to convert text into natural sounding speech has become increasingly important. "
        "This benchmark measures the real-time factor of the S1 Mini model.",

        # Very long
        "Machine learning models have revolutionized the field of speech synthesis. "
        "Modern neural text to speech systems can produce audio that is nearly indistinguishable "
        "from human speech. The key metrics for evaluating these systems include audio quality, "
        "naturalness, and most importantly for production use, the speed at which they can generate audio. "
        "A real-time factor greater than one indicates that the system can generate audio faster than "
        "it takes to play back, which is essential for interactive applications.",
    ]

    results: List[BenchmarkResult] = []

    # Warmup runs to trigger JIT compilation
    print("\n" + "=" * 70)
    print("Warmup Phase (triggering JIT compilation)")
    print("=" * 70)

    warmup_request = ServeTTSRequest(
        text="This is a warmup run to trigger JIT compilation.",
        references=[],
        reference_id=None,
        max_new_tokens=256,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        format="wav",
    )

    print("Running 2 warmup iterations...")
    for warmup_i in range(2):
        print(f"  Warmup {warmup_i + 1}/2...")
        for result in tts_engine.inference(warmup_request):
            if result.code == "final":
                break
        torch.cuda.synchronize()
    print("Warmup complete!")

    print("\n" + "=" * 70)
    print("Running Benchmark (post-warmup)")
    print("=" * 70)

    for i, text in enumerate(benchmark_texts, 1):
        print(f"\n[{i}/{len(benchmark_texts)}] Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"    Length: {len(text)} characters")

        request = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=None,
            max_new_tokens=2048,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )

        # Warm up CUDA
        torch.cuda.synchronize()

        start_time = time.perf_counter()

        audio_data = None
        sample_rate = 44100

        for result in tts_engine.inference(request):
            if result.code == "final" and result.audio is not None:
                sample_rate, audio_data = result.audio
                break

        torch.cuda.synchronize()
        generation_time = time.perf_counter() - start_time

        if audio_data is None:
            print(f"    ERROR: No audio generated")
            continue

        audio_duration = len(audio_data) / sample_rate
        rtf = audio_duration / generation_time if generation_time > 0 else 0

        # Estimate tokens (approximate from audio length)
        # ~86 tokens per second of audio based on model architecture
        estimated_tokens = int(audio_duration * 86)
        tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0

        result = BenchmarkResult(
            text=text,
            text_length=len(text),
            generation_time_s=generation_time,
            audio_duration_s=audio_duration,
            rtf=rtf,
            tokens_generated=estimated_tokens,
            tokens_per_second=tokens_per_second,
            sample_rate=sample_rate,
        )
        results.append(result)

        print(f"    Generation: {generation_time:.2f}s")
        print(f"    Audio:      {audio_duration:.2f}s")
        print(f"    RTF:        {rtf:.2f}x {'(real-time capable!)' if rtf >= 1.0 else ''}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print("\nDetailed Results:")
    print("-" * 70)
    print(f"{'Text Len':>10} | {'Gen Time':>10} | {'Audio Dur':>10} | {'RTF':>8} | {'Tokens/s':>10}")
    print("-" * 70)

    for r in results:
        rtf_indicator = "***" if r.rtf >= 1.0 else ""
        print(f"{r.text_length:>10} | {r.generation_time_s:>10.2f}s | {r.audio_duration_s:>10.2f}s | {r.rtf:>7.2f}x | {r.tokens_per_second:>10.1f} {rtf_indicator}")

    print("-" * 70)

    # Aggregate statistics
    if results:
        avg_rtf = np.mean([r.rtf for r in results])
        min_rtf = np.min([r.rtf for r in results])
        max_rtf = np.max([r.rtf for r in results])
        avg_tokens_per_sec = np.mean([r.tokens_per_second for r in results])

        total_audio = sum(r.audio_duration_s for r in results)
        total_gen_time = sum(r.generation_time_s for r in results)
        overall_rtf = total_audio / total_gen_time if total_gen_time > 0 else 0

        print(f"\n{'SUMMARY':^70}")
        print("=" * 70)
        print(f"  Total audio generated:     {total_audio:.2f} seconds")
        print(f"  Total generation time:     {total_gen_time:.2f} seconds")
        print(f"  Overall Real-Time Factor:  {overall_rtf:.2f}x")
        print("-" * 70)
        print(f"  Average RTF:               {avg_rtf:.2f}x")
        print(f"  Min RTF:                   {min_rtf:.2f}x")
        print(f"  Max RTF:                   {max_rtf:.2f}x")
        print(f"  Average Tokens/sec:        {avg_tokens_per_sec:.1f}")
        print("=" * 70)

        # Production readiness assessment
        print(f"\n{'PRODUCTION ASSESSMENT':^70}")
        print("=" * 70)

        if overall_rtf >= 1.5:
            print("  STATUS: EXCELLENT - Well above real-time")
            print(f"  Your RTX 5070 Ti can handle ~{int(overall_rtf)} concurrent streams")
        elif overall_rtf >= 1.0:
            print("  STATUS: GOOD - Real-time capable")
            print("  Suitable for interactive applications")
        elif overall_rtf >= 0.5:
            print("  STATUS: ACCEPTABLE - Near real-time")
            print("  OK for batch processing, may have latency for interactive use")
        else:
            print("  STATUS: SLOW - Below real-time")
            print("  Consider using Linux with Triton for better performance")

        print(f"\n  Audio generation rate: {overall_rtf:.2f} seconds of audio per second of compute")
        print(f"  Latency for 5s audio:  ~{5/overall_rtf:.1f} seconds")
        print(f"  Latency for 10s audio: ~{10/overall_rtf:.1f} seconds")
        print("=" * 70)

    # Cleanup
    print("\nCleaning up...")
    manager.shutdown()
    print("Done!")


if __name__ == "__main__":
    run_benchmark()
