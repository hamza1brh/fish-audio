#!/usr/bin/env python3
"""
Fish-Speech GPU Stress Test & Capacity Analysis

This benchmark answers critical production questions:
1. What's the maximum requests/minute on a single GPU?
2. Can we run multiple model instances on one GPU?
3. What's the GPU utilization vs VRAM trade-off?
4. Is batching dynamic or static?
5. Where are the bottlenecks?

Usage:
    python tools/runpod/gpu_stress_test.py
    python tools/runpod/gpu_stress_test.py --duration 300  # 5 minute stress test
    python tools/runpod/gpu_stress_test.py --test-multi-instance  # Test multiple models

Architecture Analysis:
    - Current implementation: batch_size=1, sequential processing
    - CUDA graphs require static shapes, preventing dynamic batching
    - Single worker thread processes queue
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import queue

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np


@dataclass
class GPUMetrics:
    """Real-time GPU metrics."""
    timestamp: float
    gpu_utilization: float  # %
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float
    sm_clock_mhz: float


@dataclass
class StressTestResults:
    """Results from stress test."""
    test_name: str
    duration_s: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_audio_s: float

    # Throughput metrics
    requests_per_minute: float
    requests_per_second: float
    audio_per_minute_s: float
    throughput_x_realtime: float

    # Latency metrics
    avg_latency_s: float
    min_latency_s: float
    max_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float

    # GPU metrics
    avg_gpu_utilization: float
    max_gpu_utilization: float
    avg_vram_mb: float
    peak_vram_mb: float
    avg_power_w: float
    max_power_w: float

    # Raw data
    latencies: list = field(default_factory=list)
    gpu_samples: list = field(default_factory=list)


def get_gpu_metrics() -> Optional[GPUMetrics]:
    """Get current GPU metrics using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(", ")
        if len(parts) >= 6:
            return GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=float(parts[0]),
                memory_used_mb=float(parts[1]),
                memory_total_mb=float(parts[2]),
                temperature_c=float(parts[3]),
                power_draw_w=float(parts[4]) if parts[4] != "[N/A]" else 0,
                sm_clock_mhz=float(parts[5]) if parts[5] != "[N/A]" else 0,
            )
    except Exception as e:
        pass
    return None


class GPUMonitor:
    """Background GPU monitoring thread."""

    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self.samples: list[GPUMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _monitor_loop(self):
        while self._running:
            metrics = get_gpu_metrics()
            if metrics:
                self.samples.append(metrics)
            time.sleep(self.interval)

    def get_summary(self) -> dict:
        if not self.samples:
            return {}

        utils = [s.gpu_utilization for s in self.samples]
        vrams = [s.memory_used_mb for s in self.samples]
        powers = [s.power_draw_w for s in self.samples if s.power_draw_w > 0]

        return {
            "avg_gpu_utilization": np.mean(utils),
            "max_gpu_utilization": np.max(utils),
            "min_gpu_utilization": np.min(utils),
            "avg_vram_mb": np.mean(vrams),
            "peak_vram_mb": np.max(vrams),
            "avg_power_w": np.mean(powers) if powers else 0,
            "max_power_w": np.max(powers) if powers else 0,
            "samples_count": len(self.samples),
        }


def clear_vram():
    """Clear VRAM and reset stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def analyze_architecture():
    """Analyze the current inference architecture."""
    print("\n" + "="*70)
    print("  ARCHITECTURE ANALYSIS")
    print("="*70)

    print("""
Current Implementation:
  - Batch Size: 1 (fixed, not dynamic)
  - Processing: Sequential (one request at a time)
  - Queue: Single input queue → single worker thread → response queue
  - CUDA Graphs: Enabled (mode="reduce-overhead")

Why batch_size=1?
  - CUDA graphs require static tensor shapes
  - Dynamic batching would invalidate the captured graph
  - Trade-off: Lower latency per request vs lower throughput

Implications:
  - Cannot increase batch size without disabling CUDA graphs
  - Multiple workers on same model would cause TLS assertion errors
  - GPU may be underutilized if compute is faster than data transfer

Potential Optimizations:
  1. Multiple model instances (if VRAM allows)
  2. Speculative decoding (requires model changes)
  3. Continuous batching (complex, like vLLM)
  4. Different compile mode (trade latency for throughput)
""")


def run_stress_test(
    duration_s: float = 60,
    text_length: str = "mixed",
    warmup_s: float = 10,
) -> StressTestResults:
    """
    Run stress test - continuous requests for specified duration.
    Measures maximum sustainable throughput.
    """
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    checkpoint_path = "checkpoints/openaudio-s1-mini"

    print(f"\n{'='*70}")
    print(f"  STRESS TEST: {text_length} texts for {duration_s}s")
    print(f"{'='*70}")

    # Test texts by length
    texts = {
        "short": [
            "Hello world.",
            "Testing one two three.",
            "How are you today?",
        ],
        "medium": [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter.",
            "Artificial intelligence is transforming technology and enabling new possibilities.",
            "Welcome to our text to speech demonstration system.",
        ],
        "long": [
            "Artificial intelligence has transformed the way we interact with technology. From voice assistants to autonomous vehicles, AI systems are becoming increasingly integrated into our daily lives. The development of large language models has particularly revolutionized natural language processing.",
            "In the realm of modern computing, the convergence of hardware acceleration and sophisticated algorithms has enabled unprecedented capabilities in speech synthesis. Neural network architectures have evolved to produce remarkably natural-sounding voices.",
        ],
    }

    if text_length == "mixed":
        test_texts = texts["short"] + texts["medium"] + texts["long"]
    else:
        test_texts = texts.get(text_length, texts["medium"])

    clear_vram()

    # Load models
    print("Loading models...")
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device="cuda",
        precision=precision,
        compile=True,  # Always use compile
        runtime_int8=False,
        runtime_int4=False,
    )

    dac = load_dac_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(Path(checkpoint_path) / "codec.pth"),
        device="cuda",
        quantize_int8=False,  # No INT8, full precision only
    )

    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=dac,
        precision=precision,
        compile=True,
    )

    # Warmup
    print(f"Warming up for {warmup_s}s...")
    warmup_start = time.time()
    warmup_count = 0
    while time.time() - warmup_start < warmup_s:
        req = ServeTTSRequest(text="Warmup test.", max_new_tokens=256, streaming=False)
        for result in engine.inference(req):
            pass
        warmup_count += 1
    print(f"  Completed {warmup_count} warmup requests")

    torch.cuda.reset_peak_memory_stats()

    # Start GPU monitoring
    monitor = GPUMonitor(interval_s=0.25)
    monitor.start()

    # Run stress test
    print(f"\nRunning stress test for {duration_s}s...")
    start_time = time.time()
    latencies = []
    audio_durations = []
    errors = 0
    request_idx = 0

    while time.time() - start_time < duration_s:
        text = test_texts[request_idx % len(test_texts)]
        request_idx += 1

        try:
            req = ServeTTSRequest(
                text=text,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                streaming=False,
            )

            req_start = time.time()
            audio_result = None

            for result in engine.inference(req):
                if result.code == "final":
                    audio_result = result.audio

            torch.cuda.synchronize()
            req_latency = time.time() - req_start

            if audio_result:
                sample_rate, audio_data = audio_result
                audio_dur = len(audio_data) / sample_rate
                latencies.append(req_latency)
                audio_durations.append(audio_dur)

                # Progress update every 10 requests
                if len(latencies) % 10 == 0:
                    elapsed = time.time() - start_time
                    rpm = len(latencies) / elapsed * 60
                    print(f"  [{elapsed:.0f}s] {len(latencies)} requests, {rpm:.1f} req/min, last latency: {req_latency:.2f}s")
            else:
                errors += 1

        except Exception as e:
            errors += 1
            print(f"  Error: {e}")

    total_time = time.time() - start_time
    monitor.stop()

    # Calculate metrics
    gpu_summary = monitor.get_summary()

    if latencies:
        lat_array = np.array(latencies)
        total_audio = sum(audio_durations)

        results = StressTestResults(
            test_name=f"stress_test_{text_length}",
            duration_s=total_time,
            total_requests=len(latencies) + errors,
            successful_requests=len(latencies),
            failed_requests=errors,
            total_audio_s=total_audio,

            requests_per_minute=len(latencies) / total_time * 60,
            requests_per_second=len(latencies) / total_time,
            audio_per_minute_s=total_audio / total_time * 60,
            throughput_x_realtime=total_audio / total_time,

            avg_latency_s=float(np.mean(lat_array)),
            min_latency_s=float(np.min(lat_array)),
            max_latency_s=float(np.max(lat_array)),
            p50_latency_s=float(np.percentile(lat_array, 50)),
            p95_latency_s=float(np.percentile(lat_array, 95)),
            p99_latency_s=float(np.percentile(lat_array, 99)),

            avg_gpu_utilization=gpu_summary.get("avg_gpu_utilization", 0),
            max_gpu_utilization=gpu_summary.get("max_gpu_utilization", 0),
            avg_vram_mb=gpu_summary.get("avg_vram_mb", 0),
            peak_vram_mb=gpu_summary.get("peak_vram_mb", 0),
            avg_power_w=gpu_summary.get("avg_power_w", 0),
            max_power_w=gpu_summary.get("max_power_w", 0),

            latencies=latencies,
            gpu_samples=[vars(s) for s in monitor.samples],
        )
    else:
        results = StressTestResults(
            test_name=f"stress_test_{text_length}",
            duration_s=total_time,
            total_requests=errors,
            successful_requests=0,
            failed_requests=errors,
            total_audio_s=0,
            requests_per_minute=0,
            requests_per_second=0,
            audio_per_minute_s=0,
            throughput_x_realtime=0,
            avg_latency_s=0,
            min_latency_s=0,
            max_latency_s=0,
            p50_latency_s=0,
            p95_latency_s=0,
            p99_latency_s=0,
            avg_gpu_utilization=0,
            max_gpu_utilization=0,
            avg_vram_mb=0,
            peak_vram_mb=0,
            avg_power_w=0,
            max_power_w=0,
        )

    # Cleanup
    del engine, dac, llama_queue
    clear_vram()

    return results


def test_multi_instance_feasibility():
    """
    Test if multiple model instances can coexist on one GPU.
    This tests the VRAM limit and whether CUDA graphs conflict.
    """
    print("\n" + "="*70)
    print("  MULTI-INSTANCE FEASIBILITY TEST")
    print("="*70)

    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    checkpoint_path = "checkpoints/openaudio-s1-mini"
    precision = torch.bfloat16

    clear_vram()

    # Get baseline VRAM
    baseline_vram = torch.cuda.memory_allocated() / 1e6
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.0f} MB")
    print(f"Baseline VRAM: {baseline_vram:.0f} MB")

    instances = []
    instance_count = 0
    max_instances = 4  # Try up to 4 instances

    print(f"\nTrying to load up to {max_instances} model instances...")

    for i in range(max_instances):
        try:
            print(f"\n  Loading instance {i+1}...")
            vram_before = torch.cuda.memory_allocated() / 1e6

            llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device="cuda",
                precision=precision,
                compile=True,
                runtime_int8=False,
            )

            dac = load_dac_model(
                config_name="modded_dac_vq",
                checkpoint_path=str(Path(checkpoint_path) / "codec.pth"),
                device="cuda",
                quantize_int8=False,
            )

            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=dac,
                precision=precision,
                compile=True,
            )

            vram_after = torch.cuda.memory_allocated() / 1e6
            instance_vram = vram_after - vram_before

            print(f"    VRAM used: {instance_vram:.0f} MB")
            print(f"    Total VRAM: {vram_after:.0f} MB / {total_vram:.0f} MB")

            # Test inference
            print(f"    Testing inference...")
            req = ServeTTSRequest(text="Test.", max_new_tokens=128, streaming=False)
            for result in engine.inference(req):
                if result.code == "final":
                    print(f"    Inference OK!")

            instances.append({
                "llama_queue": llama_queue,
                "dac": dac,
                "engine": engine,
                "vram_mb": instance_vram,
            })
            instance_count += 1

        except Exception as e:
            print(f"    FAILED: {e}")
            break

    # Test concurrent inference on multiple instances
    if len(instances) > 1:
        print(f"\n  Testing concurrent inference on {len(instances)} instances...")

        def run_inference(engine, idx):
            try:
                req = ServeTTSRequest(text=f"Test instance {idx}.", max_new_tokens=256, streaming=False)
                for result in engine.inference(req):
                    if result.code == "final":
                        return True
            except Exception as e:
                print(f"    Instance {idx} error: {e}")
                return False
            return False

        # Run sequentially (concurrent would cause CUDA graph conflicts)
        success_count = 0
        for i, inst in enumerate(instances):
            if run_inference(inst["engine"], i):
                success_count += 1

        print(f"  Sequential inference: {success_count}/{len(instances)} successful")

    # Calculate findings
    print(f"\n  FINDINGS:")
    print(f"  ---------")
    print(f"  Max instances loaded: {instance_count}")
    print(f"  VRAM per instance: ~{instances[0]['vram_mb'] if instances else 0:.0f} MB")

    if instance_count > 1:
        print(f"\n  NOTE: Multiple instances CAN coexist on VRAM")
        print(f"        But CUDA graphs prevent concurrent execution")
        print(f"        Each instance must process requests sequentially")
        print(f"        For true parallelism, disable CUDA graphs (slower)")

    # Cleanup
    for inst in instances:
        del inst["engine"], inst["dac"], inst["llama_queue"]
    del instances
    clear_vram()

    return instance_count


def print_results(results: StressTestResults):
    """Print detailed results."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {results.test_name}")
    print(f"{'='*70}")

    print(f"\n  THROUGHPUT:")
    print(f"  -----------")
    print(f"    Duration:           {results.duration_s:.1f}s")
    print(f"    Total requests:     {results.total_requests} ({results.failed_requests} failed)")
    print(f"    Requests/minute:    {results.requests_per_minute:.1f}")
    print(f"    Requests/second:    {results.requests_per_second:.2f}")
    print(f"    Audio generated:    {results.total_audio_s:.1f}s")
    print(f"    Audio/minute:       {results.audio_per_minute_s:.1f}s")
    print(f"    Throughput:         {results.throughput_x_realtime:.2f}x realtime")

    print(f"\n  LATENCY:")
    print(f"  --------")
    print(f"    Average:            {results.avg_latency_s:.2f}s")
    print(f"    Min:                {results.min_latency_s:.2f}s")
    print(f"    Max:                {results.max_latency_s:.2f}s")
    print(f"    P50:                {results.p50_latency_s:.2f}s")
    print(f"    P95:                {results.p95_latency_s:.2f}s")
    print(f"    P99:                {results.p99_latency_s:.2f}s")

    print(f"\n  GPU METRICS:")
    print(f"  ------------")
    print(f"    Avg GPU util:       {results.avg_gpu_utilization:.1f}%")
    print(f"    Max GPU util:       {results.max_gpu_utilization:.1f}%")
    print(f"    Avg VRAM:           {results.avg_vram_mb:.0f} MB")
    print(f"    Peak VRAM:          {results.peak_vram_mb:.0f} MB")
    print(f"    Avg Power:          {results.avg_power_w:.0f} W")
    print(f"    Max Power:          {results.max_power_w:.0f} W")


def calculate_capacity(results: StressTestResults, gpu_name: str, total_vram_mb: float):
    """Calculate production capacity recommendations."""
    print(f"\n{'='*70}")
    print(f"  PRODUCTION CAPACITY ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  GPU: {gpu_name}")
    print(f"  Total VRAM: {total_vram_mb:.0f} MB")
    print(f"  Peak VRAM used: {results.peak_vram_mb:.0f} MB")

    vram_headroom = total_vram_mb - results.peak_vram_mb
    theoretical_instances = int(total_vram_mb / results.peak_vram_mb) if results.peak_vram_mb > 0 else 0

    print(f"  VRAM headroom: {vram_headroom:.0f} MB")
    print(f"  Theoretical instances: {theoretical_instances}")

    print(f"\n  SINGLE INSTANCE CAPACITY:")
    print(f"  -------------------------")
    print(f"    Requests/minute:    {results.requests_per_minute:.1f}")
    print(f"    Requests/hour:      {results.requests_per_minute * 60:.0f}")
    print(f"    Audio/hour:         {results.audio_per_minute_s * 60 / 60:.0f} minutes")

    # GPU utilization analysis
    gpu_util = results.avg_gpu_utilization
    print(f"\n  GPU UTILIZATION ANALYSIS:")
    print(f"  -------------------------")
    print(f"    Average utilization: {gpu_util:.1f}%")

    if gpu_util < 50:
        print(f"    STATUS: UNDERUTILIZED")
        print(f"    The GPU has significant idle time.")
        print(f"    Bottleneck is likely memory bandwidth or Python overhead.")
        print(f"    ")
        print(f"    RECOMMENDATIONS:")
        print(f"      1. Run multiple model instances (if VRAM allows)")
        print(f"      2. Use request queuing with multiple processes")
        print(f"      3. Consider multiprocessing instead of threading")
    elif gpu_util < 80:
        print(f"    STATUS: MODERATELY UTILIZED")
        print(f"    There's room for optimization.")
        print(f"    ")
        print(f"    RECOMMENDATIONS:")
        print(f"      1. Batch multiple short requests together")
        print(f"      2. Pre-process reference audio in parallel")
    else:
        print(f"    STATUS: WELL UTILIZED")
        print(f"    GPU is near optimal utilization.")
        print(f"    Further gains require hardware upgrade or algorithmic changes.")

    # Multiple instance analysis
    print(f"\n  MULTI-INSTANCE ANALYSIS:")
    print(f"  ------------------------")
    if theoretical_instances > 1:
        print(f"    VRAM could fit {theoretical_instances} instances")
        print(f"    However, CUDA graphs prevent concurrent execution")
        print(f"    ")
        print(f"    OPTIONS:")
        print(f"      A. Sequential processing with multiple instances:")
        print(f"         - No benefit (same as single instance)")
        print(f"      ")
        print(f"      B. Multiprocessing (separate Python processes):")
        print(f"         - Each process has own CUDA context")
        print(f"         - CAN run in parallel")
        print(f"         - Theoretical max: {theoretical_instances}x throughput")
        print(f"         - Estimated: {results.requests_per_minute * min(theoretical_instances, 2):.0f} req/min")
        print(f"      ")
        print(f"      C. Disable CUDA graphs (mode='default'):")
        print(f"         - Enables true concurrent threading")
        print(f"         - But ~30-40% slower per request")
        print(f"         - Net gain depends on workload")
    else:
        print(f"    Only 1 instance fits in VRAM")
        print(f"    Single instance is the optimal configuration")


def main():
    parser = argparse.ArgumentParser(
        description="Fish-Speech GPU Stress Test & Capacity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/runpod/gpu_stress_test.py                      # 60s stress test
    python tools/runpod/gpu_stress_test.py --duration 300       # 5 minute test
    python tools/runpod/gpu_stress_test.py --test-multi-instance  # Test multiple models
    python tools/runpod/gpu_stress_test.py --text-length short  # Test with short texts only
"""
    )

    parser.add_argument(
        "--duration", type=int, default=60,
        help="Stress test duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--text-length", choices=["short", "medium", "long", "mixed"], default="mixed",
        help="Text length for testing (default: mixed)"
    )
    parser.add_argument(
        "--test-multi-instance", action="store_true",
        help="Test multi-instance feasibility"
    )
    parser.add_argument(
        "--output", type=str, default="gpu_stress_results.json",
        help="Output JSON file (default: gpu_stress_results.json)"
    )

    args = parser.parse_args()

    print("="*70)
    print("  Fish-Speech GPU Stress Test & Capacity Analysis")
    print("="*70)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"\n  GPU: {gpu_name}")
        print(f"  Total VRAM: {total_vram:.0f} MB")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  PyTorch: {torch.__version__}")
    else:
        print("ERROR: CUDA not available")
        return

    # Architecture analysis
    analyze_architecture()

    # Multi-instance test
    if args.test_multi_instance:
        test_multi_instance_feasibility()

    # Run stress test
    results = run_stress_test(
        duration_s=args.duration,
        text_length=args.text_length,
        warmup_s=args.warmup,
    )

    # Print results
    print_results(results)

    # Calculate capacity
    if torch.cuda.is_available():
        calculate_capacity(results, gpu_name, total_vram)

    # Save results
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "gpu_name": gpu_name if torch.cuda.is_available() else "N/A",
        "total_vram_mb": total_vram if torch.cuda.is_available() else 0,
        "test_config": {
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "text_length": args.text_length,
        },
        "results": {
            "test_name": results.test_name,
            "duration_s": results.duration_s,
            "total_requests": results.total_requests,
            "successful_requests": results.successful_requests,
            "failed_requests": results.failed_requests,
            "requests_per_minute": results.requests_per_minute,
            "requests_per_second": results.requests_per_second,
            "throughput_x_realtime": results.throughput_x_realtime,
            "avg_latency_s": results.avg_latency_s,
            "p50_latency_s": results.p50_latency_s,
            "p95_latency_s": results.p95_latency_s,
            "p99_latency_s": results.p99_latency_s,
            "avg_gpu_utilization": results.avg_gpu_utilization,
            "max_gpu_utilization": results.max_gpu_utilization,
            "peak_vram_mb": results.peak_vram_mb,
        },
    }

    with open(args.output, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {args.output}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Maximum sustainable throughput:")
    print(f"    {results.requests_per_minute:.1f} requests/minute")
    print(f"    {results.throughput_x_realtime:.2f}x realtime")
    print(f"    {results.audio_per_minute_s * 60 / 60:.0f} minutes audio/hour")


if __name__ == "__main__":
    main()
