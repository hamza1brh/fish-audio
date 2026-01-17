"""
Production Load Test for Fish-Speech
Simulates concurrent users and measures throughput under sustained load.

Run: python tools/runpod/load_test.py --duration 60 --concurrency 4
"""

import argparse
import gc
import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch


@dataclass
class RequestResult:
    request_id: int
    text: str
    start_time: float
    end_time: float
    latency_s: float
    audio_duration_s: float
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestResults:
    duration_s: float
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_audio_generated_s: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    min_latency_s: float
    max_latency_s: float
    requests_per_minute: float
    audio_minutes_per_minute: float
    avg_rtf: float
    peak_vram_mb: float


# Test texts of varying lengths
TEST_TEXTS = [
    # Short (~2-3s audio)
    "Hello, how are you today?",
    "Welcome to our service.",
    "Thank you for calling.",

    # Medium (~5-8s audio)
    "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system.",
    "Please hold while we connect you to the next available representative.",
    "Your order has been confirmed and will be shipped within two business days.",

    # Long (~10-15s audio)
    "Artificial intelligence has transformed the way we interact with technology. From voice assistants to autonomous vehicles, AI systems are becoming increasingly integrated into our daily lives.",
    "Thank you for choosing our service. Your satisfaction is our top priority. If you have any questions or concerns, please don't hesitate to reach out to our customer support team.",

    # Multi-language
    "Hello and welcome. 今天天气很好。Have a great day!",
    "Bonjour! Welcome to our international service center.",
]


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1e9,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def clear_vram():
    """Clear VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LoadTester:
    """Simulates production load on the TTS system."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/openaudio-s1-mini",
        runtime_int8: bool = True,
        dac_int8: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.runtime_int8 = runtime_int8
        self.dac_int8 = dac_int8
        self.engine = None
        self.results: list[RequestResult] = []
        self.results_lock = threading.Lock()
        self.request_counter = 0
        self.counter_lock = threading.Lock()

    def setup(self):
        """Initialize the TTS engine."""
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.dac.inference import load_model as load_dac_model
        from fish_speech.inference_engine import TTSInferenceEngine

        print("Setting up TTS engine...")
        print(f"  Runtime INT8: {self.runtime_int8}")
        print(f"  DAC INT8: {self.dac_int8}")

        clear_vram()

        # Load models
        precision = torch.bfloat16
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_path,
            device="cuda",
            precision=precision,
            compile=False,
            runtime_int8=self.runtime_int8,
        )

        self.dac = load_dac_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(Path(self.checkpoint_path) / "codec.pth"),
            device="cuda",
            quantize_int8=self.dac_int8,
        )

        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.dac,
            precision=precision,
            compile=False,
        )

        # Warmup
        print("Warming up...")
        from fish_speech.utils.schema import ServeTTSRequest
        req = ServeTTSRequest(text="Warmup.", max_new_tokens=256, streaming=False)
        for _ in self.engine.inference(req):
            pass

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print("Engine ready!")

    def process_request(self, text: str) -> RequestResult:
        """Process a single TTS request."""
        from fish_speech.utils.schema import ServeTTSRequest

        with self.counter_lock:
            request_id = self.request_counter
            self.request_counter += 1

        start_time = time.perf_counter()
        audio_duration = 0.0
        success = False
        error = None

        try:
            req = ServeTTSRequest(
                text=text,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                streaming=False,
            )

            for result in self.engine.inference(req):
                if result.code == "final" and result.audio:
                    sample_rate, audio_data = result.audio
                    audio_duration = len(audio_data) / sample_rate
                    success = True
                elif result.code == "error":
                    error = str(result.error)

        except Exception as e:
            error = str(e)

        end_time = time.perf_counter()
        latency = end_time - start_time

        return RequestResult(
            request_id=request_id,
            text=text[:50] + "..." if len(text) > 50 else text,
            start_time=start_time,
            end_time=end_time,
            latency_s=latency,
            audio_duration_s=audio_duration,
            success=success,
            error=error,
        )

    def worker(self, request_queue: queue.Queue, stop_event: threading.Event):
        """Worker thread that processes requests from queue."""
        while not stop_event.is_set():
            try:
                text = request_queue.get(timeout=0.1)
                result = self.process_request(text)

                with self.results_lock:
                    self.results.append(result)

                # Print progress
                status = "OK" if result.success else f"FAIL: {result.error}"
                print(f"  [{result.request_id:04d}] {result.latency_s:.2f}s | {result.audio_duration_s:.2f}s audio | {status}")

                request_queue.task_done()
            except queue.Empty:
                continue

    def run_load_test(
        self,
        duration_s: float = 60,
        concurrency: int = 1,
        requests_per_second: Optional[float] = None,
    ) -> LoadTestResults:
        """
        Run sustained load test.

        Args:
            duration_s: How long to run the test
            concurrency: Number of concurrent workers (for queue processing)
            requests_per_second: Target RPS (None = as fast as possible)
        """
        print(f"\n{'='*60}")
        print(f"Starting Load Test")
        print(f"{'='*60}")
        print(f"  Duration: {duration_s}s")
        print(f"  Concurrency: {concurrency}")
        print(f"  Target RPS: {requests_per_second or 'unlimited'}")
        print()

        self.results = []
        self.request_counter = 0

        request_queue = queue.Queue()
        stop_event = threading.Event()

        # Start workers
        workers = []
        for _ in range(concurrency):
            t = threading.Thread(target=self.worker, args=(request_queue, stop_event))
            t.daemon = True
            t.start()
            workers.append(t)

        # Reset peak VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Generate requests for the duration
        test_start = time.perf_counter()
        last_request_time = test_start

        while time.perf_counter() - test_start < duration_s:
            # Select random text
            text = random.choice(TEST_TEXTS)
            request_queue.put(text)

            # Rate limiting if specified
            if requests_per_second:
                target_interval = 1.0 / requests_per_second
                elapsed = time.perf_counter() - last_request_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                last_request_time = time.perf_counter()

        # Wait for queue to empty
        print("\nWaiting for pending requests to complete...")
        request_queue.join()

        # Stop workers
        stop_event.set()
        for t in workers:
            t.join(timeout=1.0)

        test_end = time.perf_counter()
        actual_duration = test_end - test_start

        # Get peak VRAM
        peak_vram = 0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e6

        # Calculate statistics
        return self._calculate_results(actual_duration, concurrency, peak_vram)

    def _calculate_results(
        self,
        duration_s: float,
        concurrency: int,
        peak_vram_mb: float,
    ) -> LoadTestResults:
        """Calculate load test statistics."""
        if not self.results:
            return LoadTestResults(
                duration_s=duration_s,
                concurrency=concurrency,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_audio_generated_s=0,
                avg_latency_s=0,
                p50_latency_s=0,
                p95_latency_s=0,
                p99_latency_s=0,
                min_latency_s=0,
                max_latency_s=0,
                requests_per_minute=0,
                audio_minutes_per_minute=0,
                avg_rtf=0,
                peak_vram_mb=peak_vram_mb,
            )

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        latencies = [r.latency_s for r in successful] if successful else [0]
        audio_durations = [r.audio_duration_s for r in successful]

        total_audio = sum(audio_durations)
        total_latency = sum(latencies)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = np.percentile(sorted_latencies, 50) if sorted_latencies else 0
        p95 = np.percentile(sorted_latencies, 95) if sorted_latencies else 0
        p99 = np.percentile(sorted_latencies, 99) if sorted_latencies else 0

        # Throughput
        requests_per_min = len(self.results) / duration_s * 60
        audio_min_per_min = total_audio / duration_s

        # RTF
        avg_rtf = total_latency / total_audio if total_audio > 0 else 0

        return LoadTestResults(
            duration_s=duration_s,
            concurrency=concurrency,
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_audio_generated_s=total_audio,
            avg_latency_s=np.mean(latencies) if latencies else 0,
            p50_latency_s=p50,
            p95_latency_s=p95,
            p99_latency_s=p99,
            min_latency_s=min(latencies) if latencies else 0,
            max_latency_s=max(latencies) if latencies else 0,
            requests_per_minute=requests_per_min,
            audio_minutes_per_minute=audio_min_per_min,
            avg_rtf=avg_rtf,
            peak_vram_mb=peak_vram_mb,
        )

    def cleanup(self):
        """Cleanup resources."""
        del self.engine, self.dac, self.llama_queue
        clear_vram()


def print_results(results: LoadTestResults):
    """Print load test results."""
    print(f"\n{'='*60}")
    print("LOAD TEST RESULTS")
    print(f"{'='*60}")

    print(f"\n## Test Configuration")
    print(f"  Duration: {results.duration_s:.1f}s")
    print(f"  Concurrency: {results.concurrency}")

    print(f"\n## Request Statistics")
    print(f"  Total requests: {results.total_requests}")
    print(f"  Successful: {results.successful_requests} ({results.successful_requests/results.total_requests*100:.1f}%)" if results.total_requests > 0 else "  Successful: 0")
    print(f"  Failed: {results.failed_requests}")

    print(f"\n## Latency")
    print(f"  Average: {results.avg_latency_s:.2f}s")
    print(f"  P50: {results.p50_latency_s:.2f}s")
    print(f"  P95: {results.p95_latency_s:.2f}s")
    print(f"  P99: {results.p99_latency_s:.2f}s")
    print(f"  Min: {results.min_latency_s:.2f}s")
    print(f"  Max: {results.max_latency_s:.2f}s")

    print(f"\n## Throughput")
    print(f"  Requests/minute: {results.requests_per_minute:.1f}")
    print(f"  Audio generated: {results.total_audio_generated_s:.1f}s total")
    print(f"  Audio minutes/minute: {results.audio_minutes_per_minute:.2f}")
    print(f"  RTF (avg): {results.avg_rtf:.3f}")

    print(f"\n## Resources")
    print(f"  Peak VRAM: {results.peak_vram_mb:.0f} MB")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Fish-Speech Production Load Test")
    parser.add_argument("--duration", type=float, default=60, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--rps", type=float, default=None, help="Target requests per second (None = unlimited)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/openaudio-s1-mini", help="Checkpoint path")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 quantization")
    parser.add_argument("--no-dac-int8", action="store_true", help="Disable DAC INT8 quantization")
    parser.add_argument("--output", type=str, default="load_test_results.json", help="Output JSON file")
    args = parser.parse_args()

    # GPU info
    print("=" * 60)
    print("Fish-Speech Production Load Test")
    print("=" * 60)

    gpu_info = get_gpu_info()
    print("\nGPU Information:")
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")

    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"\nError: Checkpoint not found at {args.checkpoint}")
        print("Run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
        return

    # Create tester
    tester = LoadTester(
        checkpoint_path=args.checkpoint,
        runtime_int8=not args.no_int8,
        dac_int8=not args.no_dac_int8,
    )

    try:
        # Setup
        tester.setup()

        # Run test
        results = tester.run_load_test(
            duration_s=args.duration,
            concurrency=args.concurrency,
            requests_per_second=args.rps,
        )

        # Print results
        print_results(results)

        # Save results
        output = {
            "gpu_info": gpu_info,
            "config": {
                "duration_s": args.duration,
                "concurrency": args.concurrency,
                "target_rps": args.rps,
                "runtime_int8": not args.no_int8,
                "dac_int8": not args.no_dac_int8,
            },
            "results": {
                "total_requests": results.total_requests,
                "successful_requests": results.successful_requests,
                "failed_requests": results.failed_requests,
                "total_audio_generated_s": results.total_audio_generated_s,
                "avg_latency_s": results.avg_latency_s,
                "p50_latency_s": results.p50_latency_s,
                "p95_latency_s": results.p95_latency_s,
                "p99_latency_s": results.p99_latency_s,
                "min_latency_s": results.min_latency_s,
                "max_latency_s": results.max_latency_s,
                "requests_per_minute": results.requests_per_minute,
                "audio_minutes_per_minute": results.audio_minutes_per_minute,
                "avg_rtf": results.avg_rtf,
                "peak_vram_mb": results.peak_vram_mb,
            },
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
