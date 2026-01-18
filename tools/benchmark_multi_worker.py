"""
Thorough Benchmark Script for Fish Speech Multi-Worker Setup

Features:
- GPU metrics collection (VRAM, utilization, power, temperature)
- Stress test modes (sustained load, ramp-up, burst, endurance)
- Real-Time Factor (RTF) calculation
- Worker health monitoring
- Audio quality verification
- HTML report generation with charts

Usage:
    # Basic test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --num-requests 10

    # Thorough stress test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 \
        --duration 120 --concurrent 4 --gpu-metrics --output-html report.html

    # Compare 1 vs 2 workers
    python tools/benchmark_multi_worker.py --compare \
        --single-endpoint http://localhost:8080 \
        --multi-endpoint http://localhost:8000 \
        --num-requests 20 --concurrent 4

    # Ramp-up test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --ramp-up

    # Burst test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --burst --burst-size 20
"""

import argparse
import asyncio
import io
import json
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==============================================================================
# Test Configuration
# ==============================================================================

# Test texts of varying lengths
TEST_TEXTS = {
    "short": [
        "Hello, this is a test.",
        "The quick brown fox jumps.",
        "Welcome to Fish Speech.",
    ],
    "medium": [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog. This is a longer sentence to test the system's performance with more text.",
        "Voice synthesis technology has made remarkable progress in recent years, enabling more natural sounding speech.",
    ],
    "long": [
        "In a world where technology advances rapidly, artificial intelligence continues to transform how we interact with machines. Voice synthesis has become remarkably natural sounding, enabling new applications in accessibility and communication.",
        "Today we're going to discuss the fascinating world of neural networks and how they enable machines to understand and generate human speech with incredible accuracy and naturalness. These systems learn patterns from vast amounts of data.",
        "The development of text-to-speech systems has evolved significantly over the past decade. Modern neural approaches can generate speech that is nearly indistinguishable from human voices, opening up new possibilities for content creation.",
    ],
}

# Audio parameters (standard WAV format from Fish Speech)
AUDIO_SAMPLE_RATE = 44100  # Fish Speech default


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class GPUMetrics:
    """GPU metrics from nvidia-smi."""

    timestamp: float
    gpu_name: str = ""
    vram_used_mb: float = 0
    vram_total_mb: float = 0
    gpu_utilization: float = 0
    power_draw_w: float = 0
    temperature_c: float = 0

    @property
    def vram_percent(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb > 0 else 0


@dataclass
class RequestResult:
    """Result of a single TTS request."""

    request_id: int
    success: bool
    latency: float
    text_length: int
    audio_size: int = 0
    audio_duration: float = 0  # seconds
    error: str = ""
    worker_url: str = ""
    timestamp: float = 0
    rtf: float = 0  # Real-time factor


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    endpoint: str
    num_requests: int = 10
    concurrency: int = 1
    duration: int = 0  # seconds, 0 = use num_requests
    warmup: int = 3
    text_lengths: list = field(default_factory=lambda: ["medium"])
    timeout: float = 120.0


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    name: str
    config: BenchmarkConfig
    start_time: datetime
    end_time: datetime
    results: list[RequestResult] = field(default_factory=list)
    gpu_metrics: list[GPUMetrics] = field(default_factory=list)
    worker_stats: dict = field(default_factory=dict)

    @property
    def successful_results(self) -> list[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> list[RequestResult]:
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        return len(self.successful_results) / len(self.results) * 100 if self.results else 0

    @property
    def total_duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


# ==============================================================================
# GPU Metrics Collection
# ==============================================================================


class GPUMetricsCollector:
    """Collects GPU metrics using nvidia-smi."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: list[GPUMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._available = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _collect_metrics(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics."""
        if not self._available:
            return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            line = result.stdout.strip().split("\n")[0]  # First GPU
            parts = [p.strip() for p in line.split(",")]

            return GPUMetrics(
                timestamp=time.time(),
                gpu_name=parts[0] if len(parts) > 0 else "",
                vram_used_mb=float(parts[1]) if len(parts) > 1 and parts[1] != "[N/A]" else 0,
                vram_total_mb=float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else 0,
                gpu_utilization=float(parts[3]) if len(parts) > 3 and parts[3] != "[N/A]" else 0,
                power_draw_w=float(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else 0,
                temperature_c=float(parts[5]) if len(parts) > 5 and parts[5] != "[N/A]" else 0,
            )
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
            return None

    def _collector_loop(self):
        """Background thread for collecting metrics."""
        while self._running:
            metrics = self._collect_metrics()
            if metrics:
                self.metrics.append(metrics)
            time.sleep(self.interval)

    def start(self):
        """Start collecting metrics."""
        if not self._available:
            logger.warning("nvidia-smi not available, GPU metrics disabled")
            return

        self._running = True
        self._thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._thread.start()
        logger.info("GPU metrics collection started")

    def stop(self):
        """Stop collecting metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info(f"GPU metrics collection stopped, {len(self.metrics)} samples collected")

    def get_baseline(self) -> Optional[GPUMetrics]:
        """Get baseline GPU metrics."""
        return self._collect_metrics()

    def get_summary(self) -> dict:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}

        return {
            "gpu_name": self.metrics[0].gpu_name,
            "samples": len(self.metrics),
            "vram": {
                "min_mb": min(m.vram_used_mb for m in self.metrics),
                "max_mb": max(m.vram_used_mb for m in self.metrics),
                "avg_mb": statistics.mean(m.vram_used_mb for m in self.metrics),
                "total_mb": self.metrics[0].vram_total_mb,
            },
            "utilization": {
                "min": min(m.gpu_utilization for m in self.metrics),
                "max": max(m.gpu_utilization for m in self.metrics),
                "avg": statistics.mean(m.gpu_utilization for m in self.metrics),
            },
            "power": {
                "min_w": min(m.power_draw_w for m in self.metrics),
                "max_w": max(m.power_draw_w for m in self.metrics),
                "avg_w": statistics.mean(m.power_draw_w for m in self.metrics),
            },
            "temperature": {
                "min_c": min(m.temperature_c for m in self.metrics),
                "max_c": max(m.temperature_c for m in self.metrics),
                "avg_c": statistics.mean(m.temperature_c for m in self.metrics),
            },
        }


# ==============================================================================
# Audio Verification
# ==============================================================================


def verify_audio(audio_data: bytes) -> tuple[bool, float, str]:
    """
    Verify audio data is valid WAV format and extract duration.

    Returns:
        (is_valid, duration_seconds, error_message)
    """
    if not audio_data or len(audio_data) < 44:  # WAV header is 44 bytes minimum
        return False, 0, "Audio data too small"

    try:
        # Try to parse as WAV
        audio_io = io.BytesIO(audio_data)
        with wave.open(audio_io, "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()

            duration = n_frames / frame_rate

            # Basic sanity checks
            if n_channels < 1 or n_channels > 2:
                return False, 0, f"Invalid channel count: {n_channels}"
            if sample_width < 1 or sample_width > 4:
                return False, 0, f"Invalid sample width: {sample_width}"
            if frame_rate < 8000 or frame_rate > 96000:
                return False, 0, f"Invalid frame rate: {frame_rate}"
            if duration < 0.1 or duration > 300:  # 0.1s to 5 minutes
                return False, 0, f"Invalid duration: {duration:.2f}s"

            return True, duration, ""

    except wave.Error as e:
        return False, 0, f"WAV parse error: {e}"
    except Exception as e:
        return False, 0, f"Audio verification error: {e}"


# ==============================================================================
# TTS Request Handler
# ==============================================================================


async def make_tts_request(
    client: httpx.AsyncClient,
    endpoint: str,
    text: str,
    request_id: int = 0,
    timeout: float = 120.0,
    verify_audio_quality: bool = True,
) -> RequestResult:
    """Make a single TTS request and return detailed result."""
    start_time = time.perf_counter()
    timestamp = time.time()

    try:
        response = await client.post(
            f"{endpoint}/v1/tts",
            json={
                "text": text,
                "format": "wav",
                "temperature": 0.7,
                "top_p": 0.7,
                "repetition_penalty": 1.5,
            },
            timeout=timeout,
        )

        end_time = time.perf_counter()
        latency = end_time - start_time

        if response.status_code == 200:
            audio_data = response.content
            audio_size = len(audio_data)

            # Verify audio quality
            audio_duration = 0
            if verify_audio_quality:
                is_valid, duration, error = verify_audio(audio_data)
                if is_valid:
                    audio_duration = duration
                else:
                    logger.warning(f"Request {request_id}: Audio verification failed - {error}")

            # Calculate RTF (Real-Time Factor)
            rtf = latency / audio_duration if audio_duration > 0 else 0

            return RequestResult(
                request_id=request_id,
                success=True,
                latency=latency,
                text_length=len(text),
                audio_size=audio_size,
                audio_duration=audio_duration,
                timestamp=timestamp,
                rtf=rtf,
            )
        else:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency=latency,
                text_length=len(text),
                error=f"HTTP {response.status_code}: {response.text[:200]}",
                timestamp=timestamp,
            )

    except httpx.TimeoutException:
        return RequestResult(
            request_id=request_id,
            success=False,
            latency=timeout,
            text_length=len(text),
            error="Request timed out",
            timestamp=timestamp,
        )
    except Exception as e:
        end_time = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            success=False,
            latency=end_time - start_time,
            text_length=len(text),
            error=str(e),
            timestamp=timestamp,
        )


# ==============================================================================
# Benchmark Functions
# ==============================================================================


def get_test_texts(text_lengths: list[str]) -> list[str]:
    """Get test texts based on requested lengths."""
    texts = []
    for length in text_lengths:
        if length in TEST_TEXTS:
            texts.extend(TEST_TEXTS[length])
    return texts if texts else TEST_TEXTS["medium"]


async def run_warmup(
    endpoint: str,
    num_warmup: int = 3,
    timeout: float = 120.0,
) -> None:
    """Run warmup requests."""
    if num_warmup <= 0:
        return

    logger.info(f"Running {num_warmup} warmup requests...")
    texts = TEST_TEXTS["short"]

    async with httpx.AsyncClient() as client:
        for i in range(num_warmup):
            text = texts[i % len(texts)]
            result = await make_tts_request(client, endpoint, text, i, timeout, verify_audio_quality=False)
            status = "OK" if result.success else f"FAIL: {result.error}"
            logger.info(f"  Warmup {i+1}/{num_warmup}: {result.latency:.2f}s - {status}")

    logger.info("Warmup completed")


async def benchmark_sequential(
    config: BenchmarkConfig,
    gpu_collector: Optional[GPUMetricsCollector] = None,
) -> BenchmarkResults:
    """Run sequential benchmark (one request at a time)."""
    logger.info(f"Running sequential benchmark: {config.num_requests} requests")

    results = BenchmarkResults(
        name="Sequential",
        config=config,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    texts = get_test_texts(config.text_lengths)

    async with httpx.AsyncClient() as client:
        for i in range(config.num_requests):
            text = texts[i % len(texts)]
            result = await make_tts_request(
                client, config.endpoint, text, i, config.timeout
            )
            results.results.append(result)

            status = "OK" if result.success else f"FAIL"
            rtf_str = f"RTF={result.rtf:.2f}" if result.rtf > 0 else ""
            logger.info(f"  [{i+1}/{config.num_requests}] {result.latency:.2f}s {rtf_str} {status}")

    results.end_time = datetime.now()

    if gpu_collector:
        results.gpu_metrics = gpu_collector.metrics.copy()

    return results


async def benchmark_concurrent(
    config: BenchmarkConfig,
    gpu_collector: Optional[GPUMetricsCollector] = None,
) -> BenchmarkResults:
    """Run concurrent benchmark."""
    logger.info(f"Running concurrent benchmark: {config.num_requests} requests, concurrency={config.concurrency}")

    results = BenchmarkResults(
        name=f"Concurrent (c={config.concurrency})",
        config=config,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    texts = get_test_texts(config.text_lengths)
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def limited_request(i: int, text: str, client: httpx.AsyncClient) -> RequestResult:
        nonlocal completed
        async with semaphore:
            result = await make_tts_request(client, config.endpoint, text, i, config.timeout)
            async with lock:
                completed += 1
                status = "OK" if result.success else "FAIL"
                logger.info(f"  [{completed}/{config.num_requests}] Request {i}: {result.latency:.2f}s {status}")
            return result

    async with httpx.AsyncClient() as client:
        tasks = [
            limited_request(i, texts[i % len(texts)], client)
            for i in range(config.num_requests)
        ]
        results.results = await asyncio.gather(*tasks)

    results.end_time = datetime.now()

    if gpu_collector:
        results.gpu_metrics = gpu_collector.metrics.copy()

    return results


async def benchmark_sustained(
    config: BenchmarkConfig,
    duration: int,
    gpu_collector: Optional[GPUMetricsCollector] = None,
) -> BenchmarkResults:
    """Run sustained load benchmark for a specified duration."""
    logger.info(f"Running sustained load test: {duration}s, concurrency={config.concurrency}")

    results = BenchmarkResults(
        name=f"Sustained ({duration}s, c={config.concurrency})",
        config=config,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    texts = get_test_texts(config.text_lengths)
    semaphore = asyncio.Semaphore(config.concurrency)
    request_counter = 0
    completed_counter = 0
    lock = asyncio.Lock()
    stop_time = time.time() + duration

    async def sustained_request(client: httpx.AsyncClient) -> Optional[RequestResult]:
        nonlocal request_counter, completed_counter

        async with lock:
            if time.time() >= stop_time:
                return None
            request_id = request_counter
            request_counter += 1

        text = texts[request_id % len(texts)]

        async with semaphore:
            if time.time() >= stop_time:
                return None

            result = await make_tts_request(client, config.endpoint, text, request_id, config.timeout)

            async with lock:
                completed_counter += 1
                elapsed = time.time() - (stop_time - duration)
                remaining = max(0, stop_time - time.time())
                logger.info(f"  [{completed_counter}] {result.latency:.2f}s | {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")

            return result

    async with httpx.AsyncClient() as client:
        while time.time() < stop_time:
            # Launch batch of concurrent requests
            tasks = [sustained_request(client) for _ in range(config.concurrency)]
            batch_results = await asyncio.gather(*tasks)
            results.results.extend([r for r in batch_results if r is not None])

    results.end_time = datetime.now()

    if gpu_collector:
        results.gpu_metrics = gpu_collector.metrics.copy()

    logger.info(f"Sustained test completed: {len(results.results)} requests in {duration}s")
    return results


async def benchmark_rampup(
    config: BenchmarkConfig,
    concurrency_levels: list[int] = None,
    requests_per_level: int = 5,
    gpu_collector: Optional[GPUMetricsCollector] = None,
) -> list[BenchmarkResults]:
    """Run ramp-up test with increasing concurrency."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8]

    logger.info(f"Running ramp-up test: levels={concurrency_levels}, requests/level={requests_per_level}")

    all_results = []

    for level in concurrency_levels:
        level_config = BenchmarkConfig(
            endpoint=config.endpoint,
            num_requests=requests_per_level,
            concurrency=level,
            text_lengths=config.text_lengths,
            timeout=config.timeout,
        )

        logger.info(f"\n--- Concurrency Level: {level} ---")
        result = await benchmark_concurrent(level_config, gpu_collector)
        all_results.append(result)

    return all_results


async def benchmark_burst(
    config: BenchmarkConfig,
    burst_size: int = 20,
    gpu_collector: Optional[GPUMetricsCollector] = None,
) -> BenchmarkResults:
    """Run burst test - send all requests simultaneously."""
    logger.info(f"Running burst test: {burst_size} simultaneous requests")

    burst_config = BenchmarkConfig(
        endpoint=config.endpoint,
        num_requests=burst_size,
        concurrency=burst_size,  # All at once
        text_lengths=config.text_lengths,
        timeout=config.timeout,
    )

    return await benchmark_concurrent(burst_config, gpu_collector)


# ==============================================================================
# Worker Health Monitoring
# ==============================================================================


async def get_worker_stats(endpoint: str) -> Optional[dict]:
    """Get worker statistics from load balancer."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/v1/workers")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.debug(f"Could not get worker stats: {e}")
    return None


async def check_endpoint_health(endpoint: str) -> tuple[bool, dict]:
    """Check if endpoint is healthy and get info."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/v1/health")
            if response.status_code == 200:
                data = response.json()
                return True, data
            return False, {}
    except Exception as e:
        return False, {"error": str(e)}


# ==============================================================================
# Results Analysis
# ==============================================================================


def analyze_results(results: BenchmarkResults) -> dict:
    """Analyze benchmark results and compute statistics."""
    successful = results.successful_results
    failed = results.failed_results

    if not successful:
        return {
            "name": results.name,
            "total_requests": len(results.results),
            "successful": 0,
            "failed": len(failed),
            "success_rate": 0,
            "errors": [r.error for r in failed[:5]],  # First 5 errors
        }

    latencies = [r.latency for r in successful]
    rtfs = [r.rtf for r in successful if r.rtf > 0]
    audio_durations = [r.audio_duration for r in successful if r.audio_duration > 0]

    analysis = {
        "name": results.name,
        "total_requests": len(results.results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": results.success_rate,
        "total_duration": results.total_duration,
        "latency": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p50": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "p99": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else max(latencies),
        },
        "throughput": {
            "requests_per_second": len(successful) / results.total_duration if results.total_duration > 0 else 0,
            "requests_per_minute": len(successful) / results.total_duration * 60 if results.total_duration > 0 else 0,
        },
    }

    # RTF statistics
    if rtfs:
        analysis["rtf"] = {
            "mean": statistics.mean(rtfs),
            "median": statistics.median(rtfs),
            "min": min(rtfs),
            "max": max(rtfs),
            "realtime_factor": f"{1/statistics.mean(rtfs):.2f}x" if statistics.mean(rtfs) > 0 else "N/A",
        }

    # Audio statistics
    if audio_durations:
        total_audio = sum(audio_durations)
        analysis["audio"] = {
            "total_duration_seconds": total_audio,
            "total_duration_minutes": total_audio / 60,
            "audio_minutes_per_hour": (total_audio / 60) / (results.total_duration / 3600) if results.total_duration > 0 else 0,
        }

    # GPU metrics if available
    if results.gpu_metrics:
        gpu_summary = GPUMetricsCollector(0).get_summary.__func__(
            type("obj", (), {"metrics": results.gpu_metrics})()
        )
        if gpu_summary:
            analysis["gpu"] = gpu_summary

    return analysis


def print_results(analysis: dict):
    """Pretty print benchmark results."""
    print()
    print("=" * 70)
    print(f"  {analysis['name']}")
    print("=" * 70)
    print(f"  Requests: {analysis['successful']}/{analysis['total_requests']} ({analysis['success_rate']:.1f}% success)")

    if analysis.get("total_duration"):
        print(f"  Duration: {analysis['total_duration']:.2f}s")

    if analysis.get("errors"):
        print(f"  Errors: {analysis['errors'][:2]}")
        return

    # Latency
    lat = analysis.get("latency", {})
    if lat:
        print()
        print("  Latency:")
        print(f"    Mean:   {lat.get('mean', 0):.2f}s")
        print(f"    Median: {lat.get('median', 0):.2f}s")
        print(f"    Min:    {lat.get('min', 0):.2f}s")
        print(f"    Max:    {lat.get('max', 0):.2f}s")
        print(f"    P95:    {lat.get('p95', 0):.2f}s")
        if lat.get('stdev', 0) > 0:
            print(f"    StdDev: {lat['stdev']:.2f}s")

    # Throughput
    thr = analysis.get("throughput", {})
    if thr:
        print()
        print("  Throughput:")
        print(f"    Requests/sec: {thr.get('requests_per_second', 0):.2f}")
        print(f"    Requests/min: {thr.get('requests_per_minute', 0):.1f}")

    # RTF
    rtf = analysis.get("rtf", {})
    if rtf:
        print()
        print("  Real-Time Factor (RTF):")
        print(f"    Mean RTF:     {rtf.get('mean', 0):.3f}")
        print(f"    Realtime:     {rtf.get('realtime_factor', 'N/A')}")

    # Audio
    audio = analysis.get("audio", {})
    if audio:
        print()
        print("  Audio Generated:")
        print(f"    Total:        {audio.get('total_duration_minutes', 0):.2f} minutes")
        print(f"    Rate:         {audio.get('audio_minutes_per_hour', 0):.1f} audio-min/hour")

    # GPU
    gpu = analysis.get("gpu", {})
    if gpu:
        print()
        print(f"  GPU Metrics ({gpu.get('gpu_name', 'Unknown')}):")
        vram = gpu.get("vram", {})
        if vram:
            print(f"    VRAM Used:    {vram.get('avg_mb', 0):.0f} MB avg ({vram.get('max_mb', 0):.0f} MB peak)")
        util = gpu.get("utilization", {})
        if util:
            print(f"    Utilization:  {util.get('avg', 0):.1f}% avg ({util.get('max', 0):.1f}% peak)")
        power = gpu.get("power", {})
        if power and power.get('avg_w', 0) > 0:
            print(f"    Power Draw:   {power.get('avg_w', 0):.1f}W avg")
        temp = gpu.get("temperature", {})
        if temp and temp.get('avg_c', 0) > 0:
            print(f"    Temperature:  {temp.get('avg_c', 0):.1f}°C avg")

    print()


# ==============================================================================
# HTML Report Generation
# ==============================================================================


def generate_html_report(
    all_analyses: list[dict],
    output_path: str,
    comparison_results: Optional[dict] = None,
) -> None:
    """Generate detailed HTML report with charts."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data for charts
    latency_data = []
    throughput_data = []
    rtf_data = []

    for analysis in all_analyses:
        name = analysis.get("name", "Unknown")
        lat = analysis.get("latency", {})
        thr = analysis.get("throughput", {})
        rtf = analysis.get("rtf", {})

        latency_data.append({
            "name": name,
            "mean": lat.get("mean", 0),
            "p95": lat.get("p95", 0),
            "min": lat.get("min", 0),
            "max": lat.get("max", 0),
        })

        throughput_data.append({
            "name": name,
            "req_per_sec": thr.get("requests_per_second", 0),
            "req_per_min": thr.get("requests_per_minute", 0),
        })

        if rtf:
            rtf_data.append({
                "name": name,
                "rtf": rtf.get("mean", 0),
            })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fish Speech Multi-Worker Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .stat {{
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-unit {{
            font-size: 0.8em;
            color: #888;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        .success {{
            color: #27ae60;
        }}
        .warning {{
            color: #f39c12;
        }}
        .error {{
            color: #e74c3c;
        }}
        .comparison {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .comparison h2 {{
            color: white;
            border-bottom-color: rgba(255,255,255,0.3);
        }}
        .comparison .stat {{
            background: rgba(255,255,255,0.2);
        }}
        .comparison .stat-value {{
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fish Speech Multi-Worker Benchmark Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
"""

    # Comparison summary if available
    if comparison_results:
        html_content += f"""
        <div class="card comparison">
            <h2>Comparison Summary</h2>
            <div class="grid">
                <div class="stat">
                    <div class="stat-label">Single Worker</div>
                    <div class="stat-value">{comparison_results.get('single_throughput', 0):.1f}</div>
                    <div class="stat-unit">req/min</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Multi-Worker</div>
                    <div class="stat-value">{comparison_results.get('multi_throughput', 0):.1f}</div>
                    <div class="stat-unit">req/min</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Speedup</div>
                    <div class="stat-value">{comparison_results.get('speedup', 0):.2f}x</div>
                    <div class="stat-unit">improvement</div>
                </div>
            </div>
        </div>
"""

    # Results table
    html_content += """
        <div class="card">
            <h2>Benchmark Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Requests</th>
                        <th>Success Rate</th>
                        <th>Mean Latency</th>
                        <th>P95 Latency</th>
                        <th>Throughput</th>
                        <th>RTF</th>
                    </tr>
                </thead>
                <tbody>
"""

    for analysis in all_analyses:
        lat = analysis.get("latency", {})
        thr = analysis.get("throughput", {})
        rtf = analysis.get("rtf", {})
        success_class = "success" if analysis.get("success_rate", 0) >= 99 else "warning" if analysis.get("success_rate", 0) >= 90 else "error"

        html_content += f"""
                    <tr>
                        <td>{analysis.get('name', 'Unknown')}</td>
                        <td>{analysis.get('successful', 0)}/{analysis.get('total_requests', 0)}</td>
                        <td class="{success_class}">{analysis.get('success_rate', 0):.1f}%</td>
                        <td>{lat.get('mean', 0):.2f}s</td>
                        <td>{lat.get('p95', 0):.2f}s</td>
                        <td>{thr.get('requests_per_minute', 0):.1f} req/min</td>
                        <td>{rtf.get('mean', 0):.3f}</td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>
"""

    # Charts
    html_content += f"""
        <div class="card">
            <h2>Latency Chart</h2>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Throughput Chart</h2>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>

        <script>
            // Latency Chart
            const latencyCtx = document.getElementById('latencyChart').getContext('2d');
            new Chart(latencyCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps([d['name'] for d in latency_data])},
                    datasets: [
                        {{
                            label: 'Mean Latency',
                            data: {json.dumps([d['mean'] for d in latency_data])},
                            backgroundColor: 'rgba(52, 152, 219, 0.8)',
                        }},
                        {{
                            label: 'P95 Latency',
                            data: {json.dumps([d['p95'] for d in latency_data])},
                            backgroundColor: 'rgba(231, 76, 60, 0.8)',
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Latency (seconds)'
                            }}
                        }}
                    }}
                }}
            }});

            // Throughput Chart
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            new Chart(throughputCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps([d['name'] for d in throughput_data])},
                    datasets: [{{
                        label: 'Requests per Minute',
                        data: {json.dumps([d['req_per_min'] for d in throughput_data])},
                        backgroundColor: 'rgba(46, 204, 113, 0.8)',
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Requests/Minute'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
"""

    # GPU Metrics section if available
    for analysis in all_analyses:
        gpu = analysis.get("gpu")
        if gpu:
            vram = gpu.get("vram", {})
            util = gpu.get("utilization", {})
            power = gpu.get("power", {})
            temp = gpu.get("temperature", {})

            html_content += f"""
        <div class="card">
            <h2>GPU Metrics - {analysis.get('name', 'Unknown')}</h2>
            <div class="grid">
                <div class="stat">
                    <div class="stat-label">GPU</div>
                    <div class="stat-value">{gpu.get('gpu_name', 'Unknown')}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">VRAM Used</div>
                    <div class="stat-value">{vram.get('avg_mb', 0):.0f}</div>
                    <div class="stat-unit">MB avg (peak: {vram.get('max_mb', 0):.0f} MB)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">GPU Utilization</div>
                    <div class="stat-value">{util.get('avg', 0):.1f}%</div>
                    <div class="stat-unit">avg (peak: {util.get('max', 0):.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Power Draw</div>
                    <div class="stat-value">{power.get('avg_w', 0):.1f}W</div>
                    <div class="stat-unit">avg</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Temperature</div>
                    <div class="stat-value">{temp.get('avg_c', 0):.1f}°C</div>
                    <div class="stat-unit">avg (peak: {temp.get('max_c', 0):.1f}°C)</div>
                </div>
            </div>
        </div>
"""
            break  # Only show GPU metrics once

    html_content += """
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved to: {output_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Thorough benchmark for Fish Speech multi-worker setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic benchmark
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000

    # Stress test with GPU metrics
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 \\
        --duration 120 --concurrent 4 --gpu-metrics --output-html report.html

    # Ramp-up test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --ramp-up

    # Burst test
    python tools/benchmark_multi_worker.py --endpoint http://localhost:8000 --burst

    # Compare single vs multi-worker
    python tools/benchmark_multi_worker.py --compare \\
        --single-endpoint http://localhost:8080 \\
        --multi-endpoint http://localhost:8000
        """,
    )

    # Basic options
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080",
        help="API endpoint to benchmark (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests per benchmark (default: 10)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=0,
        help="Concurrency level, 0 = sequential only (default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup requests (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--text-lengths",
        type=str,
        default="medium",
        help="Comma-separated text lengths: short,medium,long (default: medium)",
    )

    # Stress test options
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Sustained load test duration in seconds (0 = disabled)",
    )
    parser.add_argument(
        "--ramp-up",
        action="store_true",
        help="Enable ramp-up test (increasing concurrency)",
    )
    parser.add_argument(
        "--ramp-levels",
        type=str,
        default="1,2,4,8",
        help="Concurrency levels for ramp-up test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Enable burst test (all requests at once)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=20,
        help="Number of simultaneous requests for burst test (default: 20)",
    )

    # Comparison options
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare single vs multi-worker",
    )
    parser.add_argument(
        "--single-endpoint",
        type=str,
        default="http://localhost:8080",
        help="Single worker endpoint (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--multi-endpoint",
        type=str,
        default="http://localhost:8000",
        help="Multi-worker endpoint (default: http://localhost:8000)",
    )

    # Monitoring options
    parser.add_argument(
        "--gpu-metrics",
        action="store_true",
        help="Collect GPU metrics via nvidia-smi",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        help="Save HTML report to file",
    )

    args = parser.parse_args()

    # Parse text lengths
    text_lengths = [t.strip() for t in args.text_lengths.split(",")]

    print()
    print("=" * 70)
    print("  Fish Speech Multi-Worker Benchmark - Thorough Test")
    print("=" * 70)
    print()

    # Initialize GPU metrics collector
    gpu_collector = None
    if args.gpu_metrics:
        gpu_collector = GPUMetricsCollector(interval=1.0)
        baseline = gpu_collector.get_baseline()
        if baseline:
            print(f"GPU: {baseline.gpu_name}")
            print(f"VRAM: {baseline.vram_used_mb:.0f} / {baseline.vram_total_mb:.0f} MB ({baseline.vram_percent:.1f}%)")
            print(f"Utilization: {baseline.gpu_utilization:.1f}%")
            print()

    all_results: list[BenchmarkResults] = []
    all_analyses: list[dict] = []
    comparison_results = None

    if args.compare:
        # Comparison mode
        print("Mode: Comparison (Single vs Multi-Worker)")
        print()

        # Check endpoints
        single_ok, single_info = await check_endpoint_health(args.single_endpoint)
        multi_ok, multi_info = await check_endpoint_health(args.multi_endpoint)

        if not single_ok:
            logger.warning(f"Single worker endpoint not available: {args.single_endpoint}")
        if not multi_ok:
            logger.warning(f"Multi-worker endpoint not available: {args.multi_endpoint}")

        if single_ok:
            print(f"\n{'='*70}")
            print(f"  Single Worker ({args.single_endpoint})")
            print(f"{'='*70}")

            await run_warmup(args.single_endpoint, args.warmup, args.timeout)

            if gpu_collector:
                gpu_collector.start()

            config = BenchmarkConfig(
                endpoint=args.single_endpoint,
                num_requests=args.num_requests,
                concurrency=args.concurrent if args.concurrent > 0 else 1,
                text_lengths=text_lengths,
                timeout=args.timeout,
            )

            result = await benchmark_sequential(config, gpu_collector)
            all_results.append(result)
            analysis = analyze_results(result)
            all_analyses.append(analysis)
            print_results(analysis)

            if args.concurrent > 0:
                result = await benchmark_concurrent(config, gpu_collector)
                all_results.append(result)
                analysis = analyze_results(result)
                all_analyses.append(analysis)
                print_results(analysis)

            if gpu_collector:
                gpu_collector.stop()

        if multi_ok:
            print(f"\n{'='*70}")
            print(f"  Multi-Worker ({args.multi_endpoint})")
            print(f"{'='*70}")

            # Get worker stats
            worker_stats = await get_worker_stats(args.multi_endpoint)
            if worker_stats:
                print(f"Workers: {len(worker_stats.get('workers', []))}")
                print(f"Strategy: {worker_stats.get('strategy', 'unknown')}")

            await run_warmup(args.multi_endpoint, args.warmup, args.timeout)

            if gpu_collector:
                gpu_collector.metrics = []
                gpu_collector.start()

            config = BenchmarkConfig(
                endpoint=args.multi_endpoint,
                num_requests=args.num_requests,
                concurrency=args.concurrent if args.concurrent > 0 else 1,
                text_lengths=text_lengths,
                timeout=args.timeout,
            )

            result = await benchmark_sequential(config, gpu_collector)
            all_results.append(result)
            analysis = analyze_results(result)
            all_analyses.append(analysis)
            print_results(analysis)

            if args.concurrent > 0:
                result = await benchmark_concurrent(config, gpu_collector)
                all_results.append(result)
                analysis = analyze_results(result)
                all_analyses.append(analysis)
                print_results(analysis)

            if gpu_collector:
                gpu_collector.stop()

        # Comparison summary
        if single_ok and multi_ok and len(all_analyses) >= 2:
            print()
            print("=" * 70)
            print("  COMPARISON SUMMARY")
            print("=" * 70)

            single_thr = all_analyses[0].get("throughput", {}).get("requests_per_minute", 0)
            multi_thr = all_analyses[-1].get("throughput", {}).get("requests_per_minute", 0)
            speedup = multi_thr / single_thr if single_thr > 0 else 0

            print(f"  Single worker:  {single_thr:.1f} req/min")
            print(f"  Multi-worker:   {multi_thr:.1f} req/min")
            print(f"  Speedup:        {speedup:.2f}x")
            print()

            comparison_results = {
                "single_throughput": single_thr,
                "multi_throughput": multi_thr,
                "speedup": speedup,
            }

    else:
        # Single endpoint benchmark
        print(f"Endpoint: {args.endpoint}")
        print()

        # Check endpoint
        healthy, info = await check_endpoint_health(args.endpoint)
        if not healthy:
            logger.error(f"Endpoint not available: {args.endpoint}")
            print("\nPlease start the server first:")
            print("  Single worker:  python tools/api_server.py")
            print("  Multi-worker:   python tools/multi_worker_server.py --workers 2 --with-balancer")
            return

        # Show worker stats if available
        worker_stats = await get_worker_stats(args.endpoint)
        if worker_stats:
            print(f"Workers: {len(worker_stats.get('workers', []))}")
            print(f"Strategy: {worker_stats.get('strategy', 'unknown')}")
            print()

        # Warmup
        await run_warmup(args.endpoint, args.warmup, args.timeout)

        # Start GPU monitoring
        if gpu_collector:
            gpu_collector.start()

        config = BenchmarkConfig(
            endpoint=args.endpoint,
            num_requests=args.num_requests,
            concurrency=args.concurrent if args.concurrent > 0 else 1,
            text_lengths=text_lengths,
            timeout=args.timeout,
        )

        # Sequential test
        print(f"\n--- Sequential Test ({args.num_requests} requests) ---")
        result = await benchmark_sequential(config, gpu_collector)
        all_results.append(result)
        analysis = analyze_results(result)
        all_analyses.append(analysis)
        print_results(analysis)

        # Concurrent test
        if args.concurrent > 0:
            print(f"\n--- Concurrent Test (c={args.concurrent}, {args.num_requests} requests) ---")
            result = await benchmark_concurrent(config, gpu_collector)
            all_results.append(result)
            analysis = analyze_results(result)
            all_analyses.append(analysis)
            print_results(analysis)

        # Sustained load test
        if args.duration > 0:
            print(f"\n--- Sustained Load Test ({args.duration}s) ---")
            result = await benchmark_sustained(config, args.duration, gpu_collector)
            all_results.append(result)
            analysis = analyze_results(result)
            all_analyses.append(analysis)
            print_results(analysis)

        # Ramp-up test
        if args.ramp_up:
            print(f"\n--- Ramp-Up Test ---")
            levels = [int(l.strip()) for l in args.ramp_levels.split(",")]
            ramp_results = await benchmark_rampup(config, levels, args.num_requests // len(levels), gpu_collector)
            for result in ramp_results:
                all_results.append(result)
                analysis = analyze_results(result)
                all_analyses.append(analysis)
                print_results(analysis)

        # Burst test
        if args.burst:
            print(f"\n--- Burst Test ({args.burst_size} simultaneous) ---")
            result = await benchmark_burst(config, args.burst_size, gpu_collector)
            all_results.append(result)
            analysis = analyze_results(result)
            all_analyses.append(analysis)
            print_results(analysis)

        # Stop GPU monitoring
        if gpu_collector:
            gpu_collector.stop()
            summary = gpu_collector.get_summary()
            if summary:
                print()
                print("=" * 70)
                print("  GPU Summary")
                print("=" * 70)
                vram = summary.get("vram", {})
                util = summary.get("utilization", {})
                print(f"  VRAM: {vram.get('avg_mb', 0):.0f} MB avg, {vram.get('max_mb', 0):.0f} MB peak")
                print(f"  Utilization: {util.get('avg', 0):.1f}% avg, {util.get('max', 0):.1f}% peak")
                print()

        # Worker distribution
        final_stats = await get_worker_stats(args.endpoint)
        if final_stats and final_stats.get("workers"):
            print()
            print("=" * 70)
            print("  Worker Distribution")
            print("=" * 70)
            total = sum(w.get("total_requests", 0) for w in final_stats["workers"])
            for i, w in enumerate(final_stats["workers"]):
                reqs = w.get("total_requests", 0)
                pct = reqs / total * 100 if total > 0 else 0
                print(f"  Worker {i} ({w.get('url', 'unknown')}): {reqs} requests ({pct:.1f}%)")
            print()

    # Save JSON results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": all_analyses,
            "comparison": comparison_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    # Generate HTML report
    if args.output_html:
        generate_html_report(all_analyses, args.output_html, comparison_results)

    print()
    print("Benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())
