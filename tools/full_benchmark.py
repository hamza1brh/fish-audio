#!/usr/bin/env python3
"""
Fish Speech Full Benchmark Suite

ONE script that runs EVERYTHING:
- Detects GPU and available VRAM
- Tests single worker baseline
- Finds optimal worker count
- Compares single vs multi-worker
- Tests sustained load
- Calculates max throughput
- Generates comprehensive report

Usage:
    python tools/full_benchmark.py

That's it. No arguments needed. It figures everything out.
"""

import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Constants
BASE_PORT = 8010
BALANCER_PORT = 8000
TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "Voice synthesis technology has advanced significantly, enabling natural sounding speech generation.",
]


# ==============================================================================
# Utility Functions
# ==============================================================================

def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_section(text: str):
    print(f"\n--- {text} ---\n")


def run_cmd(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def kill_all_servers():
    """Kill all running Fish Speech servers."""
    print("  Stopping any running servers...")
    run_cmd("pkill -9 -f 'api_server.py'")
    run_cmd("pkill -9 -f 'load_balancer.py'")
    time.sleep(2)


def clear_gpu_memory():
    """Clear GPU memory."""
    print("  Clearing GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass
    time.sleep(1)


def get_gpu_info() -> dict:
    """Get GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "vram_total_mb": float(parts[1]),
            "vram_free_mb": float(parts[2]),
            "vram_used_mb": float(parts[3]),
            "utilization": float(parts[4]) if parts[4] != "[N/A]" else 0,
            "temperature": float(parts[5]) if parts[5] != "[N/A]" else 0,
            "power_draw": float(parts[6]) if parts[6] != "[N/A]" else 0,
        }
    except Exception as e:
        return {"name": "Unknown", "vram_total_mb": 0, "error": str(e)}


def get_current_vram() -> float:
    """Get current VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def estimate_max_workers(vram_total_mb: float, vram_per_worker_mb: float = 4000) -> int:
    """Estimate maximum workers based on VRAM."""
    # Reserve 2GB for system
    available = vram_total_mb - 2000
    max_workers = max(1, int(available / vram_per_worker_mb))
    return min(max_workers, 8)  # Cap at 8 workers


# ==============================================================================
# Server Management
# ==============================================================================

class ServerManager:
    """Manages Fish Speech server processes."""

    def __init__(self):
        self.workers: list[subprocess.Popen] = []
        self.balancer: Optional[subprocess.Popen] = None
        self.worker_ports: list[int] = []

    def start_worker(self, port: int, compile: bool = True) -> bool:
        """Start a single worker."""
        cmd = [
            sys.executable,
            str(project_root / "tools" / "api_server.py"),
            "--listen", f"0.0.0.0:{port}",
        ]
        if compile:
            cmd.append("--compile")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.workers.append(proc)
            self.worker_ports.append(port)
            return True
        except Exception as e:
            print(f"    Failed to start worker on {port}: {e}")
            return False

    def start_balancer(self, port: int = BALANCER_PORT) -> bool:
        """Start the load balancer."""
        if not self.worker_ports:
            return False

        cmd = [
            sys.executable,
            str(project_root / "tools" / "load_balancer.py"),
            "--port", str(port),
            "--workers", ",".join(str(p) for p in self.worker_ports),
            "--strategy", "least-connections",
        ]

        try:
            self.balancer = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return True
        except Exception as e:
            print(f"    Failed to start balancer: {e}")
            return False

    async def wait_for_healthy(self, ports: list[int], timeout: float = 120) -> list[int]:
        """Wait for workers to be healthy."""
        import httpx

        start = time.time()
        healthy = []

        while time.time() - start < timeout:
            healthy = []
            for port in ports:
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"http://localhost:{port}/v1/health")
                        if resp.status_code == 200:
                            healthy.append(port)
                except Exception:
                    pass

            if len(healthy) == len(ports):
                return healthy

            await asyncio.sleep(3)

        return healthy

    def stop_all(self):
        """Stop all servers."""
        if self.balancer:
            try:
                self.balancer.terminate()
                self.balancer.wait(timeout=5)
            except Exception:
                try:
                    self.balancer.kill()
                except Exception:
                    pass
            self.balancer = None

        for proc in self.workers:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self.workers = []
        self.worker_ports = []

    def stop_balancer(self):
        """Stop just the balancer."""
        if self.balancer:
            try:
                self.balancer.terminate()
                self.balancer.wait(timeout=5)
            except Exception:
                try:
                    self.balancer.kill()
                except Exception:
                    pass
            self.balancer = None


# ==============================================================================
# Benchmarking Functions
# ==============================================================================

@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    num_workers: int
    total_requests: int
    successful: int
    failed: int
    duration_seconds: float
    requests_per_second: float
    requests_per_minute: float
    mean_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    vram_peak_mb: float
    gpu_util_avg: float
    audio_minutes_generated: float = 0
    rtf_mean: float = 0  # Real-time factor


async def run_benchmark(
    endpoint: str,
    num_requests: int,
    concurrency: int,
    name: str = "Benchmark",
    collect_gpu: bool = True,
) -> BenchmarkResult:
    """Run a benchmark against an endpoint."""
    import httpx
    import statistics
    import io
    import wave

    print(f"    Running: {num_requests} requests, concurrency={concurrency}")

    results = []
    gpu_samples = []
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def make_request(i: int, client: httpx.AsyncClient) -> dict:
        nonlocal completed
        async with semaphore:
            start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{endpoint}/v1/tts",
                    json={
                        "text": TEST_TEXTS[i % len(TEST_TEXTS)],
                        "format": "wav",
                    },
                    timeout=120.0,
                )
                latency = time.perf_counter() - start

                audio_duration = 0
                if resp.status_code == 200:
                    try:
                        audio_io = io.BytesIO(resp.content)
                        with wave.open(audio_io, "rb") as wav:
                            audio_duration = wav.getnframes() / wav.getframerate()
                    except Exception:
                        pass

                completed += 1
                return {
                    "success": resp.status_code == 200,
                    "latency": latency,
                    "audio_duration": audio_duration,
                }
            except Exception as e:
                completed += 1
                return {"success": False, "latency": time.perf_counter() - start, "error": str(e)}

    # GPU monitoring task
    async def monitor_gpu():
        while True:
            gpu_samples.append(get_current_vram())
            await asyncio.sleep(0.5)

    # Start monitoring
    monitor_task = None
    if collect_gpu:
        monitor_task = asyncio.create_task(monitor_gpu())

    # Run benchmark
    start_time = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [make_request(i, client) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    # Stop monitoring
    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    latencies = [r["latency"] for r in successful] if successful else [0]
    audio_durations = [r.get("audio_duration", 0) for r in successful]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return BenchmarkResult(
        name=name,
        num_workers=1,  # Set by caller
        total_requests=num_requests,
        successful=len(successful),
        failed=len(failed),
        duration_seconds=total_time,
        requests_per_second=len(successful) / total_time if total_time > 0 else 0,
        requests_per_minute=len(successful) / total_time * 60 if total_time > 0 else 0,
        mean_latency=statistics.mean(latencies),
        p50_latency=sorted_latencies[n // 2] if n > 0 else 0,
        p95_latency=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1] if n > 0 else 0,
        p99_latency=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1] if n > 0 else 0,
        vram_peak_mb=max(gpu_samples) if gpu_samples else 0,
        gpu_util_avg=0,  # Would need nvidia-smi polling
        audio_minutes_generated=sum(audio_durations) / 60,
        rtf_mean=statistics.mean([r["latency"] / r["audio_duration"] for r in successful if r.get("audio_duration", 0) > 0]) if any(r.get("audio_duration", 0) > 0 for r in successful) else 0,
    )


async def warmup(endpoint: str, count: int = 3):
    """Run warmup requests."""
    import httpx
    print(f"    Warming up with {count} requests...")
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(count):
            try:
                await client.post(
                    f"{endpoint}/v1/tts",
                    json={"text": "Warmup.", "format": "wav"},
                )
            except Exception:
                pass


# ==============================================================================
# Main Benchmark Suite
# ==============================================================================

@dataclass
class FullBenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    gpu: dict
    single_worker: Optional[BenchmarkResult] = None
    multi_worker_results: list[BenchmarkResult] = field(default_factory=list)
    optimal_workers: int = 1
    max_throughput: float = 0
    max_audio_per_hour: float = 0
    speedup: float = 1.0
    scaling_efficiency: list[float] = field(default_factory=list)


async def run_full_benchmark():
    """Run the complete benchmark suite."""

    print_header("Fish Speech Full Benchmark Suite")
    print("This will test everything and find optimal configuration.")
    print("Sit back and relax - this may take 10-20 minutes.\n")

    report = FullBenchmarkReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gpu={},
    )

    # ==== Phase 1: System Check ====
    print_section("Phase 1: System Check")

    kill_all_servers()
    clear_gpu_memory()

    gpu_info = get_gpu_info()
    report.gpu = gpu_info

    print(f"  GPU: {gpu_info.get('name', 'Unknown')}")
    print(f"  VRAM Total: {gpu_info.get('vram_total_mb', 0):.0f} MB")
    print(f"  VRAM Free: {gpu_info.get('vram_free_mb', 0):.0f} MB")

    max_workers_estimate = estimate_max_workers(gpu_info.get("vram_total_mb", 8000))
    print(f"  Estimated max workers: {max_workers_estimate}")

    # ==== Phase 2: Single Worker Baseline ====
    print_section("Phase 2: Single Worker Baseline")

    server = ServerManager()

    print("  Starting single worker...")
    server.start_worker(BASE_PORT, compile=True)

    print("  Waiting for worker to initialize (this takes ~30-60s for first compile)...")
    healthy = await server.wait_for_healthy([BASE_PORT], timeout=180)

    if not healthy:
        print("  ERROR: Worker failed to start!")
        server.stop_all()
        return None

    print("  Worker ready!")

    await warmup(f"http://localhost:{BASE_PORT}")

    print("  Running single worker benchmark...")
    single_result = await run_benchmark(
        f"http://localhost:{BASE_PORT}",
        num_requests=12,
        concurrency=4,
        name="Single Worker",
    )
    single_result.num_workers = 1

    report.single_worker = single_result

    print(f"\n  Single Worker Results:")
    print(f"    Throughput:     {single_result.requests_per_minute:.1f} req/min")
    print(f"    Mean Latency:   {single_result.mean_latency:.2f}s")
    print(f"    P95 Latency:    {single_result.p95_latency:.2f}s")
    print(f"    VRAM Peak:      {single_result.vram_peak_mb:.0f} MB")
    if single_result.rtf_mean > 0:
        print(f"    RTF:            {single_result.rtf_mean:.2f} ({1/single_result.rtf_mean:.1f}x realtime)")

    vram_per_worker = single_result.vram_peak_mb
    max_workers = min(max_workers_estimate, estimate_max_workers(gpu_info.get("vram_total_mb", 8000), vram_per_worker))
    print(f"\n  VRAM per worker: ~{vram_per_worker:.0f} MB")
    print(f"  Adjusted max workers to test: {max_workers}")

    # ==== Phase 3: Worker Scaling Test ====
    print_section("Phase 3: Worker Scaling Test")

    server.stop_all()
    clear_gpu_memory()
    time.sleep(3)

    all_results = [single_result]

    for num_workers in range(2, max_workers + 1):
        print(f"\n  Testing {num_workers} workers...")

        # Start workers
        for i in range(num_workers):
            port = BASE_PORT + i
            print(f"    Starting worker {i+1} on port {port}...")
            server.start_worker(port, compile=True)

            # Stagger starts to share compile cache
            if i < num_workers - 1:
                await asyncio.sleep(10)

        print(f"    Waiting for all workers...")
        healthy = await server.wait_for_healthy(server.worker_ports, timeout=180)

        if len(healthy) < num_workers:
            print(f"    WARNING: Only {len(healthy)}/{num_workers} workers healthy")
            if len(healthy) == 0:
                print(f"    Skipping {num_workers} workers test")
                server.stop_all()
                clear_gpu_memory()
                time.sleep(3)
                continue

        # Start load balancer
        server.start_balancer(BALANCER_PORT)
        await asyncio.sleep(3)

        await warmup(f"http://localhost:{BALANCER_PORT}")

        # Run benchmark
        result = await run_benchmark(
            f"http://localhost:{BALANCER_PORT}",
            num_requests=12,
            concurrency=min(num_workers * 2, 8),
            name=f"{num_workers} Workers",
        )
        result.num_workers = num_workers

        all_results.append(result)
        report.multi_worker_results.append(result)

        print(f"\n    {num_workers} Workers Results:")
        print(f"      Throughput:   {result.requests_per_minute:.1f} req/min")
        print(f"      Speedup:      {result.requests_per_minute / single_result.requests_per_minute:.2f}x")
        print(f"      VRAM Peak:    {result.vram_peak_mb:.0f} MB")

        # Cleanup for next test
        server.stop_all()
        clear_gpu_memory()
        time.sleep(5)

    # ==== Phase 4: Analysis ====
    print_section("Phase 4: Analysis")

    # Find optimal
    best_result = max(all_results, key=lambda r: r.requests_per_minute)
    report.optimal_workers = best_result.num_workers
    report.max_throughput = best_result.requests_per_minute
    report.speedup = best_result.requests_per_minute / single_result.requests_per_minute

    # Calculate scaling efficiency
    baseline = single_result.requests_per_minute
    report.scaling_efficiency = [
        (r.requests_per_minute / baseline) / r.num_workers * 100
        for r in all_results
    ]

    # Calculate max audio per hour
    if best_result.audio_minutes_generated > 0 and best_result.duration_seconds > 0:
        report.max_audio_per_hour = (best_result.audio_minutes_generated / best_result.duration_seconds) * 3600

    # ==== Phase 5: Sustained Load Test ====
    print_section("Phase 5: Sustained Load Test (60 seconds)")

    print(f"  Starting {report.optimal_workers} worker(s) for sustained test...")

    for i in range(report.optimal_workers):
        server.start_worker(BASE_PORT + i, compile=True)
        if i < report.optimal_workers - 1:
            await asyncio.sleep(5)

    await server.wait_for_healthy(server.worker_ports, timeout=180)

    if report.optimal_workers > 1:
        server.start_balancer(BALANCER_PORT)
        await asyncio.sleep(3)
        endpoint = f"http://localhost:{BALANCER_PORT}"
    else:
        endpoint = f"http://localhost:{BASE_PORT}"

    await warmup(endpoint)

    print("  Running 60-second sustained load test...")

    # Run sustained test
    sustained_start = time.time()
    sustained_results = []
    target_duration = 60

    async def sustained_worker():
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            while time.time() - sustained_start < target_duration:
                start = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{endpoint}/v1/tts",
                        json={"text": TEST_TEXTS[len(sustained_results) % len(TEST_TEXTS)], "format": "wav"},
                    )
                    sustained_results.append({
                        "success": resp.status_code == 200,
                        "latency": time.perf_counter() - start,
                    })
                except Exception:
                    sustained_results.append({
                        "success": False,
                        "latency": time.perf_counter() - start,
                    })

    # Run with concurrency
    tasks = [sustained_worker() for _ in range(min(report.optimal_workers * 2, 8))]
    await asyncio.gather(*tasks)

    sustained_duration = time.time() - sustained_start
    sustained_success = len([r for r in sustained_results if r["success"]])

    print(f"\n  Sustained Test Results:")
    print(f"    Duration:       {sustained_duration:.1f}s")
    print(f"    Total Requests: {len(sustained_results)}")
    print(f"    Successful:     {sustained_success}")
    print(f"    Throughput:     {sustained_success / sustained_duration * 60:.1f} req/min")

    server.stop_all()

    # ==== Final Report ====
    print_header("FINAL RESULTS")

    print(f"  GPU: {report.gpu.get('name', 'Unknown')}")
    print(f"  VRAM: {report.gpu.get('vram_total_mb', 0):.0f} MB total")
    print()
    print(f"  OPTIMAL CONFIGURATION:")
    print(f"    Workers:        {report.optimal_workers}")
    print(f"    Max Throughput: {report.max_throughput:.1f} req/min ({report.max_throughput/60:.2f} req/sec)")
    print(f"    Speedup:        {report.speedup:.2f}x over single worker")
    print()

    print(f"  SCALING RESULTS:")
    print(f"    {'Workers':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency'}")
    print(f"    {'-'*10} {'-'*15} {'-'*10} {'-'*12}")
    for i, r in enumerate(all_results):
        speedup = r.requests_per_minute / baseline
        eff = report.scaling_efficiency[i] if i < len(report.scaling_efficiency) else 0
        marker = " <-- OPTIMAL" if r.num_workers == report.optimal_workers else ""
        print(f"    {r.num_workers:<10} {r.requests_per_minute:<15.1f} {speedup:<10.2f}x {eff:.1f}%{marker}")

    # Save JSON report
    json_report = {
        "timestamp": report.timestamp,
        "gpu": report.gpu,
        "optimal_workers": report.optimal_workers,
        "max_throughput_per_minute": report.max_throughput,
        "speedup_over_single": report.speedup,
        "scaling_results": [
            {
                "workers": r.num_workers,
                "throughput_per_minute": r.requests_per_minute,
                "mean_latency": r.mean_latency,
                "p95_latency": r.p95_latency,
                "vram_peak_mb": r.vram_peak_mb,
                "speedup": r.requests_per_minute / baseline,
                "efficiency": report.scaling_efficiency[i] if i < len(report.scaling_efficiency) else 0,
            }
            for i, r in enumerate(all_results)
        ],
        "sustained_test": {
            "duration_seconds": sustained_duration,
            "total_requests": len(sustained_results),
            "successful": sustained_success,
            "throughput_per_minute": sustained_success / sustained_duration * 60,
        },
    }

    output_file = "full_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\n  Results saved to: {output_file}")

    # Generate HTML report
    generate_html_report(report, all_results, sustained_results, sustained_duration)

    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE!")
    print("="*70 + "\n")

    return report


def generate_html_report(report, all_results, sustained_results, sustained_duration):
    """Generate comprehensive HTML report."""

    baseline = all_results[0].requests_per_minute if all_results else 1

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Fish Speech Full Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; font-size: 2.5em; margin-bottom: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card h2 {{ color: #f8fafc; margin-bottom: 20px; font-size: 1.4em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat {{ background: #334155; border-radius: 8px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; background: linear-gradient(135deg, #22c55e 0%, #10b981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .stat-label {{ color: #94a3b8; margin-top: 5px; }}
        .chart-container {{ height: 300px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #f8fafc; font-weight: 600; }}
        tr:hover {{ background: #334155; }}
        .best {{ background: rgba(34, 197, 94, 0.2); }}
        .gpu-info {{ display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 20px; }}
        .gpu-stat {{ text-align: center; }}
        .gpu-stat-value {{ font-size: 1.2em; font-weight: bold; color: #f8fafc; }}
        .gpu-stat-label {{ color: #64748b; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fish Speech Benchmark Report</h1>
        <p class="subtitle">Generated: {report.timestamp}</p>

        <div class="card">
            <div class="gpu-info">
                <div class="gpu-stat">
                    <div class="gpu-stat-value">{report.gpu.get('name', 'Unknown')}</div>
                    <div class="gpu-stat-label">GPU</div>
                </div>
                <div class="gpu-stat">
                    <div class="gpu-stat-value">{report.gpu.get('vram_total_mb', 0)/1024:.0f} GB</div>
                    <div class="gpu-stat-label">VRAM</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Optimal Configuration</h2>
            <div class="grid">
                <div class="stat">
                    <div class="stat-value">{report.optimal_workers}</div>
                    <div class="stat-label">Optimal Workers</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{report.max_throughput:.0f}</div>
                    <div class="stat-label">Max Req/Min</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{report.speedup:.1f}x</div>
                    <div class="stat-label">Speedup</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len([r for r in sustained_results if r['success']]) / sustained_duration * 60:.0f}</div>
                    <div class="stat-label">Sustained Req/Min</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Throughput Scaling</h2>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Scaling Efficiency</h2>
            <div class="chart-container">
                <canvas id="efficiencyChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Workers</th>
                    <th>Throughput</th>
                    <th>Speedup</th>
                    <th>Efficiency</th>
                    <th>Mean Latency</th>
                    <th>P95 Latency</th>
                    <th>VRAM Peak</th>
                </tr>
"""

    for i, r in enumerate(all_results):
        speedup = r.requests_per_minute / baseline
        eff = report.scaling_efficiency[i] if i < len(report.scaling_efficiency) else 0
        row_class = "best" if r.num_workers == report.optimal_workers else ""
        html += f"""
                <tr class="{row_class}">
                    <td>{r.num_workers}</td>
                    <td>{r.requests_per_minute:.1f} req/min</td>
                    <td>{speedup:.2f}x</td>
                    <td>{eff:.1f}%</td>
                    <td>{r.mean_latency:.2f}s</td>
                    <td>{r.p95_latency:.2f}s</td>
                    <td>{r.vram_peak_mb:.0f} MB</td>
                </tr>
"""

    worker_counts = [r.num_workers for r in all_results]
    throughputs = [r.requests_per_minute for r in all_results]
    ideal = [all_results[0].requests_per_minute * n for n in worker_counts]
    efficiencies = report.scaling_efficiency

    html += f"""
            </table>
        </div>
    </div>

    <script>
        new Chart(document.getElementById('throughputChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(worker_counts)},
                datasets: [
                    {{
                        label: 'Actual Throughput',
                        data: {json.dumps(throughputs)},
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Ideal Linear',
                        data: {json.dumps(ideal)},
                        borderColor: '#64748b',
                        borderDash: [5, 5],
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Workers', color: '#e2e8f0' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
                    y: {{ title: {{ display: true, text: 'Requests/Minute', color: '#e2e8f0' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }}, beginAtZero: true }}
                }}
            }}
        }});

        new Chart(document.getElementById('efficiencyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(worker_counts)},
                datasets: [{{
                    label: 'Scaling Efficiency (%)',
                    data: {json.dumps(efficiencies)},
                    backgroundColor: {json.dumps(['#22c55e' if e > 70 else '#eab308' if e > 50 else '#ef4444' for e in efficiencies])}
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Workers', color: '#e2e8f0' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
                    y: {{ title: {{ display: true, text: 'Efficiency (%)', color: '#e2e8f0' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }}, beginAtZero: true, max: 100 }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    output_file = "full_benchmark_report.html"
    with open(output_file, "w") as f:
        f.write(html)
    print(f"  HTML report saved to: {output_file}")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        kill_all_servers()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(run_full_benchmark())
