"""
Worker Scaling Benchmark for Fish Speech

Tests throughput scaling from 1 to N workers to find optimal configuration.

Usage:
    python tools/benchmark_worker_scaling.py --max-workers 4 --requests-per-test 15

Output:
    - Console summary with scaling efficiency
    - JSON results file
    - HTML report with charts
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class WorkerTestResult:
    """Results for a single worker count test."""
    num_workers: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    requests_per_minute: float
    mean_latency: float
    p95_latency: float
    vram_peak_mb: float
    gpu_utilization_avg: float
    worker_distribution: dict


@dataclass
class ScalingReport:
    """Complete scaling benchmark report."""
    timestamp: str
    gpu_name: str
    base_port: int
    max_workers_tested: int
    requests_per_test: int
    concurrency: int
    results: list[WorkerTestResult]

    @property
    def best_throughput(self) -> WorkerTestResult:
        return max(self.results, key=lambda r: r.requests_per_minute)

    @property
    def scaling_efficiency(self) -> list[float]:
        """Calculate scaling efficiency (actual speedup / ideal speedup)."""
        if not self.results:
            return []
        baseline = self.results[0].requests_per_minute
        return [
            (r.requests_per_minute / baseline) / r.num_workers * 100
            for r in self.results
        ]


def get_gpu_info() -> tuple[str, float, float]:
    """Get GPU name, VRAM used, and total VRAM."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(",")
        return parts[0].strip(), float(parts[1]), float(parts[2])
    except Exception:
        return "Unknown", 0, 0


def get_gpu_metrics() -> tuple[float, float, float]:
    """Get current VRAM, utilization, power."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(",")
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0, 0, 0


async def check_worker_health(port: int, timeout: float = 5.0) -> bool:
    """Check if a worker is healthy."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://localhost:{port}/v1/health")
            return response.status_code == 200
    except Exception:
        return False


async def wait_for_workers(ports: list[int], timeout: float = 120.0) -> list[int]:
    """Wait for workers to become healthy, return list of healthy ports."""
    start = time.time()
    healthy_ports = []

    while time.time() - start < timeout:
        healthy_ports = []
        for port in ports:
            if await check_worker_health(port):
                healthy_ports.append(port)

        if len(healthy_ports) == len(ports):
            return healthy_ports

        print(f"  Waiting for workers... {len(healthy_ports)}/{len(ports)} healthy")
        await asyncio.sleep(5)

    return healthy_ports


def start_worker(port: int, args) -> subprocess.Popen:
    """Start a single API server worker."""
    cmd = [
        sys.executable,
        str(project_root / "tools" / "api_server.py"),
        "--listen", f"0.0.0.0:{port}",
    ]

    if args.compile:
        cmd.append("--compile")

    if args.half:
        cmd.append("--half")

    env = os.environ.copy()
    env["WORKER_ID"] = str(port)

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def start_load_balancer(balancer_port: int, worker_ports: list[int]) -> subprocess.Popen:
    """Start the load balancer."""
    cmd = [
        sys.executable,
        str(project_root / "tools" / "load_balancer.py"),
        "--port", str(balancer_port),
        "--workers", ",".join(str(p) for p in worker_ports),
        "--strategy", "least-connections",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def stop_processes(processes: list[subprocess.Popen]):
    """Stop all processes."""
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


async def run_benchmark(
    endpoint: str,
    num_requests: int,
    concurrency: int,
    timeout: float = 120.0,
) -> dict:
    """Run benchmark and return results."""
    from tools.benchmark_multi_worker import (
        BenchmarkConfig, benchmark_concurrent, analyze_results,
        get_worker_stats, GPUMetricsCollector
    )

    config = BenchmarkConfig(
        endpoint=endpoint,
        num_requests=num_requests,
        concurrency=concurrency,
        text_lengths=["medium"],
        timeout=timeout,
    )

    # Start GPU monitoring
    gpu_collector = GPUMetricsCollector(interval=0.5)
    gpu_collector.start()

    # Run benchmark
    results = await benchmark_concurrent(config, gpu_collector)

    # Stop monitoring
    gpu_collector.stop()

    # Get analysis
    analysis = analyze_results(results)

    # Get worker distribution
    worker_stats = await get_worker_stats(endpoint)

    return {
        "analysis": analysis,
        "worker_stats": worker_stats,
        "gpu_summary": gpu_collector.get_summary(),
    }


async def test_worker_count(
    num_workers: int,
    base_port: int,
    balancer_port: int,
    num_requests: int,
    concurrency: int,
    args,
    reuse_workers: list[subprocess.Popen] = None,
) -> tuple[WorkerTestResult, list[subprocess.Popen]]:
    """Test performance with a specific number of workers."""

    print(f"\n{'='*70}")
    print(f"  Testing with {num_workers} worker(s)")
    print(f"{'='*70}")

    processes = reuse_workers or []
    worker_ports = [base_port + i for i in range(num_workers)]

    # Start new workers if needed
    workers_to_start = num_workers - len(processes)
    if workers_to_start > 0:
        print(f"  Starting {workers_to_start} new worker(s)...")
        for i in range(len(processes), num_workers):
            port = base_port + i
            print(f"    Worker {i} on port {port}")
            proc = start_worker(port, args)
            processes.append(proc)

        # Wait for new workers
        print(f"  Waiting for workers to initialize...")
        await asyncio.sleep(args.worker_init_time)

    # Check worker health
    healthy_ports = await wait_for_workers(worker_ports, timeout=60)
    if len(healthy_ports) < num_workers:
        print(f"  WARNING: Only {len(healthy_ports)}/{num_workers} workers healthy")
        worker_ports = healthy_ports

    if not worker_ports:
        print("  ERROR: No healthy workers!")
        return None, processes

    # Start/restart load balancer
    print(f"  Starting load balancer on port {balancer_port}...")
    lb_process = start_load_balancer(balancer_port, worker_ports)
    await asyncio.sleep(3)

    # Run warmup
    print(f"  Running warmup...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(2):
                await client.post(
                    f"http://localhost:{balancer_port}/v1/tts",
                    json={"text": "Warmup request.", "format": "wav"}
                )
    except Exception as e:
        print(f"  Warmup error: {e}")

    # Run benchmark
    print(f"  Running benchmark: {num_requests} requests, concurrency={concurrency}")
    benchmark_result = await run_benchmark(
        f"http://localhost:{balancer_port}",
        num_requests,
        concurrency,
    )

    # Stop load balancer
    lb_process.terminate()
    try:
        lb_process.wait(timeout=5)
    except Exception:
        lb_process.kill()

    # Extract results
    analysis = benchmark_result["analysis"]
    worker_stats = benchmark_result.get("worker_stats", {})
    gpu_summary = benchmark_result.get("gpu_summary", {})

    # Build worker distribution
    distribution = {}
    if worker_stats and "workers" in worker_stats:
        for w in worker_stats["workers"]:
            distribution[w["url"]] = w.get("total_requests", 0)

    result = WorkerTestResult(
        num_workers=num_workers,
        total_requests=analysis.get("total_requests", 0),
        successful_requests=analysis.get("successful", 0),
        failed_requests=analysis.get("failed", 0),
        total_time=analysis.get("total_duration", 0),
        requests_per_second=analysis.get("throughput", {}).get("requests_per_second", 0),
        requests_per_minute=analysis.get("throughput", {}).get("requests_per_minute", 0),
        mean_latency=analysis.get("latency", {}).get("mean", 0),
        p95_latency=analysis.get("latency", {}).get("p95", 0),
        vram_peak_mb=gpu_summary.get("vram", {}).get("max_mb", 0),
        gpu_utilization_avg=gpu_summary.get("utilization", {}).get("avg", 0),
        worker_distribution=distribution,
    )

    # Print results
    print(f"\n  Results:")
    print(f"    Throughput:    {result.requests_per_minute:.1f} req/min")
    print(f"    Mean Latency:  {result.mean_latency:.2f}s")
    print(f"    P95 Latency:   {result.p95_latency:.2f}s")
    print(f"    VRAM Peak:     {result.vram_peak_mb:.0f} MB")
    print(f"    GPU Util:      {result.gpu_utilization_avg:.1f}%")

    if distribution:
        print(f"    Distribution:")
        for url, count in distribution.items():
            print(f"      {url}: {count} requests")

    return result, processes


def generate_scaling_report_html(report: ScalingReport, output_path: str):
    """Generate HTML report for scaling benchmark."""

    worker_counts = [r.num_workers for r in report.results]
    throughputs = [r.requests_per_minute for r in report.results]
    latencies = [r.mean_latency for r in report.results]
    vram_peaks = [r.vram_peak_mb for r in report.results]
    efficiencies = report.scaling_efficiency

    # Calculate ideal linear scaling
    if report.results:
        baseline = report.results[0].requests_per_minute
        ideal_throughputs = [baseline * n for n in worker_counts]
    else:
        ideal_throughputs = []

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Fish Speech Worker Scaling Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .stat {{ padding: 15px; background: #ecf0f1; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ color: #666; }}
        .chart-container {{ height: 300px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .best {{ background: #d5f5e3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fish Speech Worker Scaling Report</h1>
        <p style="text-align:center;color:#666;">Generated: {report.timestamp} | GPU: {report.gpu_name}</p>

        <div class="card">
            <h2>Summary</h2>
            <div class="grid">
                <div class="stat">
                    <div class="stat-value">{report.best_throughput.num_workers}</div>
                    <div class="stat-label">Optimal Workers</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{report.best_throughput.requests_per_minute:.1f}</div>
                    <div class="stat-label">Best Throughput (req/min)</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{report.best_throughput.requests_per_minute / report.results[0].requests_per_minute:.2f}x</div>
                    <div class="stat-label">Speedup vs Single</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{report.best_throughput.vram_peak_mb:.0f} MB</div>
                    <div class="stat-label">Peak VRAM</div>
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
                    <th>Throughput (req/min)</th>
                    <th>Speedup</th>
                    <th>Efficiency</th>
                    <th>Mean Latency</th>
                    <th>P95 Latency</th>
                    <th>VRAM Peak</th>
                </tr>
"""

    baseline_thr = report.results[0].requests_per_minute if report.results else 1
    best_workers = report.best_throughput.num_workers

    for i, r in enumerate(report.results):
        speedup = r.requests_per_minute / baseline_thr
        efficiency = efficiencies[i] if i < len(efficiencies) else 0
        row_class = "best" if r.num_workers == best_workers else ""

        html += f"""
                <tr class="{row_class}">
                    <td>{r.num_workers}</td>
                    <td>{r.requests_per_minute:.1f}</td>
                    <td>{speedup:.2f}x</td>
                    <td>{efficiency:.1f}%</td>
                    <td>{r.mean_latency:.2f}s</td>
                    <td>{r.p95_latency:.2f}s</td>
                    <td>{r.vram_peak_mb:.0f} MB</td>
                </tr>
"""

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
                        borderColor: 'rgb(52, 152, 219)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: true,
                        tension: 0.1
                    }},
                    {{
                        label: 'Ideal Linear Scaling',
                        data: {json.dumps(ideal_throughputs)},
                        borderColor: 'rgba(149, 165, 166, 0.8)',
                        borderDash: [5, 5],
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Number of Workers' }} }},
                    y: {{ title: {{ display: true, text: 'Requests per Minute' }}, beginAtZero: true }}
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
                    backgroundColor: {json.dumps(['rgba(46, 204, 113, 0.8)' if e > 70 else 'rgba(241, 196, 15, 0.8)' if e > 50 else 'rgba(231, 76, 60, 0.8)' for e in efficiencies])}
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Number of Workers' }} }},
                    y: {{ title: {{ display: true, text: 'Efficiency (%)' }}, beginAtZero: true, max: 100 }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nHTML report saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark worker scaling for Fish Speech")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of workers to test")
    parser.add_argument("--base-port", type=int, default=8010, help="Base port for workers")
    parser.add_argument("--balancer-port", type=int, default=8000, help="Load balancer port")
    parser.add_argument("--requests-per-test", type=int, default=15, help="Requests per test")
    parser.add_argument("--concurrency", type=int, default=4, help="Request concurrency")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--worker-init-time", type=int, default=45, help="Seconds to wait for worker init")
    parser.add_argument("--output", type=str, default="scaling_results.json", help="JSON output file")
    parser.add_argument("--output-html", type=str, default="scaling_report.html", help="HTML output file")
    parser.add_argument("--reuse-workers", action="store_true", help="Keep workers running between tests")

    args = parser.parse_args()

    if args.no_compile:
        args.compile = False

    print("="*70)
    print("  Fish Speech Worker Scaling Benchmark")
    print("="*70)

    gpu_name, vram_used, vram_total = get_gpu_info()
    print(f"\nGPU: {gpu_name}")
    print(f"VRAM: {vram_used:.0f} / {vram_total:.0f} MB")
    print(f"\nTest Configuration:")
    print(f"  Max Workers:      {args.max_workers}")
    print(f"  Requests/Test:    {args.requests_per_test}")
    print(f"  Concurrency:      {args.concurrency}")
    print(f"  Compile:          {args.compile}")

    results = []
    processes = []

    try:
        # Test each worker count
        for num_workers in range(1, args.max_workers + 1):
            result, processes = await test_worker_count(
                num_workers=num_workers,
                base_port=args.base_port,
                balancer_port=args.balancer_port,
                num_requests=args.requests_per_test,
                concurrency=args.concurrency,
                args=args,
                reuse_workers=processes if args.reuse_workers else None,
            )

            if result:
                results.append(result)

            # Stop workers if not reusing
            if not args.reuse_workers:
                stop_processes(processes)
                processes = []
                await asyncio.sleep(5)

    finally:
        # Cleanup
        print("\nCleaning up...")
        stop_processes(processes)

    if not results:
        print("No results collected!")
        return

    # Create report
    report = ScalingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gpu_name=gpu_name,
        base_port=args.base_port,
        max_workers_tested=args.max_workers,
        requests_per_test=args.requests_per_test,
        concurrency=args.concurrency,
        results=results,
    )

    # Print summary
    print("\n" + "="*70)
    print("  SCALING SUMMARY")
    print("="*70)

    baseline = results[0].requests_per_minute
    print(f"\n  {'Workers':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<12} {'VRAM Peak'}")
    print(f"  {'-'*10} {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

    for i, r in enumerate(results):
        speedup = r.requests_per_minute / baseline
        efficiency = report.scaling_efficiency[i]
        marker = " <-- BEST" if r.num_workers == report.best_throughput.num_workers else ""
        print(f"  {r.num_workers:<10} {r.requests_per_minute:<15.1f} {speedup:<10.2f}x {efficiency:<12.1f}% {r.vram_peak_mb:.0f} MB{marker}")

    print(f"\n  Optimal configuration: {report.best_throughput.num_workers} workers")
    print(f"  Best throughput: {report.best_throughput.requests_per_minute:.1f} req/min")
    print(f"  Speedup over single: {report.best_throughput.requests_per_minute / baseline:.2f}x")

    # Save JSON
    json_data = {
        "timestamp": report.timestamp,
        "gpu_name": report.gpu_name,
        "config": {
            "max_workers": args.max_workers,
            "requests_per_test": args.requests_per_test,
            "concurrency": args.concurrency,
        },
        "results": [
            {
                "num_workers": r.num_workers,
                "requests_per_minute": r.requests_per_minute,
                "mean_latency": r.mean_latency,
                "p95_latency": r.p95_latency,
                "vram_peak_mb": r.vram_peak_mb,
                "gpu_utilization_avg": r.gpu_utilization_avg,
                "speedup": r.requests_per_minute / baseline,
                "efficiency": report.scaling_efficiency[i],
            }
            for i, r in enumerate(results)
        ],
        "optimal_workers": report.best_throughput.num_workers,
        "best_throughput": report.best_throughput.requests_per_minute,
    }

    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON results saved to: {args.output}")

    # Generate HTML
    generate_scaling_report_html(report, args.output_html)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
