"""
Comprehensive Fish-Speech Benchmark for RunPod
Tests: BF16 vs INT8, concurrent inference, VRAM usage, throughput
Always uses torch.compile for optimal performance.

Run:
  python tools/runpod/benchmark.py                              # Default benchmark
  python tools/runpod/benchmark.py --num-samples 10             # More samples
  python tools/runpod/benchmark.py --concurrency 4              # Concurrent test
  python tools/runpod/benchmark.py --output report.html         # HTML report
"""

import argparse
import base64
import concurrent.futures
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np

# Import matplotlib with non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def check_int8_available() -> bool:
    """Check if torchao INT8 quantization is compatible with current PyTorch version."""
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        return True
    except (ImportError, AttributeError) as e:
        return False


# Check INT8 availability at module load
INT8_AVAILABLE = check_int8_available()


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for a single benchmark run."""
    config_name: str

    # VRAM metrics
    vram_baseline_mb: float = 0.0
    vram_peak_mb: float = 0.0
    vram_delta_mb: float = 0.0
    llama_vram_mb: float = 0.0
    dac_vram_mb: float = 0.0

    # Timing metrics
    warmup_time_s: float = 0.0
    avg_latency_s: float = 0.0
    std_latency_s: float = 0.0
    min_latency_s: float = 0.0
    max_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0

    # Performance metrics
    avg_rtf: float = 0.0  # Real-time factor (gen_time / audio_duration)
    tokens_per_sec: float = 0.0
    audio_per_minute: float = 0.0  # Minutes of audio per minute of compute
    audio_per_hour: float = 0.0  # Hours of audio per hour of compute
    throughput_x_realtime: float = 0.0  # How many X realtime

    # Test info
    samples_tested: int = 0
    errors: int = 0
    compile_enabled: bool = False

    # Raw data for plotting
    latencies: list = field(default_factory=list)
    audio_durations: list = field(default_factory=list)


@dataclass
class ConcurrentMetrics:
    """Metrics for concurrent benchmark."""
    config_name: str
    concurrency: int
    total_requests: int
    total_time_s: float
    requests_per_sec: float
    avg_latency_s: float
    total_audio_duration_s: float
    throughput_x_realtime: float
    peak_vram_mb: float
    errors: int = 0


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
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
    compile_mode: bool = False,
    num_samples: int = 5,
    warmup_runs: int = 2,
    max_new_tokens: int = 1024,
) -> BenchmarkMetrics:
    """Benchmark a specific configuration with comprehensive metrics."""
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
        compile=compile_mode,
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
        compile=compile_mode,
    )

    # Warmup with timing (important for compile mode)
    print(f"Warming up ({warmup_runs} runs)...")
    warmup_start = time.perf_counter()
    for i in range(warmup_runs):
        req = ServeTTSRequest(text="Warmup test sentence.", max_new_tokens=256, streaming=False)
        for result in engine.inference(req):
            pass
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s" + (" (includes compilation)" if compile_mode else ""))

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

                torch.cuda.synchronize()
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

    # Calculate comprehensive metrics
    metrics = BenchmarkMetrics(
        config_name=config_name,
        vram_baseline_mb=total_baseline,
        vram_peak_mb=peak_vram,
        vram_delta_mb=peak_vram - total_baseline,
        llama_vram_mb=llama_vram,
        dac_vram_mb=dac_vram,
        warmup_time_s=warmup_time,
        compile_enabled=compile_mode,
        samples_tested=len(generation_times),
        errors=errors,
        latencies=generation_times.copy(),
        audio_durations=audio_durations.copy(),
    )

    if generation_times:
        gen_times = np.array(generation_times)
        audio_durs = np.array(audio_durations)

        # Latency stats
        metrics.avg_latency_s = float(np.mean(gen_times))
        metrics.std_latency_s = float(np.std(gen_times))
        metrics.min_latency_s = float(np.min(gen_times))
        metrics.max_latency_s = float(np.max(gen_times))
        metrics.p50_latency_s = float(np.percentile(gen_times, 50))
        metrics.p95_latency_s = float(np.percentile(gen_times, 95))
        metrics.p99_latency_s = float(np.percentile(gen_times, 99))

        # Performance stats
        avg_audio_dur = float(np.mean(audio_durs))
        if avg_audio_dur > 0:
            metrics.avg_rtf = metrics.avg_latency_s / avg_audio_dur
            metrics.throughput_x_realtime = avg_audio_dur / metrics.avg_latency_s

        if metrics.avg_latency_s > 0:
            avg_tokens = np.mean(tokens_generated)
            metrics.tokens_per_sec = float(avg_tokens / metrics.avg_latency_s)

        # Production estimates
        if metrics.avg_rtf > 0:
            metrics.audio_per_minute = 60 / metrics.avg_rtf  # seconds of audio per minute
            metrics.audio_per_hour = metrics.audio_per_minute * 60  # seconds of audio per hour

    # Print results
    print(f"\nResults:")
    print(f"  VRAM: {metrics.vram_baseline_mb:.0f} MB baseline → {metrics.vram_peak_mb:.0f} MB peak")
    print(f"  Latency: {metrics.avg_latency_s:.2f}s ± {metrics.std_latency_s:.2f}s (P95: {metrics.p95_latency_s:.2f}s)")
    print(f"  RTF: {metrics.avg_rtf:.3f} | Throughput: {metrics.throughput_x_realtime:.2f}x realtime")
    print(f"  Tokens/sec: {metrics.tokens_per_sec:.1f}")
    if errors > 0:
        print(f"  Errors: {errors}")

    # Cleanup
    del engine, dac, llama_queue
    clear_vram()

    return metrics


def benchmark_concurrent(
    checkpoint_path: str,
    concurrency: int,
    num_requests: int,
    runtime_int8: bool = True,
    dac_int8: bool = True,
    compile_mode: bool = False,
) -> ConcurrentMetrics:
    """Benchmark concurrent request handling."""
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_dac_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    config_name = f"Concurrent ({concurrency} workers)"
    print(f"\n{'='*60}")
    print(f"Concurrent Benchmark: {config_name}")
    print(f"{'='*60}")

    clear_vram()

    # Load models
    print("Loading models...")
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device="cuda",
        precision=precision,
        compile=compile_mode,
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
        compile=compile_mode,
    )

    # Warmup
    print("Warming up...")
    req = ServeTTSRequest(text="Warmup.", max_new_tokens=256, streaming=False)
    for result in engine.inference(req):
        pass
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Prepare requests
    texts = [TEST_TEXTS[i % len(TEST_TEXTS)] for i in range(num_requests)]
    latencies = []
    audio_durations = []
    errors = 0

    def generate_single(text):
        """Generate audio for a single text."""
        try:
            req = ServeTTSRequest(
                text=text,
                max_new_tokens=1024,
                streaming=False,
            )
            start = time.perf_counter()
            audio_dur = 0
            for result in engine.inference(req):
                if result.code == "final" and result.audio:
                    sample_rate, audio_data = result.audio
                    audio_dur = len(audio_data) / sample_rate
            elapsed = time.perf_counter() - start
            return elapsed, audio_dur, None
        except Exception as e:
            return 0, 0, str(e)

    # Run concurrent test
    print(f"Running {num_requests} requests with {concurrency} concurrent workers...")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_single, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            elapsed, audio_dur, error = future.result()
            if error:
                errors += 1
                print(f"  Error: {error}")
            else:
                latencies.append(elapsed)
                audio_durations.append(audio_dur)

    total_time = time.perf_counter() - start_time
    peak_vram = get_peak_vram_mb()

    # Calculate metrics
    total_audio = sum(audio_durations)
    completed = len(latencies)

    metrics = ConcurrentMetrics(
        config_name=config_name,
        concurrency=concurrency,
        total_requests=num_requests,
        total_time_s=total_time,
        requests_per_sec=completed / total_time if total_time > 0 else 0,
        avg_latency_s=np.mean(latencies) if latencies else 0,
        total_audio_duration_s=total_audio,
        throughput_x_realtime=total_audio / total_time if total_time > 0 else 0,
        peak_vram_mb=peak_vram,
        errors=errors,
    )

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s for {completed} requests")
    print(f"  Requests/sec: {metrics.requests_per_sec:.2f}")
    print(f"  Avg latency: {metrics.avg_latency_s:.2f}s")
    print(f"  Throughput: {metrics.throughput_x_realtime:.2f}x realtime")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")

    # Cleanup
    del engine, dac, llama_queue
    clear_vram()

    return metrics


def generate_plots(results: list[BenchmarkMetrics], concurrent_results: list[ConcurrentMetrics] = None) -> dict:
    """Generate matplotlib plots and return as base64 encoded PNGs."""
    plots = {}

    # Filter out any empty results
    results = [r for r in results if r.samples_tested > 0]
    if not results:
        return plots

    config_names = [r.config_name for r in results]

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # 1. VRAM Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(config_names))
    width = 0.35

    baseline_vram = [r.vram_baseline_mb / 1024 for r in results]  # Convert to GB
    peak_vram = [r.vram_peak_mb / 1024 for r in results]

    bars1 = ax.bar(x - width/2, baseline_vram, width, label='Baseline VRAM', color='steelblue')
    bars2 = ax.bar(x + width/2, peak_vram, width, label='Peak VRAM', color='coral')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('VRAM (GB)')
    ax.set_title('VRAM Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', padding=3)
    ax.bar_label(bars2, fmt='%.1f', padding=3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['vram_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 2. RTF Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    rtf_values = [r.avg_rtf for r in results]
    colors_rtf = ['green' if r < 0.5 else 'orange' if r < 1.0 else 'red' for r in rtf_values]

    bars = ax.bar(config_names, rtf_values, color=colors_rtf)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Realtime (RTF=1.0)')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Real-Time Factor (lower is better)')
    ax.set_title('RTF Comparison')
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    ax.bar_label(bars, fmt='%.3f', padding=3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['rtf_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 3. Compile Speedup Bar Chart (if we have compile vs non-compile pairs)
    compile_results = [r for r in results if r.compile_enabled]
    non_compile_results = [r for r in results if not r.compile_enabled]

    if compile_results and non_compile_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Match configs by base name
        speedups = []
        labels = []
        for nc_result in non_compile_results:
            base_name = nc_result.config_name
            for c_result in compile_results:
                if base_name in c_result.config_name or c_result.config_name.replace(' + compile', '') == base_name:
                    if nc_result.avg_latency_s > 0:
                        speedup = nc_result.avg_latency_s / c_result.avg_latency_s
                        speedups.append(speedup)
                        labels.append(base_name)
                    break

        if speedups:
            colors_speedup = ['green' if s > 1 else 'red' for s in speedups]
            bars = ax.bar(labels, speedups, color=colors_speedup)
            ax.axhline(y=1.0, color='gray', linestyle='--', label='No speedup')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Speedup (higher is better)')
            ax.set_title('Compile Speedup vs Non-Compile')
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.bar_label(bars, fmt='%.2fx', padding=3)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['compile_speedup'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

    # 4. Latency Distribution Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    latency_data = [r.latencies for r in results if r.latencies]

    if latency_data:
        bp = ax.boxplot(latency_data, labels=[r.config_name for r in results if r.latencies], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(latency_data)]):
            patch.set_facecolor(color)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('Latency Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['latency_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    # 5. Warmup Time Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    warmup_times = [r.warmup_time_s for r in results]
    colors_warmup = ['orange' if r.compile_enabled else 'steelblue' for r in results]

    bars = ax.bar(config_names, warmup_times, color=colors_warmup)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Warmup Time (seconds)')
    ax.set_title('Warmup Time (includes compilation overhead for compile configs)')
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.1fs', padding=3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['warmup_time'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 6. Throughput vs Concurrency Line Chart (if concurrent results exist)
    if concurrent_results and len(concurrent_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        concurrencies = [r.concurrency for r in concurrent_results]
        throughputs = [r.throughput_x_realtime for r in concurrent_results]
        requests_per_sec = [r.requests_per_sec for r in concurrent_results]

        ax.plot(concurrencies, throughputs, 'o-', color='steelblue', linewidth=2, markersize=8, label='Throughput (x realtime)')
        ax2 = ax.twinx()
        ax2.plot(concurrencies, requests_per_sec, 's--', color='coral', linewidth=2, markersize=8, label='Requests/sec')

        ax.set_xlabel('Concurrency (workers)')
        ax.set_ylabel('Throughput (x realtime)', color='steelblue')
        ax2.set_ylabel('Requests/sec', color='coral')
        ax.set_title('Throughput vs Concurrency')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['throughput_concurrency'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    # 7. Audio Production Estimates Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    audio_per_hour_mins = [r.audio_per_hour / 60 for r in results]  # Convert to minutes

    bars = ax.bar(config_names, audio_per_hour_mins, color='teal')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Audio Minutes per Hour')
    ax.set_title('Production Capacity: Audio Minutes Generated per Hour')
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.bar_label(bars, fmt='%.0f min', padding=3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plots['audio_production'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return plots


def generate_html_report(
    gpu_info: dict,
    results: list[BenchmarkMetrics],
    concurrent_results: list[ConcurrentMetrics],
    plots: dict,
    output_path: str,
):
    """Generate an HTML report with embedded plots."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build summary table
    summary_rows = ""
    for r in results:
        compile_badge = '<span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">compiled</span>' if r.compile_enabled else ''
        summary_rows += f"""
        <tr>
            <td>{r.config_name} {compile_badge}</td>
            <td>{r.vram_baseline_mb:.0f} MB</td>
            <td>{r.vram_peak_mb:.0f} MB</td>
            <td>{r.avg_rtf:.3f}</td>
            <td>{r.throughput_x_realtime:.2f}x</td>
            <td>{r.avg_latency_s:.2f}s</td>
            <td>{r.p95_latency_s:.2f}s</td>
            <td>{r.tokens_per_sec:.1f}</td>
            <td>{r.audio_per_hour/60:.0f} min</td>
        </tr>
        """

    # Build concurrent results table
    concurrent_rows = ""
    if concurrent_results:
        for r in concurrent_results:
            concurrent_rows += f"""
            <tr>
                <td>{r.concurrency}</td>
                <td>{r.total_requests}</td>
                <td>{r.total_time_s:.2f}s</td>
                <td>{r.requests_per_sec:.2f}</td>
                <td>{r.avg_latency_s:.2f}s</td>
                <td>{r.throughput_x_realtime:.2f}x</td>
                <td>{r.peak_vram_mb:.0f} MB</td>
            </tr>
            """

    # Build plots section
    plots_html = ""
    plot_titles = {
        'vram_comparison': 'VRAM Usage Comparison',
        'rtf_comparison': 'Real-Time Factor Comparison',
        'compile_speedup': 'Compile Speedup',
        'latency_distribution': 'Latency Distribution',
        'warmup_time': 'Warmup Time',
        'throughput_concurrency': 'Throughput vs Concurrency',
        'audio_production': 'Production Capacity',
    }

    for plot_name, plot_data in plots.items():
        title = plot_titles.get(plot_name, plot_name)
        plots_html += f"""
        <div class="chart-container">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{plot_data}" alt="{title}">
        </div>
        """

    # Find best configs
    if results:
        best_throughput = max(results, key=lambda x: x.throughput_x_realtime)
        best_vram = min(results, key=lambda x: x.vram_baseline_mb)
        best_latency = min(results, key=lambda x: x.avg_latency_s)
    else:
        best_throughput = best_vram = best_latency = None

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fish-Speech Benchmark Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ border: none; color: white; margin: 0; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .gpu-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .gpu-info-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .gpu-info-item .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .gpu-info-item .value {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .recommendations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .recommendation h4 {{ margin: 0 0 10px 0; opacity: 0.9; }}
        .recommendation .config {{ font-size: 18px; font-weight: bold; }}
        .recommendation .metric {{ font-size: 24px; font-weight: bold; margin-top: 10px; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 Fish-Speech Benchmark Report</h1>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="card">
        <h2>GPU Information</h2>
        <div class="gpu-info">
            <div class="gpu-info-item">
                <div class="label">GPU</div>
                <div class="value">{gpu_info.get('name', 'N/A')}</div>
            </div>
            <div class="gpu-info-item">
                <div class="label">VRAM</div>
                <div class="value">{gpu_info.get('total_memory_gb', 0):.1f} GB</div>
            </div>
            <div class="gpu-info-item">
                <div class="label">CUDA</div>
                <div class="value">{gpu_info.get('cuda_version', 'N/A')}</div>
            </div>
            <div class="gpu-info-item">
                <div class="label">PyTorch</div>
                <div class="value">{gpu_info.get('pytorch_version', 'N/A')}</div>
            </div>
            <div class="gpu-info-item">
                <div class="label">Compute Capability</div>
                <div class="value">{gpu_info.get('compute_capability', 'N/A')}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Summary Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Configuration</th>
                    <th>Baseline VRAM</th>
                    <th>Peak VRAM</th>
                    <th>RTF</th>
                    <th>Throughput</th>
                    <th>Avg Latency</th>
                    <th>P95 Latency</th>
                    <th>Tokens/sec</th>
                    <th>Audio/Hour</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
    </div>

    {"<div class='card'><h2>Concurrent Performance</h2><table><thead><tr><th>Concurrency</th><th>Requests</th><th>Total Time</th><th>Req/sec</th><th>Avg Latency</th><th>Throughput</th><th>Peak VRAM</th></tr></thead><tbody>" + concurrent_rows + "</tbody></table></div>" if concurrent_rows else ""}

    <h2>Performance Charts</h2>
    {plots_html}

    {"<div class='card'><h2>Recommendations</h2><div class='recommendations'>" + (f"<div class='recommendation'><h4>Best Throughput</h4><div class='config'>{best_throughput.config_name}</div><div class='metric'>{best_throughput.throughput_x_realtime:.2f}x realtime</div></div>" if best_throughput else "") + (f"<div class='recommendation'><h4>Lowest VRAM</h4><div class='config'>{best_vram.config_name}</div><div class='metric'>{best_vram.vram_baseline_mb:.0f} MB</div></div>" if best_vram else "") + (f"<div class='recommendation'><h4>Lowest Latency</h4><div class='config'>{best_latency.config_name}</div><div class='metric'>{best_latency.avg_latency_s:.2f}s avg</div></div>" if best_latency else "") + "</div></div>" if results else ""}

    <div class="footer">
        <p>Fish-Speech Benchmark Suite | <a href="https://github.com/fishaudio/fish-speech">GitHub</a></p>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML report saved to: {output_path}")


def run_benchmark(args):
    """Run the benchmark suite with the given arguments."""
    print("=" * 70)
    print("Fish-Speech Benchmark Suite")
    print("=" * 70)

    # GPU Info
    gpu_info = get_gpu_info()
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print(f"  INT8 quantization: {'Available' if INT8_AVAILABLE else 'Unavailable (torchao/PyTorch mismatch)'}")
    print(f"  torch.compile: Enabled (required for optimal performance)")
    print("\n  NOTE: First run includes compilation overhead (~30-60s warmup)")

    checkpoint_path = "checkpoints/openaudio-s1-mini"

    if not Path(checkpoint_path).exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini")
        return

    # Build test configurations - always use torch.compile for optimal performance
    # Non-compiled mode is not supported as it's significantly slower
    configs = [
        {"name": "BF16 (compiled)", "int8": False, "dac_int8": False, "compile": True},
    ]

    # Add INT8 configs only if torchao is compatible
    if INT8_AVAILABLE:
        configs.extend([
            {"name": "INT8 (compiled)", "int8": True, "dac_int8": False, "compile": True},
            {"name": "INT8 + DAC INT8 (compiled)", "int8": True, "dac_int8": True, "compile": True},
        ])
    else:
        print("\n  WARNING: INT8 quantization unavailable (torchao/PyTorch version mismatch)")
        print("           Skipping INT8 configurations. Only BF16 will be tested.")

    print(f"\nTesting {len(configs)} configurations...")

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        try:
            result = benchmark_config(
                config_name=config["name"],
                checkpoint_path=checkpoint_path,
                runtime_int8=config["int8"],
                dac_int8=config["dac_int8"],
                compile_mode=config["compile"],
                num_samples=args.num_samples,
                warmup_runs=args.warmup_runs,
            )
            results.append(result)

        except Exception as e:
            print(f"Error benchmarking {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Concurrent benchmarks
    # NOTE: Skipping concurrent benchmark when using torch.compile with reduce-overhead mode
    # because CUDA graphs don't work well with threading (TLS assertion errors)
    concurrent_results = []
    if args.concurrency > 0:
        print("\n" + "=" * 70)
        print("Concurrent Benchmark - SKIPPED")
        print("=" * 70)
        print("  torch.compile with reduce-overhead mode uses CUDA graphs")
        print("  which are incompatible with concurrent threading.")
        print("  Single-request performance is the primary metric for TTS.")

    if False and args.concurrency > 0:  # Disabled - see note above
        print("\n" + "=" * 70)
        print("Concurrent Benchmark")
        print("=" * 70)

        # Use INT8 if available, otherwise BF16 - always compiled
        use_int8 = INT8_AVAILABLE
        config_desc = "INT8 + DAC INT8 (compiled)" if use_int8 else "BF16 (compiled)"
        print(f"  Using {config_desc} configuration for concurrent tests")

        # Test multiple concurrency levels
        concurrency_levels = [1, 2, args.concurrency] if args.concurrency > 2 else [1, args.concurrency]
        concurrency_levels = sorted(set(concurrency_levels))

        for concurrency in concurrency_levels:
            try:
                result = benchmark_concurrent(
                    checkpoint_path=checkpoint_path,
                    concurrency=concurrency,
                    num_requests=concurrency * 2,  # 2 requests per worker
                    runtime_int8=use_int8,
                    dac_int8=use_int8,
                    compile_mode=True,  # Always use torch.compile
                )
                concurrent_results.append(result)
            except Exception as e:
                print(f"Error in concurrent benchmark: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print("\n" + "-" * 85)
        print(f"{'Config':<35} {'VRAM':<10} {'RTF':<10} {'Tokens/s':<12} {'Audio/hr':<12}")
        print("-" * 85)

        for r in results:
            audio_hr_mins = r.audio_per_hour / 60  # Convert seconds to minutes
            print(f"{r.config_name:<35} {r.vram_baseline_mb/1024:>6.1f} GB {r.avg_rtf:>8.3f} {r.tokens_per_sec:>8.1f} {audio_hr_mins:>8.0f} min")

    if concurrent_results:
        print(f"\nConcurrent throughput ({args.concurrency} workers): {concurrent_results[-1].requests_per_sec:.2f} requests/sec")

    # Generate plots
    print("\nGenerating plots...")
    plots = generate_plots(results, concurrent_results)
    print(f"  Generated {len(plots)} plots")

    # Generate HTML report
    if args.output:
        generate_html_report(gpu_info, results, concurrent_results, plots, args.output)

    # Save JSON results
    if args.json:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": gpu_info,
            "args": {
                "compile": True,  # Always enabled
                "num_samples": args.num_samples,
                "concurrency": args.concurrency,
                "warmup_runs": args.warmup_runs,
            },
            "results": [
                {
                    "config_name": r.config_name,
                    "vram_baseline_mb": r.vram_baseline_mb,
                    "vram_peak_mb": r.vram_peak_mb,
                    "vram_delta_mb": r.vram_delta_mb,
                    "llama_vram_mb": r.llama_vram_mb,
                    "dac_vram_mb": r.dac_vram_mb,
                    "warmup_time_s": r.warmup_time_s,
                    "avg_latency_s": r.avg_latency_s,
                    "std_latency_s": r.std_latency_s,
                    "min_latency_s": r.min_latency_s,
                    "max_latency_s": r.max_latency_s,
                    "p50_latency_s": r.p50_latency_s,
                    "p95_latency_s": r.p95_latency_s,
                    "p99_latency_s": r.p99_latency_s,
                    "avg_rtf": r.avg_rtf,
                    "tokens_per_sec": r.tokens_per_sec,
                    "audio_per_minute": r.audio_per_minute,
                    "audio_per_hour": r.audio_per_hour,
                    "throughput_x_realtime": r.throughput_x_realtime,
                    "samples_tested": r.samples_tested,
                    "errors": r.errors,
                    "compile_enabled": r.compile_enabled,
                }
                for r in results
            ],
            "concurrent_results": [
                {
                    "concurrency": r.concurrency,
                    "total_requests": r.total_requests,
                    "total_time_s": r.total_time_s,
                    "requests_per_sec": r.requests_per_sec,
                    "avg_latency_s": r.avg_latency_s,
                    "throughput_x_realtime": r.throughput_x_realtime,
                    "peak_vram_mb": r.peak_vram_mb,
                    "errors": r.errors,
                }
                for r in concurrent_results
            ],
        }

        with open(args.json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"JSON results saved to: {args.json}")

    # Production recommendations
    if results:
        print("\n" + "=" * 70)
        print("PRODUCTION RECOMMENDATIONS")
        print("=" * 70)

        best_throughput = max(results, key=lambda x: x.throughput_x_realtime)
        best_vram = min(results, key=lambda x: x.vram_baseline_mb)

        print(f"\nBest throughput: {best_throughput.config_name}")
        print(f"  {best_throughput.throughput_x_realtime:.2f}x real-time")
        print(f"  ~{best_throughput.audio_per_hour/60:.0f} minutes of audio per hour")

        print(f"\nLowest VRAM: {best_vram.config_name}")
        print(f"  {best_vram.vram_baseline_mb:.0f} MB baseline")

        if best_throughput.avg_latency_s > 0:
            requests_per_min = 60 / best_throughput.avg_latency_s
            print(f"\nEstimated capacity (single GPU, {best_throughput.config_name}):")
            print(f"  Requests/minute: ~{requests_per_min:.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fish-Speech Comprehensive Benchmark Suite (always uses torch.compile)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/runpod/benchmark.py                              # Default benchmark
  python tools/runpod/benchmark.py --num-samples 10             # More samples
  python tools/runpod/benchmark.py --concurrency 4              # Concurrent test
  python tools/runpod/benchmark.py --output report.html         # HTML report
  python tools/runpod/benchmark.py --json results.json          # JSON export

Note: torch.compile is always enabled for optimal performance.
        """
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per test text (default: 5)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent workers for throughput test (default: 4, 0 to disable)"
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.html",
        help="Output path for HTML report (default: benchmark_report.html)"
    )
    parser.add_argument(
        "--json",
        default="benchmark_results.json",
        help="Output path for JSON results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup iterations before benchmarking (default: 2)"
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
