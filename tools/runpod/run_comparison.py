#!/usr/bin/env python3
"""
Fish-Speech INT8 vs BF16 Comparison Benchmark

This script automates the full comparison workflow:
1. Quantizes model to INT8 (if not already done)
2. Runs heavy benchmark on BF16 model
3. Clears VRAM (via subprocess isolation)
4. Runs heavy benchmark on INT8 model
5. Generates comparison report

Usage:
    python tools/runpod/run_comparison.py
    python tools/runpod/run_comparison.py --num-samples 20  # Heavy test
    python tools/runpod/run_comparison.py --skip-quantize   # If already quantized

Why subprocesses?
    CUDA memory isn't fully released within the same Python process.
    Running each benchmark in a separate subprocess ensures clean VRAM state.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def run_command(cmd, description, capture_output=False):
    """Run a shell command with logging."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")

    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result
    else:
        return subprocess.run(cmd, shell=True)


def get_gpu_memory():
    """Get current GPU memory usage."""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(", ")
        return int(used), int(total)
    except:
        return 0, 0


def find_int8_checkpoint():
    """Find the most recent INT8 quantized checkpoint."""
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    int8_dirs = sorted(checkpoints_dir.glob("openaudio-s1-mini-int8-torchao-*"))
    if int8_dirs:
        return str(int8_dirs[-1])  # Most recent
    return None


def quantize_model():
    """Quantize the model to INT8."""
    print("\n" + "="*70)
    print("  STEP 1: Quantizing Model to INT8")
    print("="*70)

    # Check if already quantized
    existing = find_int8_checkpoint()
    if existing:
        print(f"  Found existing INT8 checkpoint: {existing}")
        response = input("  Use existing? [Y/n]: ").strip().lower()
        if response != 'n':
            return existing

    # Run quantization
    cmd = "python tools/llama/quantize.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int8"
    result = run_command(cmd, "Quantizing to INT8...")

    if result.returncode != 0:
        print("ERROR: Quantization failed!")
        sys.exit(1)

    # Find the newly created checkpoint
    return find_int8_checkpoint()


def run_benchmark_subprocess(checkpoint_path, output_prefix, num_samples, warmup_runs):
    """Run benchmark in a subprocess for clean VRAM."""

    # Create a temporary benchmark script that runs and exits
    benchmark_script = f'''
import sys
sys.path.insert(0, "{PROJECT_ROOT}")
import os
os.chdir("{PROJECT_ROOT}")

import torch
import json
from pathlib import Path

# Clear any existing CUDA state
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Import benchmark functions
from tools.runpod.benchmark import benchmark_config, get_gpu_info, clear_vram, TEST_TEXTS

checkpoint_path = "{checkpoint_path}"
num_samples = {num_samples}
warmup_runs = {warmup_runs}

print("\\nGPU Memory before benchmark:")
print(f"  Allocated: {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

# Check if pre-quantized
config_path = Path(checkpoint_path) / "config.json"
is_prequantized = False
config_name = "BF16 (compiled)"

if config_path.exists():
    with open(config_path) as f:
        config_data = json.load(f)
        if "quantization" in config_data:
            is_prequantized = True
            mode = config_data["quantization"].get("mode", "int8")
            config_name = f"{{mode.upper()}} pre-quantized (compiled)"

print(f"\\nRunning benchmark: {{config_name}}")
print(f"  Checkpoint: {{checkpoint_path}}")
print(f"  Samples: {{num_samples}} per text ({{len(TEST_TEXTS)}} texts = {{num_samples * len(TEST_TEXTS)}} total)")

# Run benchmark
result = benchmark_config(
    config_name=config_name,
    checkpoint_path=checkpoint_path,
    runtime_int8=False,  # Don't apply runtime quant - model is already quantized or BF16
    dac_int8=is_prequantized,  # Use DAC INT8 for quantized models
    compile_mode=True,
    num_samples=num_samples,
    warmup_runs=warmup_runs,
)

# Save results
output_file = "{output_prefix}_results.json"
results_dict = {{
    "config_name": result.config_name,
    "checkpoint_path": checkpoint_path,
    "is_prequantized": is_prequantized,
    "vram_baseline_mb": result.vram_baseline_mb,
    "vram_peak_mb": result.vram_peak_mb,
    "warmup_time_s": result.warmup_time_s,
    "avg_latency_s": result.avg_latency_s,
    "std_latency_s": result.std_latency_s,
    "p50_latency_s": result.p50_latency_s,
    "p95_latency_s": result.p95_latency_s,
    "p99_latency_s": result.p99_latency_s,
    "avg_rtf": result.avg_rtf,
    "tokens_per_sec": result.tokens_per_sec,
    "throughput_x_realtime": result.throughput_x_realtime,
    "audio_per_hour": result.audio_per_hour,
    "samples_tested": result.samples_tested,
    "errors": result.errors,
}}

with open(output_file, "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\\nResults saved to: {{output_file}}")
'''

    # Write temp script
    temp_script = PROJECT_ROOT / f"_temp_benchmark_{output_prefix}.py"
    with open(temp_script, "w") as f:
        f.write(benchmark_script)

    # Run in subprocess
    print(f"\n{'='*70}")
    print(f"  Running benchmark in isolated subprocess")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    result = subprocess.run([sys.executable, str(temp_script)])

    # Cleanup temp script
    temp_script.unlink()

    # Load and return results
    results_file = PROJECT_ROOT / f"{output_prefix}_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def generate_comparison_report(bf16_results, int8_results, output_path):
    """Generate HTML comparison report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate improvements
    vram_reduction = ((bf16_results["vram_baseline_mb"] - int8_results["vram_baseline_mb"])
                      / bf16_results["vram_baseline_mb"] * 100)
    speed_diff = ((int8_results["tokens_per_sec"] - bf16_results["tokens_per_sec"])
                  / bf16_results["tokens_per_sec"] * 100)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Fish-Speech INT8 vs BF16 Comparison</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .better {{ color: #27ae60; font-weight: bold; }}
        .worse {{ color: #e74c3c; font-weight: bold; }}
        .highlight {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .highlight h2 {{ margin: 0; font-size: 24px; }}
        .highlight .value {{ font-size: 48px; font-weight: bold; margin: 10px 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Fish-Speech INT8 vs BF16 Comparison</h1>
    <p>Generated: {timestamp}</p>

    <div class="highlight">
        <h2>VRAM Reduction</h2>
        <div class="value">{vram_reduction:.1f}%</div>
        <p>{bf16_results["vram_baseline_mb"]:.0f} MB → {int8_results["vram_baseline_mb"]:.0f} MB</p>
    </div>

    <div class="card">
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>BF16</th>
                <th>INT8</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td><strong>VRAM Baseline</strong></td>
                <td>{bf16_results["vram_baseline_mb"]:.0f} MB</td>
                <td>{int8_results["vram_baseline_mb"]:.0f} MB</td>
                <td class="better">-{vram_reduction:.1f}%</td>
            </tr>
            <tr>
                <td><strong>VRAM Peak</strong></td>
                <td>{bf16_results["vram_peak_mb"]:.0f} MB</td>
                <td>{int8_results["vram_peak_mb"]:.0f} MB</td>
                <td class="better">-{((bf16_results["vram_peak_mb"] - int8_results["vram_peak_mb"]) / bf16_results["vram_peak_mb"] * 100):.1f}%</td>
            </tr>
            <tr>
                <td><strong>Tokens/sec</strong></td>
                <td>{bf16_results["tokens_per_sec"]:.1f}</td>
                <td>{int8_results["tokens_per_sec"]:.1f}</td>
                <td class="{'better' if speed_diff >= 0 else 'worse'}">{speed_diff:+.1f}%</td>
            </tr>
            <tr>
                <td><strong>RTF</strong></td>
                <td>{bf16_results["avg_rtf"]:.3f}</td>
                <td>{int8_results["avg_rtf"]:.3f}</td>
                <td>{((int8_results["avg_rtf"] - bf16_results["avg_rtf"]) / bf16_results["avg_rtf"] * 100):+.1f}%</td>
            </tr>
            <tr>
                <td><strong>Throughput</strong></td>
                <td>{bf16_results["throughput_x_realtime"]:.2f}x realtime</td>
                <td>{int8_results["throughput_x_realtime"]:.2f}x realtime</td>
                <td>{((int8_results["throughput_x_realtime"] - bf16_results["throughput_x_realtime"]) / bf16_results["throughput_x_realtime"] * 100):+.1f}%</td>
            </tr>
            <tr>
                <td><strong>Avg Latency</strong></td>
                <td>{bf16_results["avg_latency_s"]:.2f}s</td>
                <td>{int8_results["avg_latency_s"]:.2f}s</td>
                <td>{((int8_results["avg_latency_s"] - bf16_results["avg_latency_s"]) / bf16_results["avg_latency_s"] * 100):+.1f}%</td>
            </tr>
            <tr>
                <td><strong>P95 Latency</strong></td>
                <td>{bf16_results["p95_latency_s"]:.2f}s</td>
                <td>{int8_results["p95_latency_s"]:.2f}s</td>
                <td>{((int8_results["p95_latency_s"] - bf16_results["p95_latency_s"]) / bf16_results["p95_latency_s"] * 100):+.1f}%</td>
            </tr>
            <tr>
                <td><strong>Audio/Hour</strong></td>
                <td>{bf16_results["audio_per_hour"]/60:.0f} min</td>
                <td>{int8_results["audio_per_hour"]/60:.0f} min</td>
                <td>{((int8_results["audio_per_hour"] - bf16_results["audio_per_hour"]) / bf16_results["audio_per_hour"] * 100):+.1f}%</td>
            </tr>
            <tr>
                <td><strong>Warmup Time</strong></td>
                <td>{bf16_results["warmup_time_s"]:.1f}s</td>
                <td>{int8_results["warmup_time_s"]:.1f}s</td>
                <td>-</td>
            </tr>
            <tr>
                <td><strong>Samples Tested</strong></td>
                <td>{bf16_results["samples_tested"]}</td>
                <td>{int8_results["samples_tested"]}</td>
                <td>-</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Recommendation</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="label">For Production (Balanced)</div>
                <div class="value">INT8</div>
                <p>{vram_reduction:.0f}% less VRAM, similar speed</p>
            </div>
            <div class="metric">
                <div class="label">For Maximum Quality</div>
                <div class="value">BF16</div>
                <p>Full precision, {bf16_results["vram_baseline_mb"]/1024:.1f} GB VRAM</p>
            </div>
            <div class="metric">
                <div class="label">For Small GPUs (&lt;8GB)</div>
                <div class="value">INT8</div>
                <p>Only {int8_results["vram_baseline_mb"]/1024:.1f} GB baseline</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Test Configuration</h2>
        <p><strong>BF16 Checkpoint:</strong> {bf16_results.get("checkpoint_path", "checkpoints/openaudio-s1-mini")}</p>
        <p><strong>INT8 Checkpoint:</strong> {int8_results.get("checkpoint_path", "N/A")}</p>
        <p><strong>Samples per text:</strong> {bf16_results["samples_tested"] // 4}</p>
        <p><strong>Total samples:</strong> {bf16_results["samples_tested"]}</p>
    </div>

    <p style="text-align: center; color: #666; margin-top: 40px;">
        Fish-Speech Benchmark Suite | <a href="https://github.com/fishaudio/fish-speech">GitHub</a>
    </p>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nComparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run INT8 vs BF16 comparison benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/runpod/run_comparison.py                    # Standard test
    python tools/runpod/run_comparison.py --num-samples 20   # Heavy test (20 samples per text)
    python tools/runpod/run_comparison.py --skip-quantize    # Skip quantization step
"""
    )

    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of samples per test text (default: 10 for heavy testing)"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=3,
        help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--skip-quantize", action="store_true",
        help="Skip quantization step (use existing INT8 checkpoint)"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_report.html",
        help="Output HTML report path"
    )

    args = parser.parse_args()

    print("="*70)
    print("  Fish-Speech INT8 vs BF16 Comparison Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples per text: {args.num_samples}")
    print(f"  Total samples: {args.num_samples * 4} (4 test texts)")
    print(f"  Warmup runs: {args.warmup_runs}")

    # Check GPU
    used, total = get_gpu_memory()
    print(f"\nGPU Memory: {used} MB / {total} MB")

    # Step 1: Quantize if needed
    bf16_checkpoint = "checkpoints/openaudio-s1-mini"

    if args.skip_quantize:
        int8_checkpoint = find_int8_checkpoint()
        if not int8_checkpoint:
            print("ERROR: No INT8 checkpoint found. Run without --skip-quantize first.")
            sys.exit(1)
    else:
        int8_checkpoint = quantize_model()

    if not int8_checkpoint:
        print("ERROR: Failed to find or create INT8 checkpoint")
        sys.exit(1)

    print(f"\nCheckpoints:")
    print(f"  BF16: {bf16_checkpoint}")
    print(f"  INT8: {int8_checkpoint}")

    # Step 2: Run BF16 benchmark (in subprocess for clean VRAM)
    print("\n" + "="*70)
    print("  STEP 2: Benchmarking BF16 Model")
    print("="*70)

    bf16_results = run_benchmark_subprocess(
        bf16_checkpoint, "bf16", args.num_samples, args.warmup_runs
    )

    if not bf16_results:
        print("ERROR: BF16 benchmark failed")
        sys.exit(1)

    # Step 3: VRAM is automatically cleared because subprocess exits
    print("\n" + "="*70)
    print("  VRAM cleared (subprocess exited)")
    print("="*70)
    used, total = get_gpu_memory()
    print(f"  GPU Memory: {used} MB / {total} MB")

    # Step 4: Run INT8 benchmark (in subprocess for clean VRAM)
    print("\n" + "="*70)
    print("  STEP 3: Benchmarking INT8 Model")
    print("="*70)

    int8_results = run_benchmark_subprocess(
        int8_checkpoint, "int8", args.num_samples, args.warmup_runs
    )

    if not int8_results:
        print("ERROR: INT8 benchmark failed")
        sys.exit(1)

    # Step 5: Generate comparison report
    print("\n" + "="*70)
    print("  STEP 4: Generating Comparison Report")
    print("="*70)

    generate_comparison_report(bf16_results, int8_results, args.output)

    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    vram_reduction = ((bf16_results["vram_baseline_mb"] - int8_results["vram_baseline_mb"])
                      / bf16_results["vram_baseline_mb"] * 100)
    speed_diff = ((int8_results["tokens_per_sec"] - bf16_results["tokens_per_sec"])
                  / bf16_results["tokens_per_sec"] * 100)

    print(f"\n{'Metric':<25} {'BF16':<20} {'INT8':<20} {'Diff':<15}")
    print("-" * 80)
    print(f"{'VRAM Baseline':<25} {bf16_results['vram_baseline_mb']:.0f} MB{'':<12} {int8_results['vram_baseline_mb']:.0f} MB{'':<12} {-vram_reduction:+.1f}%")
    print(f"{'Tokens/sec':<25} {bf16_results['tokens_per_sec']:.1f}{'':<14} {int8_results['tokens_per_sec']:.1f}{'':<14} {speed_diff:+.1f}%")
    print(f"{'RTF':<25} {bf16_results['avg_rtf']:.3f}{'':<15} {int8_results['avg_rtf']:.3f}{'':<15}")
    print(f"{'Throughput':<25} {bf16_results['throughput_x_realtime']:.2f}x{'':<14} {int8_results['throughput_x_realtime']:.2f}x")
    print(f"{'Audio/Hour':<25} {bf16_results['audio_per_hour']/60:.0f} min{'':<12} {int8_results['audio_per_hour']/60:.0f} min")

    print(f"\n  Report saved to: {args.output}")
    print(f"  BF16 results: bf16_results.json")
    print(f"  INT8 results: int8_results.json")


if __name__ == "__main__":
    main()
