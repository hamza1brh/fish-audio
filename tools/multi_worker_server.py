"""
Multi-Worker Server for Fish Speech TTS

Launches multiple independent API server processes for parallel request handling.
Each process has its own CUDA context and model instance.

Usage:
    python tools/multi_worker_server.py --workers 2 --base-port 8001
    python tools/multi_worker_server.py --workers 3 --base-port 8001 --with-balancer

Architecture:
    ┌─────────────────┐
    │  Load Balancer  │  (optional, port 8000)
    │  (Round Robin)  │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ Worker 1│     │ Worker 2│
│ :8001   │     │ :8002   │
└─────────┘     └─────────┘
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch multiple Fish Speech API server workers"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes to spawn (default: 2)"
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8001,
        help="Starting port number, workers use base-port, base-port+1, etc. (default: 8001)"
    )
    parser.add_argument(
        "--balancer-port",
        type=int,
        default=8000,
        help="Load balancer port (default: 8000)"
    )
    parser.add_argument(
        "--with-balancer",
        action="store_true",
        help="Also start the load balancer on balancer-port"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Enable torch.compile (default: True)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile"
    )
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini",
        help="Path to LLaMA checkpoint"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini/codec.pth",
        help="Path to decoder checkpoint"
    )
    parser.add_argument(
        "--stagger-start",
        type=float,
        default=5.0,
        help="Seconds to wait between starting each worker (default: 5.0, allows kernel caching)"
    )
    return parser.parse_args()


def start_worker(worker_id: int, port: int, args) -> subprocess.Popen:
    """Start a single API server worker process."""

    cmd = [
        sys.executable,
        str(project_root / "tools" / "api_server.py"),
        "--listen", f"0.0.0.0:{port}",
        "--device", args.device,
        "--llama-checkpoint-path", args.llama_checkpoint_path,
        "--decoder-checkpoint-path", args.decoder_checkpoint_path,
    ]

    if args.half:
        cmd.append("--half")

    # Handle compile flag
    use_compile = args.compile and not args.no_compile
    if use_compile:
        cmd.append("--compile")
    else:
        cmd.append("--no-compile")

    # Set environment for this worker
    env = os.environ.copy()
    env["WORKER_ID"] = str(worker_id)

    print(f"[Worker {worker_id}] Starting on port {port}...")
    print(f"[Worker {worker_id}] Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


def start_load_balancer(balancer_port: int, worker_ports: list[int]) -> subprocess.Popen:
    """Start the load balancer process."""

    cmd = [
        sys.executable,
        str(project_root / "tools" / "load_balancer.py"),
        "--port", str(balancer_port),
        "--workers", ",".join(str(p) for p in worker_ports),
    ]

    print(f"[Balancer] Starting on port {balancer_port}...")
    print(f"[Balancer] Distributing to workers: {worker_ports}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


def stream_output(process: subprocess.Popen, prefix: str):
    """Stream process output with prefix."""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"{prefix} {line.rstrip()}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Fish Speech Multi-Worker Server")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    print(f"Ports: {args.base_port} - {args.base_port + args.workers - 1}")
    print(f"Device: {args.device}")
    print(f"Compile: {args.compile and not args.no_compile}")
    if args.with_balancer:
        print(f"Load Balancer: port {args.balancer_port}")
    print("=" * 60)
    print()

    processes = []
    worker_ports = []

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n[Main] Shutting down workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait(timeout=10)
        print("[Main] All workers stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start workers with staggered timing
    for i in range(args.workers):
        port = args.base_port + i
        worker_ports.append(port)

        process = start_worker(i, port, args)
        processes.append(process)

        # Stagger worker starts to allow kernel caching to be shared
        if i < args.workers - 1 and args.stagger_start > 0:
            print(f"[Main] Waiting {args.stagger_start}s before starting next worker...")
            time.sleep(args.stagger_start)

    # Start load balancer if requested
    if args.with_balancer:
        # Wait for workers to be ready
        print("[Main] Waiting for workers to initialize...")
        time.sleep(10)

        balancer_process = start_load_balancer(args.balancer_port, worker_ports)
        processes.append(balancer_process)

    print()
    print("=" * 60)
    print("All workers started!")
    print("=" * 60)
    print()
    print("Worker endpoints:")
    for i, port in enumerate(worker_ports):
        print(f"  Worker {i}: http://localhost:{port}")
    if args.with_balancer:
        print(f"\nLoad Balancer: http://localhost:{args.balancer_port}")
    print()
    print("Press Ctrl+C to stop all workers")
    print()

    # Stream output from all processes
    import threading

    threads = []
    for i, process in enumerate(processes):
        if i < len(worker_ports):
            prefix = f"[Worker {i}]"
        else:
            prefix = "[Balancer]"

        t = threading.Thread(target=stream_output, args=(process, prefix), daemon=True)
        t.start()
        threads.append(t)

    # Wait for any process to exit
    while True:
        for i, p in enumerate(processes):
            retcode = p.poll()
            if retcode is not None:
                print(f"\n[Main] Process {i} exited with code {retcode}")
                # Don't exit on worker failure, just log it
        time.sleep(1)


if __name__ == "__main__":
    main()
