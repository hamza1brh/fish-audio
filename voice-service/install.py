#!/usr/bin/env python3
"""Cross-platform installer that auto-detects CUDA version."""
import subprocess
import sys
import shutil
from pathlib import Path


def get_cuda_version() -> str | None:
    """Detect installed CUDA version."""
    # Try nvcc first
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Parse "release 12.1" or similar
                    parts = line.split("release")[-1].strip().split(",")[0].split(".")
                    major, minor = parts[0].strip(), parts[1].strip() if len(parts) > 1 else "0"
                    return f"{major}.{minor}"
        except Exception:
            pass

    # Try nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            # nvidia-smi shows driver version, infer CUDA capability
            # Driver 535+ = CUDA 12.x, 525+ = CUDA 12.0, 515+ = CUDA 11.8
            driver = result.stdout.strip().split("\n")[0]
            major = int(driver.split(".")[0])
            if major >= 560:
                return "12.8"
            elif major >= 545:
                return "12.4"
            elif major >= 535:
                return "12.1"
            elif major >= 525:
                return "12.0"
            elif major >= 515:
                return "11.8"
        except Exception:
            pass

    return None


def get_torch_index_url(cuda_version: str | None) -> str:
    """Get PyTorch wheel index URL for CUDA version."""
    if cuda_version is None:
        print("No CUDA detected, installing CPU-only PyTorch")
        return "https://download.pytorch.org/whl/cpu"

    major, minor = cuda_version.split(".")
    cuda_major = int(major)
    cuda_minor = int(minor)

    # Map CUDA version to PyTorch wheel
    if cuda_major >= 12 and cuda_minor >= 8:
        return "https://download.pytorch.org/whl/nightly/cu128"
    elif cuda_major >= 12 and cuda_minor >= 4:
        return "https://download.pytorch.org/whl/cu124"
    elif cuda_major >= 12 and cuda_minor >= 1:
        return "https://download.pytorch.org/whl/cu121"
    elif cuda_major >= 11 and cuda_minor >= 8:
        return "https://download.pytorch.org/whl/cu118"
    else:
        print(f"CUDA {cuda_version} may not be fully supported, trying cu121")
        return "https://download.pytorch.org/whl/cu121"


def main():
    print("=" * 60)
    print("Voice Service Installer")
    print("=" * 60)

    # Detect CUDA
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"Detected CUDA: {cuda_version}")
    else:
        print("CUDA not detected (CPU-only mode)")

    torch_url = get_torch_index_url(cuda_version)
    print(f"PyTorch index: {torch_url}")
    print("-" * 60)

    # Install base requirements
    print("\n[1/3] Installing base dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True)

    # Install PyTorch
    print("\n[2/3] Installing PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchaudio",
        "--index-url", torch_url
    ], check=True)

    # Install fish-speech
    print("\n[3/3] Installing fish-speech...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "fish-speech"
    ], check=True)

    # Download model if not present
    checkpoint_dir = Path("checkpoints/openaudio-s1-mini")
    if not checkpoint_dir.exists() or not (checkpoint_dir / "model.pth").exists():
        print("\n[4/4] Downloading S1-mini model from HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                "fishaudio/openaudio-s1-mini",
                local_dir=str(checkpoint_dir),
            )
            print("  Model downloaded successfully")
        except Exception as e:
            print(f"  Warning: Could not download model: {e}")
            print("  You can download manually or set VOICE_S1_CHECKPOINT_PATH")
    else:
        print("\n[4/4] Model checkpoint already exists, skipping download")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)

    # Verify
    print("\nVerifying installation...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        print(f"  Warning: {e}")

    print("\nTo start the service:")
    print("  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000")
    print("\nTo run tests:")
    print("  pytest tests/ --gpu")


if __name__ == "__main__":
    main()

