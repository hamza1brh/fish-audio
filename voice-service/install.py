#!/usr/bin/env python3
"""Cross-platform installer that auto-detects CUDA version."""
import subprocess
import sys
import shutil
import platform
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
            # First try to get CUDA version directly from nvidia-smi
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=cuda_version", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                cuda_ver = result.stdout.strip().split("\n")[0].strip()
                # Parse "13.0" or "12.4" format
                if cuda_ver:
                    return cuda_ver
            
            # Fallback: infer from driver version
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            # nvidia-smi shows driver version, infer CUDA capability
            # Driver 580+ = CUDA 13.x, 560+ = CUDA 12.8, 545+ = CUDA 12.4, 535+ = CUDA 12.1
            driver = result.stdout.strip().split("\n")[0]
            major = int(driver.split(".")[0])
            if major >= 580:
                return "13.0"
            elif major >= 560:
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
    # CUDA 13.0+ uses cu124 wheels (backward compatible)
    if cuda_major >= 13:
        print(f"CUDA {cuda_version} detected - using cu124 wheels (backward compatible)")
        return "https://download.pytorch.org/whl/cu124"
    elif cuda_major >= 12 and cuda_minor >= 8:
        return "https://download.pytorch.org/whl/nightly/cu128"
    elif cuda_major >= 12 and cuda_minor >= 4:
        return "https://download.pytorch.org/whl/cu124"
    elif cuda_major >= 12 and cuda_minor >= 1:
        return "https://download.pytorch.org/whl/cu121"
    elif cuda_major >= 11 and cuda_minor >= 8:
        return "https://download.pytorch.org/whl/cu118"
    else:
        print(f"CUDA {cuda_version} may not be fully supported, trying cu124")
        return "https://download.pytorch.org/whl/cu124"


def install_system_dependencies():
    """Install system dependencies required for pyaudio (PortAudio)."""
    system = platform.system().lower()
    
    if system == "linux":
        print("\n[0/4] Installing system dependencies (PortAudio)...")
        print("  This is required for pyaudio to compile")
        try:
            # Try apt (Debian/Ubuntu)
            update_result = subprocess.run(
                ["sudo", "apt-get", "update"],
                check=False,
                capture_output=True,
                text=True
            )
            if update_result.returncode != 0:
                print(f"  Warning: apt-get update failed: {update_result.stderr}")
                print("  You may need to run manually: sudo apt-get update")
            
            install_result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "portaudio19-dev", "libasound2-dev"],
                check=False,
                capture_output=True,
                text=True
            )
            if install_result.returncode == 0:
                print("  ✓ System dependencies installed successfully")
            else:
                print(f"  ✗ Failed to install system dependencies: {install_result.stderr}")
                print("  Manual fix: sudo apt-get install -y portaudio19-dev libasound2-dev")
                print("  Then re-run: python install.py")
        except FileNotFoundError:
            print("  Warning: sudo not found. You may need to install PortAudio manually.")
            print("  Run: sudo apt-get install -y portaudio19-dev libasound2-dev")
        except Exception as e:
            print(f"  Warning: Could not install system dependencies: {e}")
            print("  Manual fix: sudo apt-get install -y portaudio19-dev libasound2-dev")
    elif system == "darwin":  # macOS
        print("\n[0/4] Installing system dependencies (PortAudio)...")
        try:
            result = subprocess.run(
                ["brew", "install", "portaudio"],
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  ✓ System dependencies installed successfully")
            else:
                print(f"  ✗ Failed: {result.stderr}")
                print("  Manual fix: brew install portaudio")
        except FileNotFoundError:
            print("  Warning: brew not found. Install Homebrew first or install PortAudio manually.")
        except Exception as e:
            print(f"  Warning: Could not install system dependencies: {e}")
            print("  Manual fix: brew install portaudio")
    else:
        print("\n[0/4] Skipping system dependencies (Windows or unknown OS)")
        print("  Note: pyaudio may require manual installation on Windows")


def main():
    print("=" * 60)
    print("Voice Service Installer")
    print("=" * 60)

    # Install system dependencies first
    install_system_dependencies()

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
    print("\n[1/4] Installing base dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True)

    # Install PyTorch
    print("\n[2/4] Installing PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchaudio",
        "--index-url", torch_url
    ], check=True)

    # Install fish-speech
    print("\n[3/4] Installing fish-speech...")
    print("  Attempting to install pyaudio first (optional, for CLI tools)...")
    pyaudio_result = subprocess.run([
        sys.executable, "-m", "pip", "install", "pyaudio"
    ], capture_output=True, text=True)
    
    pyaudio_installed = pyaudio_result.returncode == 0
    
    if not pyaudio_installed:
        print("  ⚠ pyaudio installation failed (this is OK - not needed for API service)")
        print("  Note: pyaudio is only used for audio playback in CLI tools")
        print("  Installing fish-speech without pyaudio dependency...")
        
        # Install fish-speech without dependencies first
        print("  Installing fish-speech package...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "fish-speech", "--no-deps"
        ], check=True)
        
        # Install fish-speech dependencies manually (excluding pyaudio)
        print("  Installing fish-speech dependencies (excluding pyaudio)...")
        fish_deps = [
            "numpy<=1.26.4",
            "transformers>=4.45.2",
            "datasets==2.18.0",
            "lightning>=2.1.0",
            "hydra-core>=1.3.2",
            "tensorboard>=2.14.1",
            "natsort>=8.4.0",
            "einops>=0.7.0",
            "librosa>=0.10.1",
            "rich>=13.5.3",
            "gradio>5.0.0",
            "wandb>=0.15.11",
            "grpcio>=1.58.0",
            "kui>=1.6.0",
            "uvicorn>=0.30.0",
            "loguru>=0.6.0",
            "loralib>=0.1.2",
            "pyrootutils>=1.0.4",
            "resampy>=0.4.3",
            "einx[torch]==0.2.2",
            "zstandard>=0.22.0",
            "pydub",
            "modelscope==1.17.1",
            "opencc-python-reimplemented==0.1.7",
            "silero-vad",
            "ormsgpack",
            "tiktoken>=0.8.0",
            "pydantic>=2.10.0,<3.0.0",  # SageMaker requires >=2.10.0
            "cachetools",
            "descript-audio-codec",
            "descript-audiotools"
        ]
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + fish_deps, check=True)
        print("  ✓ fish-speech installed successfully (without pyaudio)")
        print("  ✓ API service will work fine without pyaudio")
    else:
        print("  ✓ pyaudio installed successfully")
        print("  Installing fish-speech with all dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "fish-speech"
        ], check=True)
        print("  ✓ fish-speech installed successfully")

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

