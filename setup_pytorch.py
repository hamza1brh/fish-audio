#!/usr/bin/env python3
"""
Automatic PyTorch environment setup for RTX 50-series compatibility.

This script:
1. Detects GPU compute capability
2. Installs appropriate PyTorch version (nightly for sm_120, stable otherwise)
3. Must be run BEFORE installing other dependencies

Usage:
    python setup_pytorch.py
"""

import subprocess
import sys
import os
from pathlib import Path

def get_gpu_compute_capability():
    """Detect GPU compute capability."""
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            return f"sm_{capability[0]}{capability[1]}"
    except ImportError:
        pass
    
    # If torch not installed, try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        major, minor = result.stdout.strip().split('.')
        return f"sm_{major}{minor}"
    except:
        pass
    
    return None

def install_pytorch_nightly():
    """Install PyTorch nightly for RTX 50-series."""
    print("RTX 50-series GPU detected (sm_120)")
    print("Installing PyTorch Nightly (CUDA 12.8)...")
    
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "--pre",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
    ], check=True)
    
    print("PyTorch Nightly installed successfully")

def install_pytorch_stable(cuda_version="cu124"):
    """Install stable PyTorch for standard GPUs."""
    print(f"Installing PyTorch Stable ({cuda_version})...")
    
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
    ], check=True)
    
    print("PyTorch Stable installed successfully")

def detect_cuda_version():
    """Detect installed CUDA version."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        if "release 12.4" in result.stdout or "release 12.6" in result.stdout:
            return "cu124"
        elif "release 11.8" in result.stdout:
            return "cu118"
        elif "release 13" in result.stdout:
            return "cu124"  # Fallback for 13.x
    except:
        pass
    
    # Try checking nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )
        if "CUDA Version: 12.4" in result.stdout or "CUDA Version: 12.6" in result.stdout:
            return "cu124"
        elif "CUDA Version: 11.8" in result.stdout:
            return "cu118"
        elif "CUDA Version: 13" in result.stdout:
            return "cu124"
    except:
        pass
    
    return "cu124"  # Default

def main():
    print("=" * 60)
    print("PyTorch Compatibility Layer Setup")
    print("=" * 60)
    print()
    
    # Check for environment variable overrides
    force_nightly = os.getenv("PYTORCH_FORCE_NIGHTLY", "").lower() == "true"
    force_stable = os.getenv("PYTORCH_FORCE_STABLE", "").lower() == "true"
    
    if force_nightly:
        print("PYTORCH_FORCE_NIGHTLY is set")
        install_pytorch_nightly()
    elif force_stable:
        print("PYTORCH_FORCE_STABLE is set")
        cuda_version = detect_cuda_version()
        install_pytorch_stable(cuda_version)
    else:
        # Auto-detect GPU
        compute_cap = get_gpu_compute_capability()
        
        if compute_cap == "sm_120":
            print(f"Detected: RTX 50-series GPU ({compute_cap})")
            print("Strategy: Install PyTorch Nightly")
            install_pytorch_nightly()
        elif compute_cap:
            print(f"Detected: GPU with {compute_cap}")
            print("Strategy: Install PyTorch Stable")
            cuda_version = detect_cuda_version()
            print(f"CUDA Version: {cuda_version}")
            install_pytorch_stable(cuda_version)
        else:
            print("No GPU detected or nvidia-smi not available")
            print("Strategy: Install CPU-only PyTorch")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--upgrade",
                "torch", "torchvision", "torchaudio"
            ], check=True)
    
    # Verify installation
    print()
    print("=" * 60)
    print("Verifying PyTorch installation...")
    print("=" * 60)
    
    try:
        verify_code = """import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: sm_{cap[0]}{cap[1]}')
else:
    print('CUDA Version: N/A')
    print('Device: CPU')
"""
        result = subprocess.run([sys.executable, "-c", verify_code], 
                               check=True, text=True, capture_output=True)
        
        print(result.stdout)
        
        print()
        print("PyTorch setup complete!")
        print()
        print("Next steps:")
        print("  1. Install project dependencies: poetry install")
        print("  2. Or: pip install -r requirements.txt")
        print()
        
    except subprocess.CalledProcessError as e:
        print("ERROR: PyTorch verification failed!")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

