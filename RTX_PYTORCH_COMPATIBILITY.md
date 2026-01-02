# RTX 50-Series PyTorch Compatibility Guide

## Problem Statement

RTX 50-series GPUs (RTX 5070 Ti, etc.) use compute capability `sm_120`, which is not supported by stable PyTorch releases (2.5.1 and earlier). This causes constant dependency conflicts when:

1. Using libraries that specify PyTorch versions in their dependencies
2. Moving between local development (RTX 50-series) and cloud/production (older GPUs)
3. Running `poetry install` which always reinstalls the wrong PyTorch version

## Solution: Compatibility Layer Architecture

The solution is to **decouple PyTorch from project dependencies** and install it as a **system-level compatibility layer** that gets loaded first.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│     Application Layer (fish-speech, etc.)       │
│     - Uses standard torch imports               │
│     - Doesn't specify torch version             │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│     Compatibility Layer (Auto-selected)         │
│                                                  │
│  Local (RTX 50):    Cloud (T4/A100):           │
│  - PyTorch Nightly  - PyTorch Stable 2.5.1      │
│  - CUDA 12.8        - CUDA 12.4/11.8            │
└─────────────────────────────────────────────────┘
```

## Implementation Options

### Option 1: Environment-Based Installation (Recommended)

Create separate installation scripts that detect the environment and install the correct PyTorch version **before** installing project dependencies.

#### File Structure

```
your-project/
├── pyproject.toml          # NO torch dependencies
├── setup_pytorch.py        # Auto-detect and install PyTorch
├── install.sh             # Linux/Mac installer
├── install.ps1            # Windows installer
└── requirements-torch.txt  # PyTorch variants
```

#### 1. Create `setup_pytorch.py`

```python
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
    print("🚀 Installing PyTorch Nightly (CUDA 12.8) for RTX 50-series...")

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--pre",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
    ], check=True)

    print("✅ PyTorch Nightly installed successfully")

def install_pytorch_stable(cuda_version="cu124"):
    """Install stable PyTorch for standard GPUs."""
    print(f"📦 Installing PyTorch Stable ({cuda_version})...")

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
    ], check=True)

    print("✅ PyTorch Stable installed successfully")

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

    return "cu124"  # Default

def main():
    print("=" * 60)
    print("PyTorch Compatibility Layer Setup")
    print("=" * 60)

    # Detect GPU
    compute_cap = get_gpu_compute_capability()

    if compute_cap == "sm_120":
        print(f"🎮 Detected: RTX 50-series GPU ({compute_cap})")
        print("📌 Strategy: Install PyTorch Nightly")
        install_pytorch_nightly()
    elif compute_cap:
        print(f"🎮 Detected: GPU with {compute_cap}")
        print("📌 Strategy: Install PyTorch Stable")
        cuda_version = detect_cuda_version()
        install_pytorch_stable(cuda_version)
    else:
        print("⚠️  No GPU detected or nvidia-smi not available")
        print("📌 Strategy: Install CPU-only PyTorch")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ], check=True)

    # Verify installation
    print("\n" + "=" * 60)
    print("Verifying PyTorch installation...")
    print("=" * 60)

    result = subprocess.run([
        sys.executable, "-c",
        "import torch; print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA Available: {torch.cuda.is_available()}'); "
        "print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); "
        "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    ], check=True)

    print("\n✅ PyTorch setup complete!")
    print("\n📝 Next steps:")
    print("   1. Install project dependencies: poetry install --no-deps")
    print("   2. Or: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
```

#### 2. Update `pyproject.toml`

**CRITICAL**: Remove PyTorch from dependencies!

```toml
[tool.poetry.dependencies]
python = "^3.10"
# torch = "^2.5.1"  # ❌ REMOVE THIS
# torchaudio = "^2.5.1"  # ❌ REMOVE THIS
# torchvision = "^0.20.1"  # ❌ REMOVE THIS

# All other dependencies stay
streamlit = "^1.32.0"
groq = "^0.11.0"
python-dotenv = "^1.0.0"
soundfile = "^0.12.1"
# ... etc
```

#### 3. Create Installation Scripts

**`install.sh`** (Linux/Mac):

```bash
#!/bin/bash
set -e

echo "==================================="
echo "Fish Speech Installation Script"
echo "==================================="

# Step 1: Setup PyTorch compatibility layer
echo ""
echo "Step 1: Setting up PyTorch..."
python setup_pytorch.py

# Step 2: Install project dependencies (without torch)
echo ""
echo "Step 2: Installing project dependencies..."
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install --no-deps  # Don't reinstall torch
    poetry install            # Install remaining deps
else
    echo "Using pip..."
    pip install -r requirements.txt
fi

# Step 3: Download models if needed
echo ""
echo "Step 3: Checking models..."
python -c "
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path('checkpoints/openaudio-s1-mini')
if not (model_dir / 'model.pth').exists():
    print('Downloading model...')
    snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(model_dir))
    print('✓ Model downloaded')
else:
    print('✓ Model already exists')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run the app:"
echo "  streamlit run neymar_voice_app.py"
```

**`install.ps1`** (Windows):

```powershell
# Fish Speech Installation Script (Windows)

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Fish Speech Installation Script" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Step 1: Setup PyTorch
Write-Host ""
Write-Host "Step 1: Setting up PyTorch..." -ForegroundColor Yellow
python setup_pytorch.py

# Step 2: Install dependencies
Write-Host ""
Write-Host "Step 2: Installing dependencies..." -ForegroundColor Yellow
if (Get-Command poetry -ErrorAction SilentlyContinue) {
    Write-Host "Using Poetry..."
    poetry install --no-deps
    poetry install
} else {
    Write-Host "Using pip..."
    pip install -r requirements.txt
}

# Step 3: Download models
Write-Host ""
Write-Host "Step 3: Checking models..." -ForegroundColor Yellow
python -c @"
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path('checkpoints/openaudio-s1-mini')
if not (model_dir / 'model.pth').exists():
    print('Downloading model...')
    snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(model_dir))
    print('✓ Model downloaded')
else:
    print('✓ Model already exists')
"@

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the app:"
Write-Host "  streamlit run neymar_voice_app.py"
```

Make executable:

```bash
chmod +x install.sh setup_pytorch.py
```

#### 4. Usage

**First time setup:**

```bash
# Linux/Mac
./install.sh

# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
```

**After pulling updates:**

```bash
# PyTorch stays, just update other deps
poetry install

# Or if pyproject.toml changed PyTorch requirements:
python setup_pytorch.py  # Re-check and update if needed
poetry install
```

### Option 2: Virtual Environment Layering (Advanced)

Create a base environment with PyTorch, then project environments on top:

```bash
# Base environment (system-wide or project-level)
python -m venv .venv-pytorch
source .venv-pytorch/bin/activate
python setup_pytorch.py

# Project environment (uses base torch)
python -m venv .venv-app --system-site-packages
source .venv-app/bin/activate
poetry install
```

### Option 3: Docker with Multi-Stage Builds

For production, use Docker with conditional PyTorch installation:

**`Dockerfile`**:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Detect and install PyTorch
COPY setup_pytorch.py /tmp/
RUN python3 /tmp/setup_pytorch.py

# Install app (without torch deps)
FROM base AS app
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-deps && poetry install

COPY . .
CMD ["streamlit", "run", "neymar_voice_app.py"]
```

## Best Practices

### 1. Keep PyTorch Separate

**Do this:**

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
streamlit = "^1.32.0"
# NO torch here!

[tool.poetry.group.ml]
optional = true
# Document that torch must be installed separately
```

**Not this:**

```toml
[tool.poetry.dependencies]
torch = "^2.5.1"  # ❌ Will break on RTX 50
```

### 2. Document PyTorch Requirements

Create `PYTORCH_SETUP.md`:

````markdown
# PyTorch Setup

This project requires PyTorch but does NOT include it in dependencies
to maintain compatibility across different GPU architectures.

## Installation

### Automatic (Recommended)

```bash
python setup_pytorch.py
poetry install
```
````

### Manual

- **RTX 50-series**: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- **Other CUDA GPUs**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- **CPU only**: `pip install torch torchvision torchaudio`

````

### 3. CI/CD Configuration

**.github/workflows/test.yml**:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      # Install PyTorch FIRST
      - name: Setup PyTorch
        run: python setup_pytorch.py

      # Then install project deps
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run tests
        run: poetry run pytest
````

### 4. Environment Variables (Optional)

Allow override via environment:

```python
# setup_pytorch.py
import os

FORCE_NIGHTLY = os.getenv("PYTORCH_FORCE_NIGHTLY", "").lower() == "true"
FORCE_STABLE = os.getenv("PYTORCH_FORCE_STABLE", "").lower() == "true"

if FORCE_NIGHTLY:
    install_pytorch_nightly()
elif FORCE_STABLE:
    install_pytorch_stable()
else:
    # Auto-detect
    ...
```

Usage:

```bash
# Force nightly (for testing on non-RTX50 hardware)
PYTORCH_FORCE_NIGHTLY=true python setup_pytorch.py

# Force stable
PYTORCH_FORCE_STABLE=true python setup_pytorch.py
```

## Troubleshooting

### Issue: Poetry reinstalls wrong PyTorch

**Solution**: Use `--no-deps` on first install:

```bash
python setup_pytorch.py
poetry install --no-deps  # Skip dependencies
poetry install            # Install remaining deps (respects existing torch)
```

### Issue: "Module not found" after torch update

**Solution**: Reinstall project in development mode:

```bash
python setup_pytorch.py
pip install -e . --no-deps
```

### Issue: CUDA version mismatch

**Check versions:**

```bash
nvcc --version                    # System CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

**Fix**: Reinstall with matching CUDA:

```bash
pip uninstall torch torchvision torchaudio
python setup_pytorch.py
```

## Summary

✅ **Problem Solved**: No more PyTorch dependency conflicts
✅ **Works Everywhere**: RTX 50, cloud, production
✅ **One Command**: `./install.sh` or `python setup_pytorch.py && poetry install`
✅ **No Manual Edits**: Auto-detects environment
✅ **Fast Updates**: PyTorch persists, only app dependencies update

The key insight: **Treat PyTorch as a system dependency, not a project dependency.**
