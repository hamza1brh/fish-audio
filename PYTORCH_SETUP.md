# PyTorch Setup - Quick Reference

## The Problem

RTX 50-series GPUs (RTX 5070 Ti) need PyTorch Nightly, but `poetry install` keeps reinstalling stable PyTorch from dependencies, breaking GPU support.

## The Solution

**Separate PyTorch from project dependencies** - install it as a system-level compatibility layer first.

## Quick Start

### Option 1: Automated Installation (Recommended)

**Linux/Mac:**
```bash
chmod +x install.sh setup_pytorch.py
./install.sh
```

**Windows:**
```powershell
python setup_pytorch.py
poetry install
```

### Option 2: Manual Installation

**1. Install PyTorch First**
```bash
python setup_pytorch.py
```

This auto-detects your GPU and installs:
- **RTX 50-series**: PyTorch Nightly (CUDA 12.8)
- **Other GPUs**: PyTorch Stable (CUDA 12.4)
- **No GPU**: CPU-only PyTorch

**2. Install Project Dependencies**
```bash
poetry install
# or
pip install -r requirements.txt
```

## How It Works

```
┌─────────────────────────────────────┐
│  1. setup_pytorch.py                │
│     - Detects GPU (sm_120 = RTX 50) │
│     - Installs correct PyTorch       │
│     - Stays persistent               │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  2. poetry install                  │
│     - Installs fish-speech          │
│     - Installs streamlit, etc.      │
│     - SKIPS torch (already there)   │
└─────────────────────────────────────┘
```

## Usage in Different Environments

### Local Development (RTX 5070 Ti)
```bash
python setup_pytorch.py  # Installs nightly
poetry install
streamlit run neymar_voice_app.py
```

### Cloud/SageMaker (T4, A100, etc.)
```bash
python setup_pytorch.py  # Installs stable
poetry install
streamlit run neymar_voice_app.py
```

**Same commands, different PyTorch!**

## After Git Pull

If `pyproject.toml` hasn't changed:
```bash
poetry install  # Just updates app dependencies
```

If PyTorch-related issues occur:
```bash
python setup_pytorch.py  # Re-check PyTorch
poetry install
```

## Environment Variables (Optional)

Force specific PyTorch version:

```bash
# Force nightly (testing RTX 50 compatibility on other GPUs)
PYTORCH_FORCE_NIGHTLY=true python setup_pytorch.py

# Force stable
PYTORCH_FORCE_STABLE=true python setup_pytorch.py
```

## Verification

Check your PyTorch installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output (RTX 5070 Ti):
```
PyTorch: 2.11.0.dev20260101+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

Expected output (Cloud):
```
PyTorch: 2.5.1+cu124
CUDA: True
GPU: Tesla T4
```

## Troubleshooting

### Issue: "CUDA capability sm_120 is not compatible"

**Solution**: Reinstall PyTorch
```bash
pip uninstall torch torchvision torchaudio
python setup_pytorch.py
```

### Issue: Poetry reinstalls wrong PyTorch

**Cause**: PyTorch is in `pyproject.toml` dependencies

**Solution**: Remove torch from `pyproject.toml`:
```toml
[tool.poetry.dependencies]
python = "^3.10"
# torch = "^2.5.1"  # ❌ REMOVE THIS
streamlit = "^1.32.0"  # ✅ Keep everything else
```

Then:
```bash
python setup_pytorch.py
poetry install
```

### Issue: Import errors after update

**Solution**: Refresh installation
```bash
python setup_pytorch.py
poetry install --no-deps
poetry install
```

## Key Files

- `setup_pytorch.py` - Auto-detect and install PyTorch
- `install.sh` - Linux/Mac one-command installer
- `install.ps1` - Windows one-command installer
- `RTX_PYTORCH_COMPATIBILITY.md` - Full technical documentation

## Summary

✅ One command works everywhere: `./install.sh`
✅ No manual PyTorch version management
✅ RTX 50-series gets nightly automatically
✅ Cloud environments get stable automatically
✅ Poetry updates don't break PyTorch
✅ Same codebase, different hardware = no changes

**The golden rule**: Always run `python setup_pytorch.py` before `poetry install`.

