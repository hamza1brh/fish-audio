# SageMaker Installation Fix

## Your Environment
- **GPU**: Tesla T4 (16GB VRAM)
- **CUDA**: 13.0
- **Driver**: 580.105.08

## Problem
`pyaudio` fails to build because PortAudio system libraries are missing.

## Solution

Run these commands **before** `python install.py`:

```bash
# Install PortAudio system libraries
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Then run the installer (will auto-detect CUDA 13.0 and use cu124 wheels)
python install.py
```

**Note**: CUDA 13.0 is detected automatically. PyTorch will use cu124 wheels (backward compatible with CUDA 13.0).

## Alternative: Use the SageMaker Fix Script (Recommended)

This script handles pyaudio failures gracefully:

```bash
chmod +x install-sagemaker-fix.sh
./install-sagemaker-fix.sh
```

This will:
1. Install PortAudio system libraries
2. Try to install pyaudio (continues if it fails)
3. Install fish-speech (with or without pyaudio)
4. Verify the installation

**This is the recommended approach for SageMaker environments.**

## If You Already Ran install.py and It Failed

**Quick Fix (Recommended):**

```bash
chmod +x quick-fix-sagemaker.sh
./quick-fix-sagemaker.sh
```

**Or Manual Fix:**

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Install fish-speech without pyaudio
pip install fish-speech --no-deps

# Install dependencies manually (see quick-fix-sagemaker.sh for full list)
pip install numpy<=1.26.4 transformers>=4.45.2 datasets==2.18.0 lightning>=2.1.0 ...
```

## Why This Happens

- `fish-speech` requires `pyaudio` as a dependency
- `pyaudio` needs PortAudio C libraries to compile
- SageMaker images don't include PortAudio by default
- The voice-service API doesn't actually need `pyaudio` (it's only for CLI audio playback), but `fish-speech` requires it as a dependency

## Verification

After installation, verify:

```bash
python -c "import pyaudio; print('pyaudio OK')"
python -c "import fish_speech; print('fish-speech OK')"
```

