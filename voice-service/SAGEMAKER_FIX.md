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

## Alternative: Use the SageMaker Install Script

```bash
chmod +x install-sagemaker.sh
./install-sagemaker.sh
```

## If You Already Ran install.py and It Failed

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Install pyaudio manually
pip install pyaudio

# Then continue with fish-speech
pip install fish-speech
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

