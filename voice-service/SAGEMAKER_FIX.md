# SageMaker Installation Fix

## Problem
`pyaudio` fails to build because PortAudio system libraries are missing.

## Solution

Run these commands **before** `python install.py`:

```bash
# Install PortAudio system libraries
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

# Then run the installer
python install.py
```

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

