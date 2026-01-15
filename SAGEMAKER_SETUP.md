# Fish Speech - AWS SageMaker Setup Guide

## Quick Start (Copy-Paste This)

```bash
# Run this single command on SageMaker:
curl -sSL https://raw.githubusercontent.com/YOUR_REPO/fish-speech/main/scripts/sagemaker_install.sh | bash

# Or if you have the repo cloned:
python scripts/install_sagemaker.py
```

---

## The Problem

Fish Speech has complex dependencies that conflict when installed with `poetry install` or `pip install -e .` directly:

| Issue | Cause |
|-------|-------|
| `numpy` conflicts | Some packages need numpy<2, others need specific versions |
| `transformers` conflicts | Version mismatches with tokenizers, datasets |
| `einx`/`einops` conflicts | Install order matters |
| `descript-audio-codec` | Has its own PyTorch requirements |
| PyTorch CUDA mismatch | Wrong CUDA version for SageMaker GPU |

---

## Solution: Locked Dependencies

I've created a tested, locked requirements file that installs everything in the correct order.

---

## Step-by-Step Manual Setup

If the scripts don't work, follow these exact steps:

### Step 1: Clean Environment

```bash
# Remove any existing broken installs
pip uninstall -y torch torchvision torchaudio numpy transformers einops einx 2>/dev/null
pip cache purge
```

### Step 2: Install PyTorch First

**For SageMaker ml.g4dn (T4), ml.g5 (A10G), ml.p3 (V100), ml.p4 (A100):**

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should output: 2.2.2 True
```

### Step 3: Install numpy (Version Critical!)

```bash
pip install numpy==1.26.4
```

### Step 4: Install Core ML Dependencies

```bash
pip install \
    transformers==4.45.2 \
    tokenizers==0.20.3 \
    safetensors==0.4.5 \
    accelerate==1.1.1 \
    datasets==2.18.0
```

### Step 5: Install einops/einx (Order Matters!)

```bash
pip install einops==0.8.0
pip install "einx[torch]==0.2.2"
```

### Step 6: Install Audio Processing

```bash
pip install \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    resampy==0.4.3 \
    pydub==0.25.1 \
    audioread==3.0.1
```

### Step 7: Install DAC (Vocoder)

```bash
pip install descript-audio-codec==1.0.0
pip install descript-audiotools==0.7.2
```

### Step 8: Install Remaining Dependencies

```bash
pip install \
    lightning==2.4.0 \
    pytorch-lightning==2.4.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    pydantic==2.9.2 \
    loguru==0.7.2 \
    rich==13.9.4 \
    pyrootutils==1.0.4 \
    cachetools==5.5.0 \
    ormsgpack==1.5.0 \
    zstandard==0.23.0 \
    tiktoken==0.8.0 \
    silero-vad==5.1.2 \
    loralib==0.1.2 \
    natsort==8.4.0
```

### Step 9: Install API/Web Dependencies

```bash
pip install \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    gradio==5.6.0 \
    kui==1.6.4 \
    httpx==0.27.2 \
    requests==2.32.3
```

### Step 10: Install Optional (Chinese Support)

```bash
pip install opencc-python-reimplemented==0.1.7 || true
pip install modelscope==1.17.1 || true
```

### Step 11: Install Fish Speech

```bash
cd /path/to/fish-speech
pip install -e . --no-deps
```

**Important:** Use `--no-deps` to avoid pip trying to reinstall dependencies with wrong versions.

### Step 12: Verify Installation

```bash
python -c "
import torch
import numpy as np
import transformers
import librosa
import dac
import fish_speech

print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)
print('Transformers:', transformers.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('All imports successful!')
"
```

---

## Download Model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')
"
```

---

## Test Inference

```bash
# Simple test
python -m s1_mini.tts --text "Hello, this is a test."

# With reference audio (voice cloning)
python -m s1_mini.tts --text "Hello world" --reference_audio reference.wav
```

---

## Common Errors & Fixes

### Error: `No module named 'triton'`
**Not an error!** Triton is optional. The code will fall back to eager mode.

### Error: `numpy.core.multiarray failed to import`
```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

### Error: `cannot import name 'packaging' from 'pkg_resources'`
```bash
pip install --upgrade setuptools
```

### Error: `CUDA out of memory`
Use a larger instance (ml.g5.xlarge or ml.p3.2xlarge).

### Error: `RuntimeError: Couldn't load custom C++ ops`
```bash
pip uninstall torchaudio -y
pip install torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### Error: `ImportError: libsndfile.so`
```bash
# On SageMaker/Ubuntu:
sudo apt-get update && sudo apt-get install -y libsndfile1
```

---

## SageMaker Instance Recommendations

| Instance | GPU | VRAM | Cost | Recommendation |
|----------|-----|------|------|----------------|
| ml.g4dn.xlarge | T4 | 16GB | $ | Development/Testing |
| ml.g5.xlarge | A10G | 24GB | $$ | Production (Best Value) |
| ml.p3.2xlarge | V100 | 16GB | $$$ | High throughput |
| ml.p4d.24xlarge | A100 | 40GB | $$$$ | Overkill for TTS |

**Recommended:** `ml.g5.xlarge` - Best performance/cost ratio for TTS inference.

---

## Expected Performance on SageMaker

| Instance | RTF | Notes |
|----------|-----|-------|
| ml.g4dn.xlarge (T4) | ~1.5x | Real-time capable |
| ml.g5.xlarge (A10G) | ~2.0x | Comfortable real-time |
| ml.p3.2xlarge (V100) | ~2.5x | Fast |

With Triton enabled (Linux), you get these speeds. Much faster than Windows!

---

## Troubleshooting Checklist

- [ ] PyTorch version is 2.2.2 (not 2.0.x or 2.5.x)
- [ ] CUDA shows as available: `torch.cuda.is_available() == True`
- [ ] numpy version is 1.26.4 (not 2.x)
- [ ] Used `pip install -e . --no-deps` for fish-speech
- [ ] Model is downloaded to `checkpoints/openaudio-s1-mini/`
- [ ] libsndfile is installed (for soundfile)

---

## Support

If you still have issues, run this diagnostic:

```bash
python -c "
import sys
print('Python:', sys.version)

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
except Exception as e:
    print(f'PyTorch ERROR: {e}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except Exception as e:
    print(f'NumPy ERROR: {e}')

for pkg in ['transformers', 'librosa', 'einops', 'einx', 'dac', 'lightning', 'hydra', 'fish_speech']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f'{pkg}: {ver}')
    except Exception as e:
        print(f'{pkg}: FAILED - {e}')
"
```

Share the output if asking for help.
