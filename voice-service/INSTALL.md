# Voice Service Installation

## Quick Start

### Option 1: Auto-Install (Recommended)

```bash
cd voice-service
python install.py
```

This will:
1. Detect your CUDA version
2. Install the correct PyTorch wheels
3. Install fish-speech
4. Download the S1-mini model (~3.6 GB)

### Option 2: Docker

```bash
# Default (CUDA 12.1)
docker build -t voice-service .

# For newer GPUs (CUDA 12.4)
docker build --build-arg CUDA_VERSION=12.4 --build-arg PYTORCH_CUDA=cu124 -t voice-service .

# Run with model volume
docker run --gpus all -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  voice-service
```

### Option 3: Manual Install

```bash
# 1. Install base deps
pip install -r requirements.txt

# 2. Install PyTorch for your CUDA version
# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (A100, H100, L4)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (older GPUs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install fish-speech
pip install fish-speech

# 4. Download model
python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')"
```

## Environment-Specific Setup

### AWS SageMaker (Notebook/Studio)

```bash
# In terminal
cd /home/sagemaker-user
git clone <your-repo> voice-service
cd voice-service

# SageMaker has CUDA pre-installed
python install.py
```

### AWS EC2 (g4dn, g5, p4d, p5)

```bash
# g4dn.xlarge: T4 (CUDA 11.8 or 12.1)
# g5.xlarge: A10G (CUDA 12.1 or 12.4)
# p4d.24xlarge: A100 (CUDA 12.4)
# p5.48xlarge: H100 (CUDA 12.4)

# Deep Learning AMI has CUDA pre-installed
source activate pytorch
cd voice-service
python install.py
```

### Google Cloud (Vertex AI, GCE)

```bash
# Works with Deep Learning VM images
python install.py
```

### Local Development (Windows/Linux)

```bash
# Ensure NVIDIA drivers installed
python install.py
```

## Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

## Run the Service

```bash
# Development
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Production
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Run Tests

```bash
# Without GPU (mock provider)
pytest tests/

# With GPU (real inference)
pytest tests/ --gpu
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_HOST` | `0.0.0.0` | Server host |
| `VOICE_PORT` | `8000` | Server port |
| `VOICE_TTS_PROVIDER` | `s1_mini` | TTS provider |
| `VOICE_S1_CHECKPOINT_PATH` | `checkpoints/openaudio-s1-mini` | Model path |
| `VOICE_S1_DEVICE` | `cuda` | Device (cuda/cpu) |
| `VOICE_S1_COMPILE` | `false` | Enable torch.compile (Linux only) |
| `VOICE_LOG_LEVEL` | `info` | Log level |

## Troubleshooting

### "CUDA not available"

1. Check NVIDIA driver: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version

### "Model not found"

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')"
```

### "Out of memory"

Reduce batch size or use a GPU with more VRAM. S1-mini requires ~5GB VRAM.
