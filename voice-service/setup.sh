#!/bin/bash
# One-liner setup for any Linux environment (SageMaker, EC2, etc.)
# Usage: curl -sSL https://raw.githubusercontent.com/.../setup.sh | bash

set -e

echo "========================================"
echo "Voice Service Setup"
echo "========================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Detect CUDA version
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
elif command -v nvidia-smi &> /dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "$DRIVER" -ge 560 ]; then CUDA_VERSION="12.8"
    elif [ "$DRIVER" -ge 545 ]; then CUDA_VERSION="12.4"
    elif [ "$DRIVER" -ge 535 ]; then CUDA_VERSION="12.1"
    elif [ "$DRIVER" -ge 515 ]; then CUDA_VERSION="11.8"
    fi
fi

echo "Detected CUDA: ${CUDA_VERSION:-none}"

# Map to PyTorch wheel
case "$CUDA_VERSION" in
    12.8*) TORCH_URL="https://download.pytorch.org/whl/nightly/cu128" ;;
    12.4*|12.5*|12.6*|12.7*) TORCH_URL="https://download.pytorch.org/whl/cu124" ;;
    12.*) TORCH_URL="https://download.pytorch.org/whl/cu121" ;;
    11.8*) TORCH_URL="https://download.pytorch.org/whl/cu118" ;;
    *) TORCH_URL="https://download.pytorch.org/whl/cu121" ;;
esac

echo "Using PyTorch index: $TORCH_URL"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchaudio --index-url "$TORCH_URL"
pip install fish-speech

# Install dev dependencies if requested
if [ "$1" = "--dev" ]; then
    pip install -r requirements-dev.txt
fi

# Download model if checkpoint doesn't exist
if [ ! -d "checkpoints/openaudio-s1-mini" ]; then
    echo "Downloading S1-mini model..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')
print('Model downloaded successfully')
"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To start the service:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "To run tests:"
echo "  pytest tests/ --gpu"

