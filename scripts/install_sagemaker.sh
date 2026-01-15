#!/bin/bash
# Fish Speech - SageMaker Installation Script
#
# This script installs all dependencies in the correct order to avoid conflicts.
# Tested on SageMaker ml.g4dn, ml.g5, ml.p3, and ml.p4 instances.
#
# Usage:
#     chmod +x scripts/install_sagemaker.sh
#     ./scripts/install_sagemaker.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Fish Speech - SageMaker Installation                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
echo "Project root: $(pwd)"

# Check GPU
echo ""
echo "=========================================="
echo "CHECKING GPU"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected (CPU mode)"

# Step 1: Clean up
echo ""
echo "=========================================="
echo "STEP 1: Cleaning up existing packages"
echo "=========================================="
pip uninstall -y torch torchvision torchaudio numpy transformers einops einx descript-audio-codec 2>/dev/null || true
pip cache purge 2>/dev/null || true

# Step 2: Install PyTorch
echo ""
echo "=========================================="
echo "STEP 2: Installing PyTorch 2.2.2 + CUDA 12.1"
echo "=========================================="
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"

# Step 3: Install numpy
echo ""
echo "=========================================="
echo "STEP 3: Installing numpy 1.26.4"
echo "=========================================="
pip install numpy==1.26.4

# Step 4: Install transformers stack
echo ""
echo "=========================================="
echo "STEP 4: Installing transformers stack"
echo "=========================================="
pip install \
    transformers==4.45.2 \
    tokenizers==0.20.3 \
    safetensors==0.4.5 \
    accelerate==1.1.1 \
    datasets==2.18.0 \
    huggingface-hub==0.26.2

# Step 5: Install einops THEN einx (ORDER MATTERS!)
echo ""
echo "=========================================="
echo "STEP 5: Installing einops and einx"
echo "=========================================="
pip install einops==0.8.0
pip install "einx[torch]==0.2.2"

# Step 6: Install audio processing
echo ""
echo "=========================================="
echo "STEP 6: Installing audio processing"
echo "=========================================="
pip install \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    resampy==0.4.3 \
    pydub==0.25.1 \
    audioread==3.0.1 \
    silero-vad==5.1.2

# Step 7: Install DAC (vocoder)
echo ""
echo "=========================================="
echo "STEP 7: Installing descript-audio-codec"
echo "=========================================="
pip install descript-audio-codec==1.0.0 descript-audiotools==0.7.2

# Step 8: Install remaining dependencies
echo ""
echo "=========================================="
echo "STEP 8: Installing remaining dependencies"
echo "=========================================="
pip install \
    lightning==2.4.0 \
    pytorch-lightning==2.4.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    pydantic==2.9.2 \
    loguru==0.7.2 \
    rich==13.9.4 \
    pyrootutils==1.0.4 \
    natsort==8.4.0 \
    cachetools==5.5.0 \
    ormsgpack==1.5.0 \
    zstandard==0.23.0 \
    tiktoken==0.8.0 \
    loralib==0.1.2

# Step 9: Install web/API dependencies
echo ""
echo "=========================================="
echo "STEP 9: Installing web/API dependencies"
echo "=========================================="
pip install \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    gradio==5.6.0 \
    kui==1.6.4 \
    httpx==0.27.2 \
    requests==2.32.3

# Step 10: Install optional packages
echo ""
echo "=========================================="
echo "STEP 10: Installing optional packages"
echo "=========================================="
pip install opencc-python-reimplemented==0.1.7 || true
pip install modelscope==1.17.1 || true
pip install wandb==0.18.7 tensorboard==2.18.0 || true

# Step 11: Install fish-speech
echo ""
echo "=========================================="
echo "STEP 11: Installing fish-speech"
echo "=========================================="
pip install -e . --no-deps

# Step 12: Verification
echo ""
echo "=========================================="
echo "STEP 12: Verifying installation"
echo "=========================================="

python << 'VERIFY'
import sys
errors = []

packages = [
    "torch", "numpy", "transformers", "einops", "einx",
    "librosa", "soundfile", "dac", "lightning", "hydra",
    "pydantic", "gradio", "fish_speech"
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f"✓ {pkg}: {ver}")
    except Exception as e:
        print(f"✗ {pkg}: {e}")
        errors.append(pkg)

print()
if errors:
    print(f"Failed: {', '.join(errors)}")
    sys.exit(1)
else:
    print("All packages installed successfully!")
VERIFY

echo ""
echo "=========================================="
echo "✓ INSTALLATION COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download the model:"
echo "   python -c \"from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')\""
echo ""
echo "2. Test inference:"
echo "   python -m s1_mini.tts --text 'Hello, this is a test.'"
echo ""
echo "3. Start API server:"
echo "   python -m s1_mini.server"
echo ""
echo "For troubleshooting, see SAGEMAKER_SETUP.md"
