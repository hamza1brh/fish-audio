#!/bin/bash
# Script to run GPU tests on SageMaker
set -e

echo "=========================================="
echo "GPU Tests Setup & Run"
echo "=========================================="
echo ""

# Check CUDA availability
echo "[1/4] Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ CUDA available')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  ✗ CUDA not available')
    exit(1)
" || {
    echo "  ERROR: CUDA not available. Cannot run GPU tests."
    exit 1
}

# Check model checkpoint
echo ""
echo "[2/4] Checking model checkpoint..."
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"
if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/model.pth" ]; then
    echo "  ✓ Model checkpoint found at $CHECKPOINT_DIR"
    export VOICE_S1_CHECKPOINT_PATH="$CHECKPOINT_DIR"
else
    echo "  ⚠ Model checkpoint not found"
    echo "  Downloading from HuggingFace..."
    echo "  Note: This model requires HuggingFace authentication"
    echo ""
    python -c "
from huggingface_hub import snapshot_download, login
from pathlib import Path
import os

checkpoint_dir = Path('$CHECKPOINT_DIR')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Try to get token from environment or use login
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
if token:
    print('  Using HuggingFace token from environment...')
else:
    print('  Attempting to use cached HuggingFace credentials...')
    print('  If this fails, run: huggingface-cli login')
    print('  Or set: export HF_TOKEN=your_token_here')

print('  Downloading S1-mini model...')
try:
    snapshot_download(
        'fishaudio/openaudio-s1-mini',
        local_dir=str(checkpoint_dir),
        token=token,
    )
    print('  ✓ Model downloaded')
except Exception as e:
    if '401' in str(e) or 'Unauthorized' in str(e) or 'GatedRepo' in str(e):
        print('  ✗ Authentication required')
        print('  Please run: huggingface-cli login')
        print('  Or set: export HF_TOKEN=your_token_here')
        print('  Then run this script again')
    else:
        print(f'  ✗ Download failed: {e}')
    raise
" || {
        echo ""
        echo "  ✗ Failed to download model"
        echo ""
        echo "  To fix:"
        echo "    1. Run: huggingface-cli login"
        echo "    2. Or set: export HF_TOKEN=your_huggingface_token"
        echo "    3. Then run this script again"
        echo ""
        echo "  Or set VOICE_S1_CHECKPOINT_PATH to an existing checkpoint path"
        exit 1
    }
    export VOICE_S1_CHECKPOINT_PATH="$CHECKPOINT_DIR"
fi

# Set environment variables
echo ""
echo "[3/4] Setting environment variables..."
export VOICE_TTS_PROVIDER="s1_mini"
export VOICE_S1_DEVICE="cuda"
export VOICE_S1_COMPILE="false"
export VOICE_LOG_LEVEL="info"
echo "  ✓ Environment configured"

# Run GPU tests
echo ""
echo "[4/4] Running GPU tests..."
echo "=========================================="
pytest tests/ --gpu -v --tb=short

echo ""
echo "=========================================="
echo "GPU tests complete!"
echo "=========================================="

