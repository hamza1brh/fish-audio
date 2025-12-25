#!/bin/bash
# Fix torchaudio version mismatch with PyTorch
set -e

echo "=========================================="
echo "Fixing PyTorch/torchaudio Version Mismatch"
echo "=========================================="
echo ""

# Check current versions
echo "[1/4] Checking current versions..."
python -c "
import torch
try:
    import torchaudio
    print(f'PyTorch: {torch.__version__}')
    print(f'torchaudio: {torchaudio.__version__}')
except ImportError:
    print('torchaudio not installed')
"

# Get PyTorch version and CUDA version
echo ""
echo "[2/4] Detecting PyTorch and CUDA versions..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | cut -d+ -f1)
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "cu121")
    echo "PyTorch: $PYTORCH_VERSION"
    echo "CUDA: $CUDA_VERSION"
    
    # Determine torchaudio index URL based on CUDA
    if [[ "$CUDA_VERSION" == "12.4"* ]] || [[ "$CUDA_VERSION" == "13"* ]]; then
        TORCHAUDIO_URL="https://download.pytorch.org/whl/cu124"
    elif [[ "$CUDA_VERSION" == "12.1"* ]]; then
        TORCHAUDIO_URL="https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
        TORCHAUDIO_URL="https://download.pytorch.org/whl/cu118"
    else
        TORCHAUDIO_URL="https://download.pytorch.org/whl/cu121"
    fi
else
    TORCHAUDIO_URL="https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "[3/4] Uninstalling mismatched torchaudio..."
pip uninstall -y torchaudio 2>/dev/null || true

echo ""
echo "[4/4] Reinstalling torchaudio matching PyTorch version..."
pip install torchaudio --index-url "$TORCHAUDIO_URL" --force-reinstall

echo ""
echo "=========================================="
echo "Verifying installation..."
python -c "
import torch
import torchaudio
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ torchaudio: {torchaudio.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
try:
    # Test import
    import torchaudio
    print('✓ torchaudio imports successfully')
except Exception as e:
    print(f'✗ torchaudio import failed: {e}')
    exit(1)
"

echo ""
echo "=========================================="
echo "Fix complete! Try running GPU tests again:"
echo "  pytest tests/ --gpu -v"
echo "=========================================="

