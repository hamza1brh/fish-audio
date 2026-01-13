#!/bin/bash
set -e

echo "==================================="
echo "Fish Speech Installation Script"
echo "==================================="

# Step 1: Setup PyTorch compatibility layer
echo ""
echo "Step 1: Setting up PyTorch..."
python setup_pytorch.py

# Step 2: Install project dependencies
echo ""
echo "Step 2: Installing project dependencies..."
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install
else
    echo "Using pip..."
    pip install streamlit groq python-dotenv soundfile numpy
    pip install fish-speech
fi

# Step 3: Download models if needed
echo ""
echo "Step 3: Checking models..."
python -c "
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
    
    model_dir = Path('checkpoints/openaudio-s1-mini')
    if not (model_dir / 'model.pth').exists():
        print('Downloading model...')
        snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(model_dir))
        print('Model downloaded')
    else:
        print('Model already exists')
except ImportError:
    print('huggingface_hub not installed yet, skipping model download')
    print('You can download manually later')
"

echo ""
echo "Installation complete!"
echo ""
echo "To run the app:"
echo "  streamlit run neymar_voice_app.py"



