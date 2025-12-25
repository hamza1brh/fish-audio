#!/bin/bash
# SageMaker installation fix - handles pyaudio issues
set -e

echo "=========================================="
echo "SageMaker Voice Service Installation Fix"
echo "=========================================="
echo ""

# Step 1: Install system dependencies
echo "[1/4] Installing system dependencies (PortAudio)..."
sudo apt-get update -qq
sudo apt-get install -y portaudio19-dev libasound2-dev > /dev/null 2>&1 || {
    echo "  Warning: Could not install system dependencies"
    echo "  You may need to run: sudo apt-get install -y portaudio19-dev libasound2-dev"
}
echo "  ✓ System dependencies installed"

# Step 2: Try to install pyaudio
echo ""
echo "[2/4] Attempting to install pyaudio..."
if pip install pyaudio > /dev/null 2>&1; then
    echo "  ✓ pyaudio installed successfully"
    PYAudio_OK=true
else
    echo "  ⚠ pyaudio installation failed (this is OK - not needed for API service)"
    PYAudio_OK=false
fi

# Step 3: Install fish-speech
echo ""
echo "[3/4] Installing fish-speech..."
if [ "$PYAudio_OK" = true ]; then
    # Normal installation
    pip install fish-speech
    echo "  ✓ fish-speech installed with all dependencies"
else
    # Install without pyaudio
    echo "  Installing fish-speech without pyaudio dependency..."
    pip install fish-speech --no-deps
    
    # Install dependencies manually (excluding pyaudio)
    echo "  Installing dependencies..."
    pip install \
        "numpy<=1.26.4" \
        "transformers>=4.45.2" \
        "datasets==2.18.0" \
        "lightning>=2.1.0" \
        "hydra-core>=1.3.2" \
        "tensorboard>=2.14.1" \
        "natsort>=8.4.0" \
        "einops>=0.7.0" \
        "librosa>=0.10.1" \
        "rich>=13.5.3" \
        "gradio>5.0.0" \
        "wandb>=0.15.11" \
        "grpcio>=1.58.0" \
        "kui>=1.6.0" \
        "uvicorn>=0.30.0" \
        "loguru>=0.6.0" \
        "loralib>=0.1.2" \
        "pyrootutils>=1.0.4" \
        "resampy>=0.4.3" \
        "einx[torch]==0.2.2" \
        "zstandard>=0.22.0" \
        "pydub" \
        "modelscope==1.17.1" \
        "opencc-python-reimplemented==0.1.7" \
        "silero-vad" \
        "ormsgpack" \
        "tiktoken>=0.8.0" \
        "pydantic==2.9.2" \
        "cachetools" \
        "descript-audio-codec" \
        "descript-audiotools"
    echo "  ✓ fish-speech installed without pyaudio"
fi

# Step 4: Verify installation
echo ""
echo "[4/4] Verifying installation..."
python -c "import fish_speech; print('  ✓ fish-speech imported successfully')" || {
    echo "  ✗ fish-speech import failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Note: pyaudio is optional and only needed for CLI audio playback."
echo "The voice-service API will work fine without it."
echo ""
echo "To start the service:"
echo "  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000"

