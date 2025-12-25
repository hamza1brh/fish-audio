#!/bin/bash
# Quick fix for SageMaker - run this if install.py failed
set -e

echo "Quick fix for SageMaker pyaudio issue..."
echo ""

# Install system deps
echo "Installing PortAudio..."
sudo apt-get update -qq
sudo apt-get install -y portaudio19-dev libasound2-dev

# Install fish-speech without pyaudio
echo "Installing fish-speech without pyaudio..."
pip install fish-speech --no-deps

# Install dependencies manually
echo "Installing dependencies..."
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

echo ""
echo "✓ Installation complete!"
echo "The API service will work fine without pyaudio."

