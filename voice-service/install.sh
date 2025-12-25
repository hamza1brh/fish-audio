#!/bin/bash
# Fast installation script for Linux/Mac
# Installs PyTorch separately to avoid Poetry hanging

set -e

echo "Installing PyTorch first (this may take a few minutes)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing other dependencies with Poetry..."
poetry install --no-interaction

echo "Installation complete!"





