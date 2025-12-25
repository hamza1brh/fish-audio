#!/bin/bash
# SageMaker-specific installation script
set -e

echo "Installing system dependencies for SageMaker..."
sudo apt-get update
sudo apt-get install -y portaudio19-dev libasound2-dev

echo "Running main installer..."
python install.py

