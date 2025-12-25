#!/bin/bash
# Quick fix for pyrootutils issue
cd ~/fish-audio/voice-service

# Create .project-root marker in parent directory
touch ../.project-root
echo "Created .project-root marker"

# Set PYTHONPATH
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
echo "Set PYTHONPATH to include parent directory"

# Run tests
echo "Running GPU tests..."
pytest tests/ --gpu -v

