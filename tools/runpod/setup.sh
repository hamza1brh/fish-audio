#!/bin/bash
# RunPod Setup Script for Fish-Speech
# Tested with: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# GPU: RTX 4090 24GB

set -e

echo "=============================================="
echo "Fish-Speech RunPod Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check CUDA
echo -e "${YELLOW}Checking CUDA...${NC}"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Clone repo if not exists
if [ ! -d "fish-speech" ]; then
    echo -e "${YELLOW}Cloning Fish-Speech repository...${NC}"
    git clone https://github.com/fishaudio/fish-speech.git
    cd fish-speech
else
    echo -e "${GREEN}Fish-Speech directory exists, updating...${NC}"
    cd fish-speech
    git pull
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -e . --quiet

# Install additional dependencies for benchmarking
echo -e "${YELLOW}Installing benchmark dependencies...${NC}"
pip install torchao streamlit aiohttp locust --quiet

# Download model checkpoints
echo -e "${YELLOW}Downloading model checkpoints...${NC}"
if [ ! -d "checkpoints/openaudio-s1-mini" ]; then
    huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
else
    echo -e "${GREEN}Checkpoints already exist${NC}"
fi

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "
import torch
from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.dac.inference import load_model
print('All imports successful!')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo -e "${GREEN}=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Run benchmarks:     python tools/runpod/benchmark.py"
echo "2. Run load test:      python tools/runpod/load_test.py"
echo "3. Start Streamlit:    streamlit run tools/testing/streamlit_quant_test.py --server.port 8501"
echo ""
echo "To expose Streamlit externally on RunPod:"
echo "  streamlit run tools/testing/streamlit_quant_test.py --server.port 8501 --server.address 0.0.0.0"
echo -e "${NC}"
