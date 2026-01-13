#!/bin/bash
# SageMaker Quick Setup Script
# 
# This script sets up the fish-speech voice app on AWS SageMaker
# with automatic PyTorch detection for the SageMaker GPU

set -e

echo "=========================================="
echo "Fish Speech SageMaker Setup"
echo "=========================================="
echo ""

# Step 1: Check environment
echo "Step 1: Checking SageMaker environment..."
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || echo "No GPU detected"
echo ""

echo "Python:"
python --version
echo ""

# Step 2: Setup PyTorch (auto-detects SageMaker GPU)
echo "Step 2: Setting up PyTorch..."
echo "This will auto-detect your SageMaker GPU and install the correct PyTorch version"
echo ""

python setup_pytorch.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: PyTorch setup failed!"
    exit 1
fi

echo ""

# Step 3: Install dependencies
echo "Step 3: Installing project dependencies..."
echo ""

# Check if poetry is available
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install
else
    echo "Using pip..."
    pip install --user streamlit groq python-dotenv soundfile numpy
    pip install --user fish-speech
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Dependency installation failed!"
    exit 1
fi

echo ""

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
echo ""

python -c "
import torch
import fish_speech
import streamlit

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f'✓ Compute Capability: sm_{cap[0]}{cap[1]}')
print('✓ fish-speech imported')
print('✓ streamlit imported')
print('')
print('All components verified!')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Verification failed!"
    exit 1
fi

echo ""

# Step 5: Setup environment file
echo "Step 5: Setting up environment..."
echo ""

if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Add your Groq API key here for LLM integration
GROQ_API_KEY=your_groq_api_key_here
EOL
    echo "✓ Created .env file"
    echo "⚠️  Remember to add your GROQ_API_KEY to .env for LLM features"
else
    echo "✓ .env file already exists"
fi

echo ""

# Step 6: Download models
echo "Step 6: Checking models..."
echo ""

python -c "
from pathlib import Path

model_dir = Path('checkpoints/openaudio-s1-mini')
if not (model_dir / 'model.pth').exists():
    print('Downloading model from HuggingFace...')
    print('This may take a few minutes (~2GB download)')
    from huggingface_hub import snapshot_download
    snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(model_dir))
    print('✓ Model downloaded')
else:
    print('✓ Model already exists')
"

echo ""

# Summary
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Environment Details:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute Cap: sm_{cap[0]}{cap[1]}')
    
    # Check if stable or nightly
    if 'dev' in torch.__version__:
        print('  Type: Nightly (unexpected for SageMaker)')
    else:
        print('  Type: Stable (correct for SageMaker)')
"
echo ""
echo "Next steps:"
echo ""
echo "1. Add your GROQ_API_KEY to .env file (optional)"
echo "   nano .env"
echo ""
echo "2. Run the voice app:"
echo "   streamlit run neymar_voice_app.py"
echo ""
echo "3. Access via ngrok (for external access):"
echo "   pip install pyngrok"
echo "   streamlit run neymar_voice_app.py &"
echo "   python -c 'from pyngrok import ngrok; print(ngrok.connect(8501))'"
echo ""
echo "Enjoy!"



