# Fish Speech Installation Script (Windows)

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Fish Speech Installation Script" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Step 1: Setup PyTorch
Write-Host ""
Write-Host "Step 1: Setting up PyTorch..." -ForegroundColor Yellow
python setup_pytorch.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch setup failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Install dependencies
Write-Host ""
Write-Host "Step 2: Installing dependencies..." -ForegroundColor Yellow
if (Get-Command poetry -ErrorAction SilentlyContinue) {
    Write-Host "Using Poetry..."
    poetry install
} else {
    Write-Host "Using pip..."
    pip install streamlit groq python-dotenv soundfile numpy
    pip install fish-speech
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dependency installation failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Download models
Write-Host ""
Write-Host "Step 3: Checking models..." -ForegroundColor Yellow
python -c @"
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
"@

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the app:"
Write-Host "  streamlit run neymar_voice_app.py"



