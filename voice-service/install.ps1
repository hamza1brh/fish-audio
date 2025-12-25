# Fast installation script for Windows
# Installs PyTorch separately to avoid Poetry hanging

Write-Host "Installing PyTorch first (this may take a few minutes)..." -ForegroundColor Yellow
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing other dependencies with Poetry..." -ForegroundColor Yellow
poetry install --no-interaction

Write-Host "Installation complete!" -ForegroundColor Green






