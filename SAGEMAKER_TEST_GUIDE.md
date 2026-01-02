# SageMaker Setup Guide - Testing PyTorch Adaptability

## Overview

Test the PyTorch auto-detection system on AWS SageMaker with different GPU types (T4, A100, etc.) to verify it works across environments.

## Prerequisites

- AWS SageMaker Studio Lab or SageMaker Notebook Instance
- GPU instance (ml.g4dn.xlarge for T4, ml.p3.2xlarge for V100, etc.)
- Git access

## Quick Start

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/fish-speech.git
cd fish-speech
```

Or if using your current repo:
```bash
cd ~
git clone <your-repo-url>
cd fish-speech
```

### 2. Run Auto-Setup

```bash
# One command - auto-detects SageMaker GPU
python setup_pytorch.py && pip install streamlit groq python-dotenv soundfile numpy fish-speech
```

### 3. Set Up Environment

```bash
# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4. Test Voice App

```bash
# Launch app (use ngrok or port forwarding for access)
streamlit run neymar_voice_app.py --server.port 8501 --server.address 0.0.0.0
```

## Detailed Setup Process

### Step 1: Prepare SageMaker Environment

**Open Terminal in SageMaker Studio**:
1. Open JupyterLab/Studio
2. File → New → Terminal
3. Check GPU availability:

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
+-------------------------------+----------------------+----------------------+
```

### Step 2: Test PyTorch Auto-Detection

**Create test script** (`test_env.py`):

```python
#!/usr/bin/env python3
"""Test SageMaker environment detection."""

import subprocess
import sys

print("=" * 60)
print("SageMaker Environment Test")
print("=" * 60)
print()

# Check GPU
print("GPU Information:")
result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("No GPU detected")

print()

# Check Python
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check CUDA
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("CUDA Compiler:")
    print(result.stdout)
else:
    print("nvcc not found (normal for runtime-only CUDA)")

print()
print("=" * 60)
print("Ready to test setup_pytorch.py")
print("=" * 60)
```

Run it:
```bash
python test_env.py
```

### Step 3: Run PyTorch Setup

```bash
# This should detect Tesla T4 (or whatever GPU SageMaker has)
# and install PyTorch Stable (NOT nightly, since it's not sm_120)
python setup_pytorch.py
```

**Expected output**:
```
============================================================
PyTorch Compatibility Layer Setup
============================================================

Detected: GPU with sm_75  (or sm_80 for A100)
Strategy: Install PyTorch Stable
CUDA Version: cu124
Installing PyTorch Stable (cu124)...
...
PyTorch: 2.5.1+cu124
CUDA Available: True
CUDA Version: 12.4
Device: Tesla T4
Compute Capability: sm_75

PyTorch setup complete!
```

### Step 4: Install Project Dependencies

```bash
# Option 1: Using pip
pip install streamlit groq python-dotenv soundfile numpy
pip install fish-speech

# Option 2: Using poetry (if available)
pip install poetry
poetry install
```

### Step 5: Verify Installation

```bash
python -c "
import torch
import fish_speech
import streamlit

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ fish-speech imported')
print(f'✓ streamlit imported')
print('All imports successful!')
"
```

### Step 6: Test Voice Generation

Create a quick test script (`test_voice_gen.py`):

```python
#!/usr/bin/env python3
"""Quick voice generation test on SageMaker."""

import torch
import time
from pathlib import Path

print("=" * 60)
print("Voice Generation Test")
print("=" * 60)
print()

# Check setup
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

# Test with subprocess (like the app does)
import subprocess
import sys

PROJECT_ROOT = Path.cwd()

# Simple test: Check if model can be loaded
model_path = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
if not model_path.exists():
    print("Model not found. Downloading...")
    from huggingface_hub import snapshot_download
    snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(model_path))
    print("✓ Model downloaded")

print("✓ Model exists")
print()
print("Ready to test voice generation!")
```

Run it:
```bash
python test_voice_gen.py
```

### Step 7: Launch Streamlit App

**Option A: Using ngrok (easier for testing)**

```bash
# Install ngrok
pip install pyngrok

# Start app with ngrok
streamlit run neymar_voice_app.py &
python -c "from pyngrok import ngrok; public_url = ngrok.connect(8501); print(f'\n\n🌐 Access app at: {public_url}\n\n')"
```

**Option B: Using SageMaker port forwarding**

```bash
# Start app
streamlit run neymar_voice_app.py --server.port 8501 --server.address 0.0.0.0
```

Then use SageMaker's port forwarding or proxy to access.

## Verification Checklist

Test that the system adapted correctly:

- [ ] `setup_pytorch.py` detected SageMaker GPU (T4/V100/A100)
- [ ] Installed **stable** PyTorch (not nightly, since not RTX 50)
- [ ] CUDA works (`torch.cuda.is_available()` is True)
- [ ] GPU operations work (matrix multiplication test)
- [ ] fish-speech imports successfully
- [ ] Voice app launches
- [ ] Can generate voice samples
- [ ] Different from local (nightly vs stable)

## Expected Differences: Local vs SageMaker

| Aspect | Local (RTX 5070 Ti) | SageMaker (T4/A100) |
|--------|---------------------|---------------------|
| **GPU** | RTX 5070 Ti | Tesla T4 / A100 |
| **Compute Cap** | sm_120 | sm_75 / sm_80 |
| **PyTorch** | Nightly (2.11.0.dev) | **Stable (2.5.1)** |
| **CUDA** | 12.8 | 12.0-12.4 |
| **Detection** | Auto → Nightly | Auto → **Stable** |
| **Installation** | Same command! | Same command! |

**Key Test**: Same `setup_pytorch.py` command installs different PyTorch versions based on GPU!

## Troubleshooting SageMaker

### Issue: "CUDA out of memory"

SageMaker instances have limited GPU memory.

**Solution**: Use smaller batch sizes or restart kernel:
```python
# In neymar_voice_app.py, models reload per generation
# so memory shouldn't accumulate
```

### Issue: "Permission denied" when installing

**Solution**: Use `--user` flag:
```bash
pip install --user streamlit groq python-dotenv
```

### Issue: Models take long to download

**Solution**: Models are ~2GB. First download will be slow. They cache in `checkpoints/` for subsequent runs.

### Issue: Can't access Streamlit UI

**Solutions**:

1. **Use ngrok**:
```bash
pip install pyngrok
streamlit run neymar_voice_app.py &
python -c "from pyngrok import ngrok; print(ngrok.connect(8501))"
```

2. **Use SageMaker proxy**:
Check SageMaker Studio documentation for port forwarding

3. **Use Jupyter notebook**:
```python
# In notebook cell
import subprocess
subprocess.Popen(["streamlit", "run", "neymar_voice_app.py"])
```

### Issue: Different PyTorch than local

**This is correct!** The system should install:
- Local (RTX 50): Nightly
- SageMaker (T4/V100/A100): Stable

If both installed the same version, the auto-detection isn't working.

## Performance Comparison

Expected generation times:

| Environment | GPU | PyTorch | Time per Gen |
|-------------|-----|---------|--------------|
| **Local** | RTX 5070 Ti | Nightly | ~2-3s |
| **SageMaker T4** | Tesla T4 | Stable | ~3-5s |
| **SageMaker V100** | Tesla V100 | Stable | ~2-3s |
| **SageMaker A100** | Tesla A100 | Stable | ~1-2s |

## Testing Script for SageMaker

Save as `test_sagemaker_setup.sh`:

```bash
#!/bin/bash
set -e

echo "=================================="
echo "SageMaker PyTorch Setup Test"
echo "=================================="
echo ""

# Test 1: Environment
echo "Test 1: Check environment..."
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Test 2: Setup PyTorch
echo ""
echo "Test 2: Running setup_pytorch.py..."
python setup_pytorch.py

# Test 3: Verify
echo ""
echo "Test 3: Verify installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Should be STABLE (not nightly) on SageMaker
assert 'dev' not in torch.__version__, 'ERROR: Should install stable, not nightly!'
print('✓ Correctly installed stable PyTorch for SageMaker GPU')
"

# Test 4: Install dependencies
echo ""
echo "Test 4: Installing dependencies..."
pip install --user streamlit groq python-dotenv soundfile numpy
pip install --user fish-speech

# Test 5: Test imports
echo ""
echo "Test 5: Testing imports..."
python -c "import fish_speech; import streamlit; print('✓ All imports work')"

echo ""
echo "=================================="
echo "✓ SageMaker setup complete!"
echo "=================================="
echo ""
echo "Run app:"
echo "  streamlit run neymar_voice_app.py"
```

Make executable and run:
```bash
chmod +x test_sagemaker_setup.sh
./test_sagemaker_setup.sh
```

## Success Criteria

The test is successful if:

1. ✅ `setup_pytorch.py` runs without errors
2. ✅ Installs **stable** PyTorch (not nightly)
3. ✅ CUDA is available
4. ✅ GPU operations work
5. ✅ Different PyTorch version than local
6. ✅ Same installation commands as local
7. ✅ Voice app works on SageMaker

## Post-Test: Compare Setups

After testing on both environments:

**Local (RTX 5070 Ti)**:
```bash
python -c "import torch; print(torch.__version__)"
# Expected: 2.11.0.dev20260101+cu128 (nightly)
```

**SageMaker (T4/V100/A100)**:
```bash
python -c "import torch; print(torch.__version__)"
# Expected: 2.5.1+cu124 (stable)
```

**Both used the same command**:
```bash
python setup_pytorch.py
```

✅ **This proves the adaptability works!**

## Cleanup After Testing

If you want to save SageMaker costs:

```bash
# Save backup (optional)
python backup_pytorch.py

# Stop Streamlit
pkill -f streamlit

# Can also stop the notebook instance from AWS Console
```

## Summary

This test will prove:

✅ Same setup script works on different hardware
✅ Auto-detects GPU type and installs appropriate PyTorch
✅ No manual version management needed
✅ Portable across local (RTX 50) and cloud (T4/V100/A100)
✅ `poetry install` or `pip install` doesn't break PyTorch
✅ One codebase, multiple environments

**Let me know the results from SageMaker!** This will confirm the system works as designed.

