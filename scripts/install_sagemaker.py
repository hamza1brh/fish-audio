#!/usr/bin/env python3
"""
Fish Speech - SageMaker Installation Script

This script installs all dependencies in the correct order to avoid conflicts.
Tested on SageMaker ml.g4dn, ml.g5, ml.p3, and ml.p4 instances.

Usage:
    python scripts/install_sagemaker.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd, check=True, capture=False):
    """Run a shell command."""
    print(f"\n{'='*60}")
    print(f">>> {cmd}")
    print('='*60)

    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"⚠ Command returned non-zero exit code: {result.returncode}")
        return result.returncode == 0


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*60)
    print("CHECKING GPU")
    print("="*60)

    success, stdout, _ = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                             check=False, capture=True)
    if success and stdout.strip():
        print(f"GPU detected: {stdout.strip()}")
        return True
    else:
        print("⚠ No GPU detected. Continuing with CPU-only setup...")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Fish Speech - SageMaker Installation                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Project root: {project_root}")

    has_gpu = check_gpu()

    # Step 1: Clean up any broken installs
    print("\n" + "="*60)
    print("STEP 1: Cleaning up existing packages")
    print("="*60)

    packages_to_remove = [
        "torch", "torchvision", "torchaudio",
        "numpy", "transformers", "einops", "einx",
        "descript-audio-codec", "descript-audiotools"
    ]

    for pkg in packages_to_remove:
        run(f"pip uninstall -y {pkg}", check=False)

    run("pip cache purge", check=False)

    # Step 2: Install PyTorch
    print("\n" + "="*60)
    print("STEP 2: Installing PyTorch 2.2.2 with CUDA 12.1")
    print("="*60)

    if has_gpu:
        pytorch_url = "https://download.pytorch.org/whl/cu121"
    else:
        pytorch_url = "https://download.pytorch.org/whl/cpu"

    if not run(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url {pytorch_url}"):
        print("❌ PyTorch installation failed!")
        return 1

    # Verify PyTorch
    print("\nVerifying PyTorch...")
    run('python -c "import torch; print(f\'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}\')"')

    # Step 3: Install numpy (critical version)
    print("\n" + "="*60)
    print("STEP 3: Installing numpy 1.26.4")
    print("="*60)

    if not run("pip install numpy==1.26.4"):
        print("❌ numpy installation failed!")
        return 1

    # Step 4: Install transformers stack
    print("\n" + "="*60)
    print("STEP 4: Installing transformers stack")
    print("="*60)

    transformers_pkgs = [
        "transformers==4.45.2",
        "tokenizers==0.20.3",
        "safetensors==0.4.5",
        "accelerate==1.1.1",
        "datasets==2.18.0",
        "huggingface-hub==0.26.2",
    ]
    run(f"pip install {' '.join(transformers_pkgs)}")

    # Step 5: Install einops THEN einx (order matters!)
    print("\n" + "="*60)
    print("STEP 5: Installing einops and einx (ORDER MATTERS)")
    print("="*60)

    run("pip install einops==0.8.0")
    run("pip install 'einx[torch]==0.2.2'")

    # Step 6: Install audio processing
    print("\n" + "="*60)
    print("STEP 6: Installing audio processing libraries")
    print("="*60)

    audio_pkgs = [
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "resampy==0.4.3",
        "pydub==0.25.1",
        "audioread==3.0.1",
        "silero-vad==5.1.2",
    ]
    run(f"pip install {' '.join(audio_pkgs)}")

    # Step 7: Install DAC (vocoder)
    print("\n" + "="*60)
    print("STEP 7: Installing descript-audio-codec")
    print("="*60)

    run("pip install descript-audio-codec==1.0.0 descript-audiotools==0.7.2")

    # Step 8: Install remaining dependencies
    print("\n" + "="*60)
    print("STEP 8: Installing remaining dependencies")
    print("="*60)

    remaining_pkgs = [
        "lightning==2.4.0",
        "pytorch-lightning==2.4.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "pydantic==2.9.2",
        "loguru==0.7.2",
        "rich==13.9.4",
        "pyrootutils==1.0.4",
        "natsort==8.4.0",
        "cachetools==5.5.0",
        "ormsgpack==1.5.0",
        "zstandard==0.23.0",
        "tiktoken==0.8.0",
        "loralib==0.1.2",
    ]
    run(f"pip install {' '.join(remaining_pkgs)}")

    # Step 9: Install web/API dependencies
    print("\n" + "="*60)
    print("STEP 9: Installing web/API dependencies")
    print("="*60)

    web_pkgs = [
        "fastapi==0.115.5",
        "uvicorn==0.32.1",
        "gradio==5.6.0",
        "kui==1.6.4",
        "httpx==0.27.2",
        "requests==2.32.3",
    ]
    run(f"pip install {' '.join(web_pkgs)}")

    # Step 10: Install optional packages
    print("\n" + "="*60)
    print("STEP 10: Installing optional packages")
    print("="*60)

    run("pip install opencc-python-reimplemented==0.1.7", check=False)
    run("pip install modelscope==1.17.1", check=False)
    run("pip install wandb==0.18.7 tensorboard==2.18.0", check=False)

    # Step 11: Install fish-speech (no deps to avoid conflicts)
    print("\n" + "="*60)
    print("STEP 11: Installing fish-speech")
    print("="*60)

    if not run("pip install -e . --no-deps"):
        print("❌ fish-speech installation failed!")
        return 1

    # Step 12: Verification
    print("\n" + "="*60)
    print("STEP 12: Verifying installation")
    print("="*60)

    verify_code = '''
import sys
errors = []

# Core packages
packages = [
    ("torch", "import torch; v=torch.__version__; cuda=torch.cuda.is_available()"),
    ("numpy", "import numpy; v=numpy.__version__"),
    ("transformers", "import transformers; v=transformers.__version__"),
    ("einops", "import einops; v=einops.__version__"),
    ("einx", "import einx; v=einx.__version__"),
    ("librosa", "import librosa; v=librosa.__version__"),
    ("soundfile", "import soundfile; v=soundfile.__version__"),
    ("dac", "import dac; v='OK'"),
    ("lightning", "import lightning; v=lightning.__version__"),
    ("hydra", "import hydra; v=hydra.__version__"),
    ("pydantic", "import pydantic; v=pydantic.__version__"),
    ("gradio", "import gradio; v=gradio.__version__"),
    ("fish_speech", "import fish_speech; v='OK'"),
]

print()
for name, code in packages:
    try:
        exec(code)
        status = f"v" if "v" in dir() else "OK"
        print(f"✓ {name}: {v}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        errors.append(name)

print()
if errors:
    print(f"Failed: {len(errors)} packages - {', '.join(errors)}")
    sys.exit(1)
else:
    print("All packages installed successfully!")
'''

    success, _, _ = run(f'python -c "{verify_code}"', capture=True)
    run(f'python -c "{verify_code}"')

    # Summary
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE")
    print("="*60)

    print("""
Next steps:

1. Download the model:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')"

2. Test inference:
   python -m s1_mini.tts --text "Hello, this is a test."

3. Start API server:
   python -m s1_mini.server

For troubleshooting, see SAGEMAKER_SETUP.md
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
