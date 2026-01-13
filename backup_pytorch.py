#!/usr/bin/env python3
"""
Backup current PyTorch installation before testing new setup.

This script saves your current PyTorch configuration so you can restore it if needed.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def get_current_torch_info():
    """Get current PyTorch installation details."""
    try:
        result = subprocess.run([
            sys.executable, "-c",
            "import torch, torchvision, torchaudio; "
            "import json; "
            "info = {"
            "  'torch_version': torch.__version__, "
            "  'torchvision_version': torchvision.__version__, "
            "  'torchaudio_version': torchaudio.__version__, "
            "  'cuda_available': torch.cuda.is_available(), "
            "  'cuda_version': torch.version.cuda if torch.cuda.is_available() else None, "
            "  'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
            "  'compute_cap': f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}' if torch.cuda.is_available() else None"
            "}; "
            "print(json.dumps(info))"
        ], capture_output=True, text=True, check=True)
        
        return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"ERROR: Could not get PyTorch info: {e}")
        return None

def save_backup(info):
    """Save backup information to file."""
    backup_dir = Path(".pytorch_backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"pytorch_backup_{timestamp}.json"
    
    backup_data = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "torch_info": info,
        "restore_command": generate_restore_command(info)
    }
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    # Also save as "latest"
    latest_file = backup_dir / "pytorch_backup_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    return backup_file

def generate_restore_command(info):
    """Generate command to restore this exact PyTorch version."""
    torch_ver = info['torch_version']
    torchvision_ver = info['torchvision_version']
    torchaudio_ver = info['torchaudio_version']
    
    # Detect if it's nightly or stable
    if 'dev' in torch_ver or '+cu128' in torch_ver:
        # Nightly version
        return (
            f"pip install --pre torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/nightly/cu128"
        )
    else:
        # Stable version - extract CUDA version from torch version
        if '+cu124' in torch_ver:
            cuda_ver = 'cu124'
        elif '+cu118' in torch_ver:
            cuda_ver = 'cu118'
        else:
            cuda_ver = 'cu124'  # default
        
        # Remove +cuXXX suffix from version
        torch_ver_clean = torch_ver.split('+')[0]
        torchvision_ver_clean = torchvision_ver.split('+')[0]
        torchaudio_ver_clean = torchaudio_ver.split('+')[0]
        
        return (
            f"pip install torch=={torch_ver_clean} "
            f"torchvision=={torchvision_ver_clean} "
            f"torchaudio=={torchaudio_ver_clean} "
            f"--index-url https://download.pytorch.org/whl/{cuda_ver}"
        )

def main():
    print("=" * 60)
    print("PyTorch Backup Tool")
    print("=" * 60)
    print()
    
    print("Detecting current PyTorch installation...")
    info = get_current_torch_info()
    
    if not info:
        print("ERROR: Could not detect PyTorch. Is it installed?")
        sys.exit(1)
    
    print("\nCurrent PyTorch Configuration:")
    print(f"  PyTorch:     {info['torch_version']}")
    print(f"  TorchVision: {info['torchvision_version']}")
    print(f"  TorchAudio:  {info['torchaudio_version']}")
    print(f"  CUDA:        {info['cuda_version'] or 'N/A'}")
    print(f"  GPU:         {info['gpu_name'] or 'CPU'}")
    if info['compute_cap']:
        print(f"  Compute Cap: {info['compute_cap']}")
    
    print("\nSaving backup...")
    backup_file = save_backup(info)
    print(f"Backup saved to: {backup_file}")
    
    print("\n" + "=" * 60)
    print("Backup complete!")
    print("=" * 60)
    
    print("\nTo restore this configuration later:")
    print("  python restore_pytorch.py")
    print("\nOr manually:")
    print(f"  pip uninstall torch torchvision torchaudio")
    print(f"  {generate_restore_command(info)}")
    
    print("\nYou can now safely test the new PyTorch setup:")
    print("  python setup_pytorch.py")

if __name__ == "__main__":
    main()



