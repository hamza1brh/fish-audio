#!/usr/bin/env python3
"""
Restore PyTorch to a previous backup.

This script restores PyTorch to a previously saved configuration.
"""

import subprocess
import sys
import json
from pathlib import Path

def list_backups():
    """List all available backups."""
    backup_dir = Path(".pytorch_backups")
    
    if not backup_dir.exists():
        return []
    
    backups = sorted(backup_dir.glob("pytorch_backup_*.json"), reverse=True)
    return [b for b in backups if b.name != "pytorch_backup_latest.json"]

def load_backup(backup_file):
    """Load backup information."""
    try:
        with open(backup_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load backup: {e}")
        return None

def restore_pytorch(backup_data):
    """Restore PyTorch from backup."""
    restore_cmd = backup_data['restore_command']
    
    print("\nUninstalling current PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchvision", "torchaudio"
    ])
    
    print("\nRestoring PyTorch...")
    print(f"Command: {restore_cmd}")
    
    # Split command and run
    parts = restore_cmd.split()
    subprocess.run([sys.executable, "-m"] + parts, check=True)
    
    print("\nVerifying installation...")
    result = subprocess.run([
        sys.executable, "-c",
        "import torch; "
        "print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA: {torch.cuda.is_available()}'); "
        "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
    ], check=True, capture_output=True, text=True)
    
    print(result.stdout)

def main():
    print("=" * 60)
    print("PyTorch Restore Tool")
    print("=" * 60)
    print()
    
    backups = list_backups()
    
    if not backups:
        print("ERROR: No backups found!")
        print("\nRun 'python backup_pytorch.py' first to create a backup.")
        sys.exit(1)
    
    # Check for latest backup
    latest_file = Path(".pytorch_backups") / "pytorch_backup_latest.json"
    
    if len(sys.argv) > 1:
        # Backup file specified
        backup_file = Path(sys.argv[1])
        if not backup_file.exists():
            print(f"ERROR: Backup file not found: {backup_file}")
            sys.exit(1)
    elif latest_file.exists():
        # Use latest
        backup_file = latest_file
        print("Using latest backup...")
    else:
        # Show list and ask
        print("Available backups:")
        for i, backup in enumerate(backups, 1):
            data = load_backup(backup)
            if data:
                info = data['torch_info']
                print(f"\n{i}. {backup.name}")
                print(f"   Date: {data['datetime']}")
                print(f"   PyTorch: {info['torch_version']}")
                print(f"   GPU: {info['gpu_name'] or 'CPU'}")
        
        print("\nEnter backup number to restore (or press Enter for latest):")
        choice = input("> ").strip()
        
        if choice:
            try:
                idx = int(choice) - 1
                backup_file = backups[idx]
            except (ValueError, IndexError):
                print("ERROR: Invalid choice")
                sys.exit(1)
        else:
            backup_file = backups[0]  # Latest
    
    print(f"\nRestoring from: {backup_file}")
    backup_data = load_backup(backup_file)
    
    if not backup_data:
        sys.exit(1)
    
    info = backup_data['torch_info']
    print("\nWill restore:")
    print(f"  PyTorch:     {info['torch_version']}")
    print(f"  TorchVision: {info['torchvision_version']}")
    print(f"  TorchAudio:  {info['torchaudio_version']}")
    print(f"  GPU:         {info['gpu_name'] or 'CPU'}")
    
    confirm = input("\nProceed with restore? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    restore_pytorch(backup_data)
    
    print("\n" + "=" * 60)
    print("Restore complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

