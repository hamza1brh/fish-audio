#!/usr/bin/env python3
"""
Safe PyTorch Setup Test

This script:
1. Backs up current PyTorch
2. Tests the new setup
3. Verifies it works
4. Shows rollback instructions

Usage:
    python test_pytorch_setup.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    
    return result.returncode == 0

def main():
    print("=" * 60)
    print("Safe PyTorch Setup Test")
    print("=" * 60)
    
    # Step 1: Backup current setup
    print("\nStep 1: Backing up current PyTorch installation...")
    if not run_command(f"{sys.executable} backup_pytorch.py", "Backup Current PyTorch"):
        print("\nERROR: Backup failed!")
        sys.exit(1)
    
    print("\nBackup complete! You can restore anytime with:")
    print("  python restore_pytorch.py")
    
    # Step 2: Test new setup
    print("\n" + "=" * 60)
    print("Step 2: Testing New PyTorch Setup")
    print("=" * 60)
    
    confirm = input("\nProceed with testing new setup? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled. Your current PyTorch is unchanged.")
        sys.exit(0)
    
    print("\nRunning setup_pytorch.py...")
    if not run_command(f"{sys.executable} setup_pytorch.py", "New PyTorch Setup"):
        print("\nERROR: New setup failed!")
        print("\nRestoring backup...")
        run_command(f"{sys.executable} restore_pytorch.py", "Restore Backup")
        sys.exit(1)
    
    # Step 3: Test that it works
    print("\n" + "=" * 60)
    print("Step 3: Testing PyTorch Functionality")
    print("=" * 60)
    
    test_code = """
import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Cap: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    
    # Test GPU operation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("GPU Test: PASSED")
    except Exception as e:
        print(f"GPU Test: FAILED - {e}")
        sys.exit(1)
else:
    print("CPU Mode")
    # Test CPU operation
    try:
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print("CPU Test: PASSED")
    except Exception as e:
        print(f"CPU Test: FAILED - {e}")
        sys.exit(1)

print("All tests PASSED")
"""
    
    if not run_command(f'{sys.executable} -c "{test_code}"', "PyTorch Functionality Test"):
        print("\nWARNING: PyTorch test failed!")
        print("\nRestore backup? (y/N): ", end='')
        restore = input().strip().lower()
        
        if restore == 'y':
            print("\nRestoring backup...")
            run_command(f"{sys.executable} restore_pytorch.py", "Restore Backup")
        else:
            print("\nBackup preserved. Restore anytime with:")
            print("  python restore_pytorch.py")
        sys.exit(1)
    
    # Step 4: Test with Streamlit app
    print("\n" + "=" * 60)
    print("Step 4: Testing with Neymar Voice App")
    print("=" * 60)
    
    print("\nChecking if fish-speech imports work...")
    test_import = """
import torch
print("torch OK")
import fish_speech
print("fish_speech OK")
import streamlit
print("streamlit OK")
print("All imports successful!")
"""
    
    if not run_command(f'{sys.executable} -c "{test_import}"', "Import Test"):
        print("\nWARNING: Import test failed!")
        print("\nThis might be normal if fish-speech needs to be reinstalled.")
        print("Try: poetry install")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\n✅ New PyTorch setup appears to be working!")
    
    print("\nWhat's installed:")
    run_command(
        f'{sys.executable} -c "import torch; print(f\'PyTorch: {{torch.__version__}}\'); print(f\'Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}}\')"',
        "Current PyTorch"
    )
    
    print("\nNext steps:")
    print("  1. Test the voice app: streamlit run neymar_voice_app.py")
    print("  2. If it works: You're all set!")
    print("  3. If it doesn't: python restore_pytorch.py")
    
    print("\nBackups are stored in: .pytorch_backups/")
    print("Restore anytime with: python restore_pytorch.py")

if __name__ == "__main__":
    main()



