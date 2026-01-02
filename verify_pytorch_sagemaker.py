#!/usr/bin/env python3
"""
Quick verification script for SageMaker PyTorch setup.
Run this to confirm the auto-detection worked correctly.
"""

import sys
import subprocess

print("=" * 70)
print("SageMaker PyTorch Verification")
print("=" * 70)
print()

# Check PyTorch
try:
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: sm_{cap[0]}{cap[1]}")
        
        # Determine if correct version was installed
        is_nightly = 'dev' in torch.__version__
        is_stable = not is_nightly
        
        print()
        print("=" * 70)
        print("Analysis:")
        print("=" * 70)
        
        if cap[0] == 12 and cap[1] == 0:  # RTX 50-series
            if is_nightly:
                print("✅ CORRECT: RTX 50-series detected, PyTorch Nightly installed")
            else:
                print("❌ WRONG: RTX 50-series needs Nightly, but Stable installed")
        elif cap[0] >= 7:  # T4, V100, A100, etc.
            if is_stable:
                print("✅ CORRECT: Standard GPU detected, PyTorch Stable installed")
            else:
                print("⚠️  UNEXPECTED: Standard GPU has Nightly (works, but not optimal)")
        
        print()
        print(f"PyTorch Type: {'Nightly' if is_nightly else 'Stable'}")
        print(f"Expected: {'Nightly' if cap[0] == 12 else 'Stable'}")
        
        # Test GPU computation
        print()
        print("Testing GPU computation...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✅ GPU computation works!")
        
    else:
        print("⚠️  CUDA not available (CPU mode)")
        
    print()
    print("=" * 70)
    print("✅ Verification Complete")
    print("=" * 70)
    
except ImportError as e:
    print(f"❌ ERROR: Could not import PyTorch: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

