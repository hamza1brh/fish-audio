"""
Quick test for DAC INT8 quantization.
Run: python tools/testing/test_dac_quantization.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

def test_dac_quantization():
    from fish_speech.models.dac.inference import load_model

    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    device = "cuda"

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    # Clear VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("=" * 60)
    print("Testing DAC INT8 Quantization")
    print("=" * 60)

    # Test 1: Load without quantization
    print("\n1. Loading DAC decoder WITHOUT quantization...")
    torch.cuda.synchronize()
    vram_before = torch.cuda.memory_allocated() / 1e6

    decoder_bf16 = load_model(
        config_name="modded_dac_vq",
        checkpoint_path=checkpoint_path,
        device=device,
        quantize_int8=False,
    )

    torch.cuda.synchronize()
    vram_after_bf16 = torch.cuda.memory_allocated() / 1e6
    dac_vram_bf16 = vram_after_bf16 - vram_before
    print(f"   VRAM used by DAC (BF16): {dac_vram_bf16:.1f} MB")

    # Unload
    del decoder_bf16
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Test 2: Load with quantization
    print("\n2. Loading DAC decoder WITH INT8 quantization...")
    torch.cuda.synchronize()
    vram_before = torch.cuda.memory_allocated() / 1e6

    decoder_int8 = load_model(
        config_name="modded_dac_vq",
        checkpoint_path=checkpoint_path,
        device=device,
        quantize_int8=True,
    )

    torch.cuda.synchronize()
    vram_after_int8 = torch.cuda.memory_allocated() / 1e6
    dac_vram_int8 = vram_after_int8 - vram_before
    print(f"   VRAM used by DAC (INT8): {dac_vram_int8:.1f} MB")

    # Summary
    savings = dac_vram_bf16 - dac_vram_int8
    percent_savings = (savings / dac_vram_bf16) * 100 if dac_vram_bf16 > 0 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DAC BF16:     {dac_vram_bf16:.1f} MB")
    print(f"DAC INT8:     {dac_vram_int8:.1f} MB")
    print(f"Savings:      {savings:.1f} MB ({percent_savings:.1f}%)")
    print("=" * 60)

    # Cleanup
    del decoder_int8
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_dac_quantization()
