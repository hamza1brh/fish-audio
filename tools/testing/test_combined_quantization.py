"""
Test combined LLaMA + DAC INT8 quantization for maximum VRAM savings.
Run: python tools/testing/test_combined_quantization.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

def test_combined_quantization():
    from fish_speech.models.text2semantic.inference import init_model
    from fish_speech.models.dac.inference import load_model as load_dac_model

    checkpoint_path = Path("checkpoints/openaudio-s1-mini")
    device = "cuda"
    precision = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    print("=" * 70)
    print("Testing Combined LLaMA + DAC INT8 Quantization")
    print("=" * 70)

    # Test configurations
    configs = [
        ("BF16 (baseline)", False, False, False),
        ("LLaMA INT8 + DAC BF16", False, True, False),
        ("LLaMA BF16 + DAC INT8", False, False, True),
        ("LLaMA INT8 + DAC INT8 (recommended)", False, True, True),
    ]

    results = []

    for name, runtime_int4, runtime_int8, dac_int8 in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        # Clear VRAM
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        vram_start = torch.cuda.memory_allocated() / 1e6

        # Load LLaMA model
        print("Loading LLaMA model...")
        model, decode_one_token = init_model(
            checkpoint_path=str(checkpoint_path),
            device=device,
            precision=precision,
            compile=False,
            runtime_int4=runtime_int4,
            runtime_int8=runtime_int8,
        )
        torch.cuda.synchronize()
        vram_after_llama = torch.cuda.memory_allocated() / 1e6
        llama_vram = vram_after_llama - vram_start
        print(f"  LLaMA VRAM: {llama_vram:.1f} MB")

        # Load DAC decoder
        print("Loading DAC decoder...")
        dac = load_dac_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(checkpoint_path / "codec.pth"),
            device=device,
            quantize_int8=dac_int8,
        )
        torch.cuda.synchronize()
        vram_after_dac = torch.cuda.memory_allocated() / 1e6
        dac_vram = vram_after_dac - vram_after_llama
        print(f"  DAC VRAM: {dac_vram:.1f} MB")

        total_vram = vram_after_dac - vram_start
        print(f"  Total VRAM: {total_vram:.1f} MB")

        results.append({
            "name": name,
            "llama": llama_vram,
            "dac": dac_vram,
            "total": total_vram,
        })

        # Cleanup
        del model, dac, decode_one_token
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<40} {'LLaMA':>10} {'DAC':>10} {'Total':>10}")
    print("-" * 70)

    baseline_total = results[0]["total"]
    for r in results:
        savings = baseline_total - r["total"]
        savings_pct = (savings / baseline_total) * 100 if baseline_total > 0 else 0
        print(f"{r['name']:<40} {r['llama']:>9.1f}M {r['dac']:>9.1f}M {r['total']:>9.1f}M")
        if savings > 0:
            print(f"{'':40} {'':>10} {'':>10} (saves {savings:.0f}M / {savings_pct:.0f}%)")

    print("=" * 70)
    print("\nNote: KV cache will add ~2 GB during generation at full sequence length.")
    print("=" * 70)


if __name__ == "__main__":
    test_combined_quantization()
