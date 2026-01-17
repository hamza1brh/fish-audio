"""
VRAM breakdown analysis for Fish-Speech models.
Shows exactly where GPU memory is being used.
"""

import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

CHECKPOINTS_DIR = Path("checkpoints")
MODELS = {
    "BF16": "openaudio-s1-mini",
    "INT8": "openaudio-s1-mini-int8-torchao-20260116_182651",
    "INT4": "openaudio-s1-mini-int4-g128-torchao-20260116_182842",
}


def get_vram_mb():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0


def clear_vram():
    """Clear VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    print("=" * 70)
    print("VRAM BREAKDOWN ANALYSIS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("No CUDA device available!")
        return

    # Get total VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_vram:.2f} GB")
    print()

    # Test each model
    for model_name, checkpoint_name in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {model_name} ({checkpoint_name})")
        print("=" * 70)

        clear_vram()
        baseline = get_vram_mb()
        print(f"Baseline VRAM: {baseline:.1f} MB")

        # Load LLaMA model
        print("\n1. Loading LLaMA model...")
        from fish_speech.models.text2semantic.llama import DualARTransformer

        checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
        llama_model = DualARTransformer.from_pretrained(
            str(checkpoint_path),
            load_weights=True,
        )
        llama_model = llama_model.to(device=device, dtype=torch.bfloat16)
        llama_model.eval()

        after_llama = get_vram_mb()
        llama_vram = after_llama - baseline
        print(f"   LLaMA VRAM: {llama_vram:.1f} MB ({llama_vram/1000:.2f} GB)")

        # Load DAC decoder
        print("\n2. Loading DAC decoder...")
        from fish_speech.models.dac.inference import load_model as load_decoder_model

        decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(CHECKPOINTS_DIR / "openaudio-s1-mini" / "codec.pth"),
            device=device,
        )

        after_dac = get_vram_mb()
        dac_vram = after_dac - after_llama
        print(f"   DAC VRAM: {dac_vram:.1f} MB ({dac_vram/1000:.2f} GB)")

        # Total
        total_used = after_dac - baseline
        print(f"\n3. TOTAL VRAM: {total_used:.1f} MB ({total_used/1000:.2f} GB)")
        print(f"   - LLaMA: {llama_vram:.1f} MB ({llama_vram/total_used*100:.1f}%)")
        print(f"   - DAC:   {dac_vram:.1f} MB ({dac_vram/total_used*100:.1f}%)")

        # Cleanup
        del llama_model
        del decoder
        clear_vram()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The DAC decoder (~1.8 GB on disk) is ALWAYS loaded in BF16.
This is a constant overhead regardless of LLaMA quantization.

To see real VRAM savings from quantization:
1. The LLaMA model portion should show differences
2. But DAC dominates total VRAM usage
3. Additionally, KV cache and activations grow during generation

Options to reduce total VRAM:
- Quantize the DAC decoder too (not currently implemented)
- Use a smaller DAC model
- Reduce max sequence length (smaller KV cache)
""")


if __name__ == "__main__":
    main()
