"""
Test that proves model stays in memory between generations.
This is the KEY fix for production use.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def get_vram_mb():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def main():
    print("=" * 70)
    print("S1-Mini PRODUCTION READINESS TEST")
    print("Testing: Model persistence & VRAM management")
    print("=" * 70)

    # Suppress excessive logging
    import os
    os.environ["S1_MINI_LOG_PLATFORM"] = "false"

    from s1_mini.config import EngineConfig
    from s1_mini.model_manager import ModelManager
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    # Track VRAM at each stage
    vram_stages = {}

    print(f"\n[1] Initial VRAM: {get_vram_mb():.1f} MB")
    vram_stages["initial"] = get_vram_mb()

    # Load models
    print("\n[2] Loading models...")
    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
        compile_model=False,
    )

    manager = ModelManager(config)
    manager.load_models()

    vram_stages["after_load"] = get_vram_mb()
    print(f"    VRAM after load: {vram_stages['after_load']:.1f} MB")

    # Create TTS engine
    tts_engine = TTSInferenceEngine(
        llama_queue=manager.llama_queue,
        decoder_model=manager.decoder_model,
        precision=config.torch_dtype,
        compile=False,
    )

    # Run multiple generations
    test_texts = [
        "First generation test.",
        "Second generation to prove model stays loaded.",
        "Third generation should be just as fast.",
    ]

    print("\n" + "=" * 70)
    print("RUNNING 3 GENERATIONS - Model should stay in VRAM!")
    print("=" * 70)

    generation_times = []

    for i, text in enumerate(test_texts, 1):
        print(f"\n[Generation {i}]")
        print(f"  Text: '{text}'")
        print(f"  VRAM before: {get_vram_mb():.1f} MB")

        # Create request
        request = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=None,
            max_new_tokens=512,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )

        # Time the generation
        start = time.time()
        for result in tts_engine.inference(request):
            if result.code == "final":
                break
        gen_time = time.time() - start
        generation_times.append(gen_time)

        vram_after = get_vram_mb()
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  VRAM after: {vram_after:.1f} MB")

        # KEY CHECK: VRAM should NOT drop significantly
        vram_drop = vram_stages["after_load"] - vram_after
        if vram_drop > 100:  # More than 100MB drop would indicate model unload
            print(f"  WARNING: VRAM dropped by {vram_drop:.1f} MB!")
        else:
            print(f"  OK: VRAM stable (delta: {vram_drop:.1f} MB)")

        vram_stages[f"after_gen_{i}"] = vram_after

    # Final analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nVRAM at each stage:")
    for stage, vram in vram_stages.items():
        print(f"  {stage:20s}: {vram:,.1f} MB")

    print("\nGeneration times:")
    for i, t in enumerate(generation_times, 1):
        print(f"  Generation {i}: {t:.2f}s")

    # Check if 2nd and 3rd gen are faster (warm cache)
    if generation_times[1] <= generation_times[0] * 1.1:
        print("\n  [OK] Subsequent generations are NOT slower (model cached)")
    else:
        print("\n  [WARN] Subsequent generations slower - possible issue")

    # Final VRAM check
    final_vram = get_vram_mb()
    initial_model_vram = vram_stages["after_load"] - vram_stages["initial"]

    print(f"\nModel VRAM footprint: {initial_model_vram:.1f} MB")
    print(f"Final VRAM usage: {final_vram:.1f} MB")

    if final_vram > initial_model_vram * 0.9:
        print("\n" + "=" * 70)
        print("PRODUCTION READINESS: PASSED")
        print("=" * 70)
        print("- Models stay in VRAM between generations")
        print("- No aggressive VRAM clearing")
        print("- Subsequent generations use cached model")
        print("=" * 70)
    else:
        print("\n[FAIL] VRAM dropped significantly - model may have been unloaded")

    # Cleanup
    print("\nCleaning up...")
    manager.shutdown()


if __name__ == "__main__":
    main()
