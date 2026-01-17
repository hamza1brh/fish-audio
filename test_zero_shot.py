#!/usr/bin/env python3
"""
Test zero-shot voice cloning with S1-Mini.

Usage:
    python test_zero_shot.py
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from s1_mini import ProductionTTSEngine, EngineConfig


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================

    # Reference audio for voice cloning (use WAV for compatibility)
    REFERENCE_AUDIO_PATH = "NeymarVO.wav"

    # IMPORTANT: This must be the text spoken in the reference audio
    REFERENCE_TEXT = """Eles me chamam de famoso.
Mas meus fãs não são mais meus.
Algoritmos decidem quem me vê.
Agentes decidem quem lucra comigo.
As mídias e as plataformas possuem a minha voz.
Não você.
A fome é passageira.
O holofote de hoje é o silêncio de amanhã.
Mas a minha história merece mais do que uma manchete.
Meu espírito, meu amor, minha arte, podem viver além do jogo."""

    # Text to synthesize with the cloned voice (Portuguese to match the voice)
    TEXT_TO_SYNTHESIZE = "Olá! Este é um teste de clonagem de voz. A inteligência artificial está reproduzindo minha voz."

    # Output file
    OUTPUT_PATH = "output_cloned.wav"

    # ==========================================================================
    # Load reference audio
    # ==========================================================================

    print("=" * 60)
    print("S1-Mini Zero-Shot Voice Cloning Test")
    print("=" * 60)

    ref_path = Path(REFERENCE_AUDIO_PATH)
    if not ref_path.exists():
        print(f"ERROR: Reference audio not found: {ref_path}")
        return 1

    print(f"\nReference audio: {ref_path}")
    print(f"Reference text: '{REFERENCE_TEXT}'")
    print(f"Text to synthesize: '{TEXT_TO_SYNTHESIZE}'")

    # Read reference audio bytes
    with open(ref_path, "rb") as f:
        reference_audio_bytes = f.read()
    print(f"Reference audio size: {len(reference_audio_bytes):,} bytes")

    # ==========================================================================
    # Initialize engine
    # ==========================================================================

    print("\n" + "=" * 60)
    print("Initializing S1-Mini Engine...")
    print("=" * 60)

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
        compile_model=False,  # Disable compilation for faster startup on Windows
    )

    engine = ProductionTTSEngine(config)

    print("\nStarting engine (loading models)...")
    start_time = time.time()
    engine.start()
    load_time = time.time() - start_time
    print(f"Engine ready in {load_time:.1f}s")

    # ==========================================================================
    # Generate with voice cloning
    # ==========================================================================

    print("\n" + "=" * 60)
    print("Generating audio with zero-shot voice cloning...")
    print("=" * 60)

    start_time = time.time()

    response = engine.generate(
        text=TEXT_TO_SYNTHESIZE,
        reference_audio=reference_audio_bytes,
        reference_text=REFERENCE_TEXT,
        temperature=0.7,
        top_p=0.8,
        return_bytes=True,
    )

    generation_time = time.time() - start_time

    if not response.success:
        print(f"\nERROR: Generation failed: {response.error}")
        engine.stop()
        return 1

    # ==========================================================================
    # Save output
    # ==========================================================================

    print(f"\nGeneration completed in {generation_time:.2f}s")

    if response.metrics:
        print(f"Audio duration: {response.metrics.audio_duration_seconds:.2f}s")
        print(f"Real-time factor: {response.metrics.realtime_factor:.2f}x")

    # Save audio
    with open(OUTPUT_PATH, "wb") as f:
        f.write(response.audio_bytes)
    print(f"\nSaved output to: {OUTPUT_PATH}")

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    engine.stop()

    print("\n" + "=" * 60)
    print("SUCCESS! Listen to output_cloned.wav to verify voice cloning.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
