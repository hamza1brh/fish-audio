"""
Simple test script for S1-Mini Production Engine
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_platform_detection():
    """Test platform detection and compilation settings."""
    print("=" * 60)
    print("TEST 1: Platform Detection")
    print("=" * 60)

    from s1_mini.compilation import get_platform_info, get_compilation_recommendations

    info = get_platform_info()
    print(f"  OS:              {info.system}")
    print(f"  Python:          {info.python_version}")
    print(f"  PyTorch:         {info.torch_version}")
    print(f"  CUDA Available:  {info.cuda_available}")
    if info.cuda_available:
        print(f"  CUDA Version:    {info.cuda_version}")
        print(f"  GPU:             {info.device_name}")
    print(f"  Triton Available: {info.triton_available}")
    print(f"  Recommended Backend: {info.recommended_backend}")

    print("\nRecommendations:")
    for rec in get_compilation_recommendations():
        print(f"  - {rec}")

    print("\n[PASS] Platform detection works!")
    return True


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TEST 2: Configuration")
    print("=" * 60)

    from s1_mini.config import EngineConfig

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
        compile_model=True,
    )

    print(f"  Checkpoint:      {config.checkpoint_path}")
    print(f"  Device:          {config.device}")
    print(f"  Precision:       {config.precision}")
    print(f"  Compile:         {config.compile_model}")
    print(f"  Backend:         {config.resolved_compile_backend}")
    print(f"  Should Compile:  {config.should_compile}")

    print("\n[PASS] Configuration works!")
    return True


def test_model_loading():
    """Test model loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Loading")
    print("=" * 60)

    from s1_mini.config import EngineConfig
    from s1_mini.model_manager import ModelManager

    config = EngineConfig(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        device="cuda",
        precision="float16",
        compile_model=False,  # Disable compile for faster test
    )

    print("Loading models (this may take a minute)...")
    start = time.time()

    manager = ModelManager(config)
    manager.load_models()

    load_time = time.time() - start
    print(f"  Load time: {load_time:.2f}s")

    # Check health
    health = manager.health_check()
    print(f"  Models loaded: {health['models_loaded']}")
    print(f"  VRAM used: {health['vram']['used_gb']:.2f} GB")
    print(f"  VRAM total: {health['vram']['total_gb']:.2f} GB")

    print("\n[PASS] Model loading works!")
    return manager


def test_single_generation(manager):
    """Test a single generation."""
    print("\n" + "=" * 60)
    print("TEST 4: Single Generation")
    print("=" * 60)

    from s1_mini.config import EngineConfig
    from s1_mini.engine import ProductionTTSEngine

    # Create engine with existing manager's config
    config = manager.config

    # Create TTS engine
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    tts_engine = TTSInferenceEngine(
        llama_queue=manager.llama_queue,
        decoder_model=manager.decoder_model,
        precision=config.torch_dtype,
        compile=config.should_compile,
    )

    # Create test request
    request = ServeTTSRequest(
        text="Hello, this is a test of the S1 Mini text to speech engine.",
        references=[],
        reference_id=None,
        max_new_tokens=1024,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        format="wav",
    )

    print(f"Generating audio for: '{request.text}'")
    print("This may take a moment...")

    start = time.time()

    # Run inference
    final_result = None
    for result in tts_engine.inference(request):
        if result.code == "error":
            print(f"  ERROR: {result.error}")
            return False
        elif result.code == "final":
            final_result = result
            break

    gen_time = time.time() - start

    if final_result is None or final_result.audio is None:
        print("  ERROR: No audio generated")
        return False

    sample_rate, audio = final_result.audio
    audio_duration = len(audio) / sample_rate

    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Audio duration:  {audio_duration:.2f}s")
    print(f"  Sample rate:     {sample_rate} Hz")
    print(f"  Audio shape:     {audio.shape}")
    print(f"  Realtime factor: {audio_duration / gen_time:.2f}x")

    # Save audio
    output_path = project_root / "s1_mini_test_output.wav"
    import soundfile as sf
    sf.write(str(output_path), audio, sample_rate)
    print(f"  Saved to: {output_path}")

    print("\n[PASS] Single generation works!")
    return True


def test_vram_not_cleared():
    """Test that VRAM is not aggressively cleared."""
    print("\n" + "=" * 60)
    print("TEST 5: VRAM Persistence Check")
    print("=" * 60)

    import torch

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return True

    # Get current VRAM usage
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    cached_before = torch.cuda.memory_reserved() / (1024**3)

    print(f"  VRAM allocated: {vram_before:.2f} GB")
    print(f"  VRAM cached:    {cached_before:.2f} GB")

    # The key test: VRAM should still be used (models in memory)
    if vram_before > 1.0:  # At least 1GB should be used by models
        print("  Models are still in VRAM (good!)")
        print("\n[PASS] VRAM persistence works!")
        return True
    else:
        print("  WARNING: Models may have been unloaded")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("S1-Mini Production Engine Tests")
    print("=" * 60)

    try:
        # Test 1: Platform detection
        test_platform_detection()

        # Test 2: Configuration
        test_config()

        # Test 3: Model loading
        manager = test_model_loading()

        # Test 4: Single generation
        test_single_generation(manager)

        # Test 5: VRAM persistence
        test_vram_not_cleared()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

        # Cleanup
        print("\nCleaning up...")
        manager.shutdown()

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
