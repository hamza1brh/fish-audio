#!/usr/bin/env python3
"""
CLI inference speed test for fish-speech on SageMaker.
Tests voice generation speed without needing Streamlit UI.
"""

import subprocess
import time
from pathlib import Path

print("=" * 70)
print("Fish Speech Inference Speed Test (SageMaker)")
print("=" * 70)
print()

# Check PyTorch setup
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute: sm_{cap[0]}{cap[1]}")
print()

PROJECT_ROOT = Path.cwd()
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"

# Test text
TEST_TEXT = "Hello, this is a test of voice generation speed on AWS SageMaker using a Tesla T4 GPU."

print("=" * 70)
print("Test Configuration")
print("=" * 70)
print(f"Text: {TEST_TEXT}")
print(f"Reference: NeymarVO.mp3")
print()

# Check if model exists
if not (CHECKPOINTS_DIR / "model.pth").exists():
    print("Downloading model...")
    from huggingface_hub import snapshot_download
    snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(CHECKPOINTS_DIR))
    print("✅ Model downloaded")
    print()

# Run 3 test generations
print("=" * 70)
print("Running 3 test generations...")
print("=" * 70)
print()

times = []

for i in range(1, 4):
    print(f"Test {i}/3:")
    
    # Step 1: Extract VQ tokens
    print("  1. Extracting VQ tokens...")
    start = time.time()
    
    vq_cmd = [
        "python", "-m", "tools.vqgan.extract_vq",
        str(CHECKPOINTS_DIR / "codec.pth"),
        "NeymarVO.mp3",
        f"--output-path", f"test_vq_{i}.npy"
    ]
    
    result = subprocess.run(vq_cmd, capture_output=True, text=True)
    vq_time = time.time() - start
    
    if result.returncode != 0:
        print(f"     ❌ Failed: {result.stderr}")
        continue
    else:
        print(f"     ✅ Done in {vq_time:.2f}s")
    
    # Step 2: Generate semantic tokens
    print("  2. Generating semantic tokens...")
    start = time.time()
    
    semantic_cmd = [
        "python", "-m", "tools.llama.generate",
        "--text", TEST_TEXT,
        "--prompt-text", "",
        "--prompt-tokens", f"test_vq_{i}.npy",
        "--checkpoint-path", str(CHECKPOINTS_DIR / "model.pth"),
        "--num-samples", "1",
        "--compile",
        "--max-new-tokens", "1024",
        "--top-p", "0.7",
        "--temperature", "0.7",
        "--iterative-prompt",
        "--chunk-length", "200"
    ]
    
    result = subprocess.run(semantic_cmd, capture_output=True, text=True)
    semantic_time = time.time() - start
    
    if result.returncode != 0:
        print(f"     ❌ Failed: {result.stderr}")
        continue
    else:
        print(f"     ✅ Done in {semantic_time:.2f}s")
    
    # Step 3: Decode to audio
    print("  3. Decoding to audio...")
    start = time.time()
    
    decode_cmd = [
        "python", "-m", "tools.vqgan.inference",
        "--input-path", "codes_0.npy",
        "--checkpoint-path", str(CHECKPOINTS_DIR / "codec.pth"),
        "--output-path", f"test_output_{i}.wav"
    ]
    
    result = subprocess.run(decode_cmd, capture_output=True, text=True)
    decode_time = time.time() - start
    
    if result.returncode != 0:
        print(f"     ❌ Failed: {result.stderr}")
        continue
    else:
        print(f"     ✅ Done in {decode_time:.2f}s")
    
    total_time = vq_time + semantic_time + decode_time
    times.append(total_time)
    
    print(f"  Total: {total_time:.2f}s")
    print()

# Clean up temp files
print("Cleaning up...")
for i in range(1, 4):
    Path(f"test_vq_{i}.npy").unlink(missing_ok=True)
    Path(f"test_output_{i}.wav").unlink(missing_ok=True)
Path("codes_0.npy").unlink(missing_ok=True)

# Results
print()
print("=" * 70)
print("Results")
print("=" * 70)
print()

if times:
    avg_time = sum(times) / len(times)
    print(f"Completed: {len(times)}/3 generations")
    print(f"Times: {', '.join(f'{t:.2f}s' for t in times)}")
    print(f"Average: {avg_time:.2f}s per generation")
    print()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("Reference (expected times):")
    print("  - RTX 5070 Ti (Nightly): ~2-3s")
    print("  - Tesla T4 (Stable): ~3-5s")
    print("  - Tesla V100 (Stable): ~2-3s")
    print("  - Tesla A100 (Stable): ~1-2s")
else:
    print("❌ No successful generations")

print()
print("=" * 70)
print("✅ Test Complete")
print("=" * 70)

