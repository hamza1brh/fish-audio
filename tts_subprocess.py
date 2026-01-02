#!/usr/bin/env python3
"""
Simple TTS CLI using the exact same subprocess approach as neymar_voice_app.py
Works reliably on SageMaker.
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
import hashlib

try:
    import torch
    import soundfile as sf
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        data, samplerate = sf.read(str(audio_path))
        return len(data) / samplerate
    except Exception:
        return 0.0


def generate_tts(
    text: str,
    reference_audio: str = "NeymarVO.mp3",
    output_dir: str = "tts_outputs",
    temperature: float = 0.7,
    top_p: float = 0.7,
    chunk_length: int = 2000,
    max_tokens: int = 1024
):
    """Generate TTS using subprocess (same as neymar_voice_app.py)."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"tts_{timestamp}"
    output_path = output_dir / f"{output_name}.wav"
    
    PROJECT_ROOT = Path.cwd()
    
    print("=" * 70)
    print(f"Generating: {output_name}")
    print("=" * 70)
    print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Reference: {reference_audio}")
    print()
    
    total_start = time.time()
    
    try:
        # Step 1: Extract VQ tokens from reference (with caching)
        print("[1/3] Extracting VQ tokens...", end=" ", flush=True)
        step1_start = time.time()
        
        # Use content hash for caching
        with open(reference_audio, "rb") as f:
            audio_content = f.read()
        audio_hash = hashlib.md5(audio_content).hexdigest()[:10]
        vq_tokens_file = output_dir / f"ref_tokens_{audio_hash}.npy"
        
        if not vq_tokens_file.exists():
            vq_cmd = [
                sys.executable,
                "-m", "tools.vqgan.inference",
                str(reference_audio),
                "--checkpoint-path", str(PROJECT_ROOT / "checkpoints/openaudio-s1-mini/codec.pth")
            ]
            
            result = subprocess.run(vq_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            
            if result.returncode != 0:
                print(f"FAILED\nError: {result.stderr}")
                return False
            
            # Move generated codes to cache
            codes_path = PROJECT_ROOT / f"{Path(reference_audio).stem}_codes.npy"
            if codes_path.exists():
                codes_path.rename(vq_tokens_file)
        
        step1_time = time.time() - step1_start
        print(f"Done in {step1_time:.2f}s")
        
        # Step 2: Generate semantic tokens
        print("[2/3] Generating semantic tokens...", end=" ", flush=True)
        step2_start = time.time()
        
        semantic_cmd = [
            sys.executable,
            "-m", "tools.llama.generate",
            "--text", text,
            "--prompt-text", "",
            "--prompt-tokens", str(vq_tokens_file),
            "--checkpoint-path", str(PROJECT_ROOT / "checkpoints/openaudio-s1-mini/model.pth"),
            "--num-samples", "1",
            "--max-new-tokens", str(max_tokens),
            "--top-p", str(top_p),
            "--temperature", str(temperature),
            "--iterative-prompt",
            "--chunk-length", str(chunk_length)
        ]
        
        if sys.platform == "linux":
            semantic_cmd.append("--compile")
        
        result = subprocess.run(semantic_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            print(f"FAILED\nError: {result.stderr}")
            return False
        
        step2_time = time.time() - step2_start
        print(f"Done in {step2_time:.2f}s")
        
        # Step 3: Decode to audio
        print("[3/3] Decoding to audio...", end=" ", flush=True)
        step3_start = time.time()
        
        codes_file = PROJECT_ROOT / "codes_0.npy"
        if not codes_file.exists():
            print(f"FAILED\nError: codes_0.npy not found")
            return False
        
        decode_cmd = [
            sys.executable,
            "-m", "tools.vqgan.inference",
            str(codes_file),
            "--checkpoint-path", str(PROJECT_ROOT / "checkpoints/openaudio-s1-mini/codec.pth"),
            "--output-path", str(output_path.with_suffix(''))
        ]
        
        result = subprocess.run(decode_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            print(f"FAILED\nError: {result.stderr}")
            codes_file.unlink(missing_ok=True)
            return False
        
        # Clean up temp files
        codes_file.unlink(missing_ok=True)
        
        step3_time = time.time() - step3_start
        print(f"Done in {step3_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # Check if output was created (might have .wav or no extension)
        actual_output = None
        for ext in ['', '.wav']:
            check_path = Path(str(output_path.with_suffix('')) + ext)
            if check_path.exists():
                actual_output = check_path
                if check_path != output_path:
                    check_path.rename(output_path)
                    actual_output = output_path
                break
        
        if not actual_output or not actual_output.exists():
            print(f"FAILED\nError: Output file not created")
            return False
        
        audio_duration = get_audio_duration(output_path)
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        # Results
        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print(f"Output: {output_path}")
        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Generation Time: {total_time:.2f}s")
        print(f"Real-Time Factor: {rtf:.2f}x")
        print()
        print(f"  VQ Extract:    {step1_time:.2f}s")
        print(f"  Semantic Gen:  {step2_time:.2f}s")
        print(f"  Audio Decode:  {step3_time:.2f}s")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fish Speech TTS CLI (Subprocess)")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--reference", type=str, default="NeymarVO.mp3")
    parser.add_argument("--output-dir", type=str, default="tts_outputs")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--chunk-length", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("-i", "--interactive", action="store_true")
    
    args = parser.parse_args()
    
    # Check GPU
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: sm_{cap[0]}{cap[1]}")
    print()
    
    # Interactive mode
    if args.interactive:
        print("Interactive TTS (Ctrl+C to exit)")
        print()
        
        success_count = 0
        total_time = 0
        total_audio = 0
        
        while True:
            try:
                text = input("Enter text (or 'q' to quit): ").strip()
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                
                if not text:
                    print("Please enter some text.")
                    continue
                
                result = generate_tts(
                    text=text,
                    reference_audio=args.reference,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    chunk_length=args.chunk_length,
                    max_tokens=args.max_tokens
                )
                
                if result:
                    success_count += 1
                
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        if success_count > 0:
            print()
            print(f"Generated {success_count} audio files in {args.output_dir}/")
        
        return
    
    # Single generation
    if not args.text:
        parser.print_help()
        print()
        print("Examples:")
        print("  python tts_subprocess.py --text 'Hello world'")
        print("  python tts_subprocess.py -i")
        return
    
    generate_tts(
        text=args.text,
        reference_audio=args.reference,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_length=args.chunk_length,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()

