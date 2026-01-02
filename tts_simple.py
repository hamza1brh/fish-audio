#!/usr/bin/env python3
"""
Simplified TTS CLI using fish-speech Python API directly.
Works on SageMaker without subprocess issues.
"""

import time
import sys
from pathlib import Path
from datetime import datetime
import json

try:
    import torch
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install torch soundfile numpy")
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
    max_tokens: int = 1024,
    chunk_length: int = 2000
):
    """Generate TTS using fish-speech Python API."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"tts_{timestamp}"
    output_path = output_dir / f"{output_name}.wav"
    
    print("=" * 70)
    print(f"Generating: {output_name}")
    print("=" * 70)
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Reference: {reference_audio}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()
    
    total_start = time.time()
    
    try:
        # Import fish-speech modules
        from fish_speech.models.dac.inference import load_model as load_codec
        from fish_speech.models.text2semantic.inference import load_model as load_t2s
        from fish_speech.text import encode_text
        
        # Step 1: Load codec model
        print("[1/4] Loading codec model...", end=" ", flush=True)
        step1_start = time.time()
        
        codec_model = load_codec(
            checkpoint_path=Path("checkpoints/openaudio-s1-mini/codec.pth"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        step1_time = time.time() - step1_start
        print(f"Done in {step1_time:.2f}s")
        
        # Step 2: Extract reference audio features
        print("[2/4] Extracting reference features...", end=" ", flush=True)
        step2_start = time.time()
        
        ref_audio, sr = sf.read(reference_audio)
        ref_audio = torch.from_numpy(ref_audio).float()
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if torch.cuda.is_available():
            ref_audio = ref_audio.cuda()
        
        with torch.no_grad():
            ref_codes = codec_model.encode(ref_audio.unsqueeze(0))
        
        step2_time = time.time() - step2_start
        print(f"Done in {step2_time:.2f}s")
        
        # Step 3: Load text2semantic model and generate
        print("[3/4] Generating semantic tokens...", end=" ", flush=True)
        step3_start = time.time()
        
        t2s_model = load_t2s(
            checkpoint_path=Path("checkpoints/openaudio-s1-mini/model.pth"),
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile_model=sys.platform == "linux"
        )
        
        # Generate codes
        encoded_text = encode_text(text)
        
        with torch.no_grad():
            generated_codes = t2s_model.generate(
                text=encoded_text,
                prompt_codes=ref_codes,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens
            )
        
        step3_time = time.time() - step3_start
        print(f"Done in {step3_time:.2f}s")
        
        # Step 4: Decode to audio
        print("[4/4] Decoding to audio...", end=" ", flush=True)
        step4_start = time.time()
        
        with torch.no_grad():
            audio = codec_model.decode(generated_codes)
        
        # Save audio
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        audio = audio.cpu().numpy()
        
        sf.write(str(output_path), audio.T, 24000)
        
        step4_time = time.time() - step4_start
        print(f"Done in {step4_time:.2f}s")
        
        total_time = time.time() - total_start
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
        print(f"  Codec Load:    {step1_time:.2f}s")
        print(f"  VQ Extract:    {step2_time:.2f}s")
        print(f"  Semantic Gen:  {step3_time:.2f}s")
        print(f"  Audio Decode:  {step4_time:.2f}s")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fish Speech Simple TTS CLI")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--reference", type=str, default="NeymarVO.mp3")
    parser.add_argument("--output-dir", type=str, default="tts_outputs")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--chunk-length", type=int, default=2000)
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
        
        while True:
            try:
                text = input("Enter text (or 'q' to quit): ").strip()
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                
                if not text:
                    print("Please enter some text.")
                    continue
                
                generate_tts(
                    text=text,
                    reference_audio=args.reference,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    chunk_length=args.chunk_length
                )
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        return
    
    # Single generation
    if not args.text:
        parser.print_help()
        print()
        print("Examples:")
        print("  python tts_simple.py --text 'Hello world'")
        print("  python tts_simple.py -i")
        return
    
    generate_tts(
        text=args.text,
        reference_audio=args.reference,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        chunk_length=args.chunk_length
    )


if __name__ == "__main__":
    main()

