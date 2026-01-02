#!/usr/bin/env python3
"""
Fish Speech CLI TTS Generator with Performance Metrics
Generates audio files with speed tracking and torch.compile optimization.
"""

import subprocess
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


class TTSGenerator:
    def __init__(self, checkpoint_dir: str = "checkpoints/openaudio-s1-mini"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path("tts_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.stats_file = self.output_dir / "generation_stats.json"
        self.stats = self._load_stats()
        
        # Check model
        if not (self.checkpoint_dir / "model.pth").exists():
            print("Model not found. Downloading...")
            self._download_model()
        
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"Compute Capability: sm_{cap[0]}{cap[1]}")
        print()
    
    def _download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(self.checkpoint_dir))
        print("Model downloaded successfully")
    
    def _load_stats(self):
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                return json.load(f)
        return {"generations": []}
    
    def _save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            data, samplerate = sf.read(str(audio_path))
            return len(data) / samplerate
        except Exception as e:
            print(f"Warning: Could not read audio duration: {e}")
            return 0.0
    
    def generate(
        self,
        text: str,
        reference_audio: str = "NeymarVO.mp3",
        reference_text: str = "",
        output_name: str = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        chunk_length: int = 2000,
        use_compile: bool = True
    ):
        """
        Generate TTS audio with performance tracking.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference voice audio
            reference_text: Transcript of reference (empty for zero-shot)
            output_name: Custom output filename (auto-generated if None)
            temperature: Sampling temperature (0.1-1.0)
            top_p: Top-p sampling (0.1-1.0)
            max_tokens: Max semantic tokens to generate
            chunk_length: Chunk length for generation
            use_compile: Use torch.compile (Linux only)
        """
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"tts_{timestamp}"
        
        output_path = self.output_dir / f"{output_name}.wav"
        vq_tokens_path = self.output_dir / f"{output_name}_vq.npy"
        semantic_tokens_path = Path("codes_0.npy")
        
        print("=" * 70)
        print(f"Generating: {output_name}")
        print("=" * 70)
        print(f"Text: {text}")
        print(f"Reference: {reference_audio}")
        print(f"Compile: {'Yes' if use_compile else 'No'}")
        print()
        
        total_start = time.time()
        
        # Step 1: Extract VQ tokens from reference
        print("[1/3] Extracting VQ tokens...", end=" ", flush=True)
        step1_start = time.time()
        
        # Use fish-speech inference API directly
        vq_cmd = [
            "python", "-m", "fish_speech.models.dac.inference",
            str(reference_audio),
            "--checkpoint-path", str(self.checkpoint_dir / "codec.pth"),
            "--output-path", str(vq_tokens_path.with_suffix(''))
        ]
        
        result = subprocess.run(vq_cmd, capture_output=True, text=True)
        step1_time = time.time() - step1_start
        
        if result.returncode != 0:
            print(f"FAILED")
            print(f"Error: {result.stderr}")
            return False
        
        print(f"Done in {step1_time:.2f}s")
        
        # Step 2: Generate semantic tokens
        print("[2/3] Generating semantic tokens...", end=" ", flush=True)
        step2_start = time.time()
        
        semantic_cmd = [
            "python", "-m", "tools.llama.generate",
            "--text", text,
            "--prompt-text", reference_text,
            "--prompt-tokens", str(vq_tokens_path),
            "--checkpoint-path", str(self.checkpoint_dir / "model.pth"),
            "--num-samples", "1",
            "--max-new-tokens", str(max_tokens),
            "--top-p", str(top_p),
            "--temperature", str(temperature),
            "--iterative-prompt",
            "--chunk-length", str(chunk_length)
        ]
        
        if use_compile and sys.platform == "linux":
            semantic_cmd.append("--compile")
        
        result = subprocess.run(semantic_cmd, capture_output=True, text=True)
        step2_time = time.time() - step2_start
        
        if result.returncode != 0:
            print(f"FAILED")
            print(f"Error: {result.stderr}")
            vq_tokens_path.unlink(missing_ok=True)
            return False
        
        print(f"Done in {step2_time:.2f}s")
        
        # Step 3: Decode to audio
        print("[3/3] Decoding to audio...", end=" ", flush=True)
        step3_start = time.time()
        
        decode_cmd = [
            "python", "-m", "tools.vqgan.inference",
            "--input-path", str(semantic_tokens_path),
            "--checkpoint-path", str(self.checkpoint_dir / "codec.pth"),
            "--output-path", str(output_path)
        ]
        
        result = subprocess.run(decode_cmd, capture_output=True, text=True)
        step3_time = time.time() - step3_start
        
        if result.returncode != 0:
            print(f"FAILED")
            print(f"Error: {result.stderr}")
            vq_tokens_path.unlink(missing_ok=True)
            semantic_tokens_path.unlink(missing_ok=True)
            return False
        
        print(f"Done in {step3_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # Get audio duration
        audio_duration = self._get_audio_duration(output_path)
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        # Cleanup temp files
        vq_tokens_path.unlink(missing_ok=True)
        semantic_tokens_path.unlink(missing_ok=True)
        
        # Display results
        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print(f"Output: {output_path}")
        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Generation Time: {total_time:.2f}s")
        print(f"Real-Time Factor: {rtf:.2f}x")
        print()
        print(f"  VQ Extraction: {step1_time:.2f}s")
        print(f"  Semantic Gen:  {step2_time:.2f}s")
        print(f"  Audio Decode:  {step3_time:.2f}s")
        print("=" * 70)
        
        # Save stats
        self.stats["generations"].append({
            "timestamp": datetime.now().isoformat(),
            "output": str(output_path),
            "text": text,
            "audio_duration": audio_duration,
            "generation_time": total_time,
            "rtf": rtf,
            "steps": {
                "vq_extraction": step1_time,
                "semantic_generation": step2_time,
                "audio_decode": step3_time
            },
            "config": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "chunk_length": chunk_length,
                "compile": use_compile
            }
        })
        self._save_stats()
        
        return True
    
    def show_stats(self):
        """Show statistics of all generations."""
        if not self.stats["generations"]:
            print("No generations yet.")
            return
        
        print()
        print("=" * 70)
        print("Generation Statistics")
        print("=" * 70)
        print(f"Total Generations: {len(self.stats['generations'])}")
        print()
        
        total_audio = sum(g["audio_duration"] for g in self.stats["generations"])
        total_time = sum(g["generation_time"] for g in self.stats["generations"])
        avg_rtf = sum(g["rtf"] for g in self.stats["generations"]) / len(self.stats["generations"])
        
        print(f"Total Audio Generated: {total_audio:.2f}s")
        print(f"Total Generation Time: {total_time:.2f}s")
        print(f"Average RTF: {avg_rtf:.2f}x")
        print()
        
        print("Recent Generations:")
        for i, gen in enumerate(self.stats["generations"][-5:], 1):
            print(f"  {i}. {gen['output']}")
            print(f"     Duration: {gen['audio_duration']:.2f}s | Time: {gen['generation_time']:.2f}s | RTF: {gen['rtf']:.2f}x")
        
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fish Speech CLI TTS Generator")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--reference", type=str, default="NeymarVO.mp3", help="Reference audio file")
    parser.add_argument("--reference-text", type=str, default="", help="Reference audio transcript")
    parser.add_argument("--output", type=str, help="Output filename (without extension)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.7, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument("--chunk-length", type=int, default=2000, help="Chunk length")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--stats", action="store_true", help="Show generation statistics")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    tts = TTSGenerator()
    
    # Show stats mode
    if args.stats:
        tts.show_stats()
        return
    
    # Interactive mode
    if args.interactive:
        print("Interactive TTS Generator (Ctrl+C to exit)")
        print()
        
        while True:
            try:
                text = input("Enter text to synthesize (or 'q' to quit): ").strip()
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                
                if not text:
                    print("Please enter some text.")
                    continue
                
                tts.generate(
                    text=text,
                    reference_audio=args.reference,
                    reference_text=args.reference_text,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    chunk_length=args.chunk_length,
                    use_compile=not args.no_compile
                )
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        tts.show_stats()
        return
    
    # Single generation mode
    if not args.text:
        parser.print_help()
        print()
        print("Examples:")
        print("  # Single generation")
        print("  python tts_cli.py --text 'Hello world'")
        print()
        print("  # Interactive mode")
        print("  python tts_cli.py -i")
        print()
        print("  # Show statistics")
        print("  python tts_cli.py --stats")
        print()
        print("  # Custom settings")
        print("  python tts_cli.py --text 'Test' --temperature 0.8 --chunk-length 2000")
        return
    
    tts.generate(
        text=args.text,
        reference_audio=args.reference,
        reference_text=args.reference_text,
        output_name=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        chunk_length=args.chunk_length,
        use_compile=not args.no_compile
    )


if __name__ == "__main__":
    main()

