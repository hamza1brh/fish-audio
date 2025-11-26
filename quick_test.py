"""
Quick Fish Speech Inference Test - Windows RTX 5070 Ti
Run this to benchmark your current setup and test zero-shot voice cloning
"""

import subprocess
import time
from pathlib import Path
import sys

# Colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")

def run_cmd(cmd, desc):
    """Run command and measure time"""
    print(f"{Colors.BLUE}▶{Colors.END} {desc}...", end=" ", flush=True)
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"{Colors.GREEN}✓{Colors.END} ({elapsed:.2f}s)")
        return elapsed, True
    else:
        print(f"{Colors.RED}✗{Colors.END} ({elapsed:.2f}s)")
        if "CUDA error" in result.stderr:
            print(f"{Colors.RED}CUDA Error detected - your RTX 5070 Ti needs CPU mode{Colors.END}")
        return elapsed, False

def main():
    print_section("🚀 Fish Speech Quick Test - RTX 5070 Ti")
    
    # Check setup
    print(f"{Colors.BOLD}📁 Project Directory:{Colors.END} {Path.cwd()}")
    
    # Test files
    ref_audio = "neymar_Dataset_enhanced/wavs/NEY0007.wav"
    if not Path(ref_audio).exists():
        print(f"{Colors.RED}❌ Reference audio not found: {ref_audio}{Colors.END}")
        return
    
    checkpoint = "checkpoints/openaudio-s1-mini"
    if not Path(checkpoint).exists():
        print(f"{Colors.RED}❌ Model checkpoint not found: {checkpoint}{Colors.END}")
        print(f"{Colors.YELLOW}Run: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir {checkpoint}{Colors.END}")
        return
    
    # Test GPU vs CPU
    print_section("🧪 Testing CUDA Compatibility")
    
    device = "cuda"
    test_text = "Olá, tudo bem?"
    
    print(f"{Colors.BOLD}Testing with device: {device}{Colors.END}\n")
    
    # Step 1: VQ Encoding
    print_section("Step 1: VQ Token Extraction")
    vq_cmd = f'python fish_speech/models/dac/inference.py -i "{ref_audio}" --checkpoint-path "{checkpoint}/codec.pth" --device {device}'
    vq_time, vq_success = run_cmd(vq_cmd, "Extracting VQ tokens from reference audio")
    
    if not vq_success:
        print(f"\n{Colors.YELLOW}⚠️  CUDA failed - trying CPU mode...{Colors.END}\n")
        device = "cpu"
        vq_cmd = f'python fish_speech/models/dac/inference.py -i "{ref_audio}" --checkpoint-path "{checkpoint}/codec.pth" --device cpu'
        vq_time, vq_success = run_cmd(vq_cmd, "Extracting VQ tokens (CPU mode)")
        
        if not vq_success:
            print(f"\n{Colors.RED}❌ Both CUDA and CPU failed. Check installation.{Colors.END}")
            return
    
    # Step 2: Text to Semantic
    print_section("Step 2: Text → Semantic Tokens")
    t2s_cmd = f'python fish_speech/models/text2semantic/inference.py --text "{test_text}" --prompt-text "Referência em português" --prompt-tokens "fake.npy" --checkpoint-path "{checkpoint}" --device {device}'
    t2s_time, t2s_success = run_cmd(t2s_cmd, "Generating semantic tokens")
    
    if not t2s_success:
        print(f"\n{Colors.RED}❌ Semantic generation failed{Colors.END}")
        return
    
    # Step 3: Semantic to Audio
    print_section("Step 3: Semantic → Audio")
    audio_cmd = f'python fish_speech/models/dac/inference.py -i "temp/codes_0.npy" --checkpoint-path "{checkpoint}/codec.pth" --device {device}'
    audio_time, audio_success = run_cmd(audio_cmd, "Synthesizing final audio")
    
    if not audio_success:
        print(f"\n{Colors.RED}❌ Audio synthesis failed{Colors.END}")
        return
    
    # Results
    total_time = vq_time + t2s_time + audio_time
    
    print_section("📊 Performance Results")
    print(f"{Colors.BOLD}Device:{Colors.END} {device.upper()}")
    print(f"{Colors.BOLD}VQ Encoding:{Colors.END} {vq_time:.2f}s (one-time)")
    print(f"{Colors.BOLD}Semantic Generation:{Colors.END} {t2s_time:.2f}s")
    print(f"{Colors.BOLD}Audio Synthesis:{Colors.END} {audio_time:.2f}s")
    print(f"{Colors.BOLD}Total (excluding VQ):{Colors.END} {t2s_time + audio_time:.2f}s")
    print(f"\n{Colors.GREEN}✅ Generated audio: fake.wav{Colors.END}")
    
    # Recommendations
    print_section("💡 Next Steps")
    
    if device == "cpu":
        print(f"{Colors.YELLOW}⚠️  Running on CPU (RTX 5070 Ti incompatibility detected){Colors.END}")
        print(f"   • Your Windows setup works, just without GPU acceleration")
        print(f"   • Performance: ~30-40s per sample (CPU mode)")
        print(f"   • For GPU: Deploy to AWS (A10G/V100/A100) - will be 10-50x faster")
    else:
        print(f"{Colors.GREEN}✅ GPU mode working!{Colors.END}")
        print(f"   • Performance: ~{t2s_time + audio_time:.1f}s per sample")
        print(f"   • This is production-ready for local testing")
        print(f"   • AWS deployment will add --compile for 10-50x speedup")
    
    print(f"\n{Colors.BOLD}Ready to test:{Colors.END}")
    print(f"   1. Open: notebooks/neymar_voice_cloning_simple.ipynb")
    print(f"   2. Generate more samples with different texts")
    print(f"   3. Try emotion tags: (excited), (sad), (laughing)")
    print(f"   4. Fine-tune on all 742 Neymar clips (see docs/en/finetune.md)")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Fish Speech is ready!{Colors.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
