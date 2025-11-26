#!/usr/bin/env python3
"""
Fish Speech Zero-Shot Inference Script - Neymar Voice Cloning

Performs zero-shot voice cloning using OpenAudio S1-mini with Neymar's voice.
Supports multiple target texts in Portuguese and English.

Usage:
    python neymar_zero_shot_inference.py --target 0
    python neymar_zero_shot_inference.py --target 2 --compile --half
    python neymar_zero_shot_inference.py --list-targets
"""

import sys
import platform
import subprocess
import argparse
from pathlib import Path
import shutil

try:
    import torch
except ImportError:
    print("Error: PyTorch not found. Please install PyTorch first.")
    sys.exit(1)


# ============================================================
# TARGET TEXTS CONFIGURATION
# ============================================================

# Fish Speech Emotion Tags Reference:
# Basic: (angry) (sad) (excited) (surprised) (satisfied) (delighted) (scared) (worried) 
#        (upset) (nervous) (frustrated) (depressed) (empathetic) (embarrassed) (disgusted)
#        (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful)
# Advanced: (disdainful) (unhappy) (anxious) (hysterical) (indifferent) (impatient) (guilty)
#           (scornful) (panicked) (furious) (reluctant) (keen) (disapproving) (negative) (denying)
#           (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering)
#           (hesitating) (yielding) (painful) (awkward) (amused)
# Tones: (in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
# Effects: (laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting) (groaning)

TARGET_TEXTS = [
    # 🎬 DRAMATIC TRAILER NARRATION - Slow, serious, cinematic with natural pauses
    # Uses (soft tone) for slower, deliberate delivery + line breaks for pauses
    """(serious) (soft tone) Eles me chamam de famoso.

Mas meus fãs não são mais meus.

Algoritmos decidem quem me vê.

Agentes decidem quem lucra comigo.

As mídias e as plataformas possuem a minha voz.

Não você.

A fome é passageira.

O holofote de hoje é o silêncio de amanhã.

Mas a minha história merece mais do que uma manchete.

Meu espírito, meu amor, minha arte, podem viver além do jogo.""",
    
    # 🇧🇷 Portuguese (Brazil) - Emotional Interview about Career Journey
    """(sincere) Olá pessoal, muito obrigado por estarem aqui comigo hoje. (sighing) Sabe, quando eu olho para trás e penso em toda a minha jornada... (moved) é impossível não me emocionar. (excited) Eu vim de uma família humilde, de Santos, e desde pequeno eu sonhava em ser jogador de futebol. (proud) Meu pai sempre acreditou em mim, mesmo quando ninguém mais acreditava. (grateful) Hoje eu posso dizer que tudo valeu a pena. (laughing) Ha, ha, ha! (joyful) Cada gol, cada título, cada momento de alegria... (soft tone) mas também aprendi muito com as derrotas e as lesões. (confident) O mais importante é nunca desistir dos seus sonhos!""",
    
    # 🇺🇸 English - Motivational Speech with Multiple Emotions
    """(confident) Hey everyone, thank you so much for being here today. (sincere) I want to share something really important with you. (serious) Life is not always easy, and the road to success is filled with obstacles. (empathetic) I know many of you are going through difficult times right now. (soft tone) But listen to me... (excited) every single challenge is an opportunity to grow stronger! (laughing) Ha, ha! (amused) You know what's funny? (serious) People used to say I was too small to play football. (proud) Look at me now! (shouting) Never let anyone tell you what you can or cannot do! (moved) Your dreams are valid, your efforts matter. (grateful) Thank you for believing in yourselves. (joyful) Now go out there and make it happen!""",
    
    # 🇪🇸 Spanish - Passionate Football Story
    """(excited) ¡Hola a todos mis amigos en España y Latinoamérica! (joyful) Es un placer enorme poder hablar con ustedes hoy. (sincere) Quiero contarles sobre mi amor por el fútbol. (moved) Desde que era un niño pequeño en Brasil, (soft tone) soñaba con vestir la camiseta de la selección brasileña. (proud) El fútbol me dio todo en la vida. (grateful) Me dio amigos increíbles, experiencias únicas, y la oportunidad de viajar por el mundo. (laughing) ¡Ja, ja, ja! (amused) ¿Saben qué es lo mejor? (excited) Que todavía sigo amando este deporte como el primer día. (serious) Hay momentos difíciles, claro. (sighing) Las lesiones, la presión, las críticas... (confident) Pero el amor por el juego siempre me levanta. (shouting) ¡Viva el fútbol! ¡Viva la pasión!""",
    
    # 🇫🇷 French - Reflective Interview about Life Lessons
    """(sincere) Bonjour à tous, merci beaucoup d'être là aujourd'hui. (soft tone) Je voulais partager avec vous quelques réflexions sur ma vie et ma carrière. (moved) Quand je repense à tout ce que j'ai vécu... (sighing) c'est vraiment incroyable. (proud) J'ai eu la chance de jouer dans les plus grands clubs du monde, comme le Paris Saint-Germain. (grateful) Paris restera toujours dans mon cœur. (laughing) Ha, ha, ha! (amused) Les Parisiens m'ont tellement bien accueilli. (serious) Mais vous savez, (empathetic) la vie d'un footballeur n'est pas toujours facile. (soft tone) Il y a des moments de doute, de tristesse. (confident) L'important, c'est de toujours se relever. (excited) Chaque jour est une nouvelle opportunité! (joyful) Et je suis reconnaissant pour chaque instant.""",
    
    # 🇩🇪 German - Determined Speech about Challenges
    """(serious) Hallo zusammen, danke dass ihr heute hier seid. (sincere) Ich möchte mit euch über etwas Wichtiges sprechen. (confident) In meiner Karriere habe ich viele Herausforderungen gemeistert. (sighing) Es gab Momente, in denen ich fast aufgeben wollte. (moved) Die Verletzungen waren besonders schwer. (soft tone) Aber wisst ihr was? (excited) Genau diese schweren Zeiten haben mich stärker gemacht! (laughing) Ha, ha, ha! (proud) Jetzt bin ich stolzer als je zuvor. (grateful) Ich bin dankbar für jede Erfahrung, gut oder schlecht. (empathetic) Wenn ihr gerade schwierige Zeiten durchmacht, (comforting) wisst, dass bessere Tage kommen werden. (shouting) Gebt niemals auf! (joyful) Das Leben ist schön, und eure Träume sind es wert, verfolgt zu werden!""",
]

# Reference audio configuration - Using NeymarVO trailer voiceover (29 seconds)
REFERENCE_AUDIO_FILE = "NeymarVO.mp3"
REFERENCE_TRANSCRIPT = """Eles me chamam de famoso, mas meus fãs não são mais meus. Algoritmos decidem quem me vê. Agentes decidem quem lucra comigo. As mídias e as plataformas possuem a minha voz, não você. A fome é passageira. O holofote de hoje é o silêncio de amanhã. Mas a minha história merece mais do que uma manchete. Meu espírito, meu amor, minha arte podem viver além do jogo."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_environment_info():
    """Print system and environment information."""
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]} (sm_{compute_capability[0]}{compute_capability[1]})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Check for RTX 50-series (Blackwell) support
        if compute_capability[0] >= 12:
            print("✓ Blackwell GPU detected (SM 12.0+)")
            if torch.__version__ < "2.7.0":
                print("⚠ WARNING: PyTorch < 2.7.0 may not fully support Blackwell GPUs.")
                print("  Recommended: PyTorch 2.7+ with CUDA 12.8 or 12.9")
    else:
        print("=" * 60)
        print("⚠ WARNING: CUDA NOT AVAILABLE")
        print("⚠ CPU inference will be EXTREMELY slow (10-100x slower)")
        print("⚠ For best performance, ensure CUDA-enabled PyTorch is installed")
        print("=" * 60)
    
    print("=" * 60)


def run_command(cmd, description=None, repo_root=None):
    """
    Run a command in a subprocess, echo it, and raise on error.
    
    Args:
        cmd: List of command arguments
        description: Optional description of what the command does
        repo_root: Working directory for the command
    """
    if description:
        print(f"\n{'='*60}")
        print(description)
        print(f"{'='*60}")
    
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    print()
    
    # Set environment variables to avoid torchcodec issues on Windows
    import os
    env = os.environ.copy()
    env["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
    env["TORCHAUDIO_BACKEND"] = "soundfile"
    
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=False,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(str(c) for c in cmd)}"
        )
    
    print(f"\n✓ Success\n")
    return result


def list_targets():
    """Display all available target texts."""
    print("\n" + "=" * 60)
    print("AVAILABLE TARGET TEXTS (6 Samples with Emotion Tags)")
    print("=" * 60)
    LANG_FLAGS = ["🎬 Trailer", "🇧🇷 PT", "🇺🇸 EN", "🇪🇸 ES", "🇫🇷 FR", "🇩🇪 DE"]
    for idx, text in enumerate(TARGET_TEXTS):
        lang = LANG_FLAGS[idx] if idx < len(LANG_FLAGS) else "??"
        # Show first 80 chars of text
        preview = text[:80].replace('\n', ' ') + "..."
        marker = "⭐" if idx == 0 else "  "
        print(f"{marker} [{idx}] {lang}: {preview}")
    print("=" * 60)


def verify_paths(repo_root, checkpoints_dir, codec_checkpoint, reference_audio):
    """Verify all required paths exist."""
    print("\n" + "=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)
    
    paths_ok = True
    
    if not checkpoints_dir.exists():
        print(f"✗ Checkpoints directory not found: {checkpoints_dir}")
        paths_ok = False
    else:
        print(f"✓ Checkpoints directory: {checkpoints_dir}")
    
    if not codec_checkpoint.exists():
        print(f"✗ Codec checkpoint not found: {codec_checkpoint}")
        paths_ok = False
    else:
        print(f"✓ Codec checkpoint: {codec_checkpoint}")
    
    if not reference_audio.exists():
        print(f"✗ Reference audio not found: {reference_audio}")
        paths_ok = False
    else:
        print(f"✓ Reference audio: {reference_audio}")
    
    print("=" * 60)
    
    if not paths_ok:
        raise FileNotFoundError("Required files are missing. Please check paths above.")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot voice cloning with Neymar's voice using Fish Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python neymar_zero_shot_inference.py --target 0
  python neymar_zero_shot_inference.py --target 2 --compile
  python neymar_zero_shot_inference.py --target 3 --compile --half
  python neymar_zero_shot_inference.py --list-targets
        """
    )
    
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=0,
        help=f'Target text index (0-{len(TARGET_TEXTS)-1}). Use --list-targets to see all options.'
    )
    
    parser.add_argument(
        '--list-targets', '-l',
        action='store_true',
        help='List all available target texts and exit'
    )
    
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Enable torch.compile for ~10x faster inference (requires Triton on Windows)'
    )
    
    parser.add_argument(
        '--half',
        action='store_true',
        help='Enable half precision (FP16) for faster inference on supported GPUs'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Custom output filename (default: neymar_zero_shot_N.wav)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        default=REFERENCE_AUDIO_FILE,
        help=f'Reference audio file (default: {REFERENCE_AUDIO_FILE})'
    )
    
    # === NEW: Inference Parameters for Natural Speech ===
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,  # Lower than default 0.8 for slower, more consistent pacing
        help='Sampling temperature (0.1-1.0). Lower = slower, more consistent. (default: 0.6)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.7,  # Slightly lower for more focused output
        help='Top-p nucleus sampling (0.1-1.0). Lower = more focused. (default: 0.7)'
    )
    
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.3,  # Higher to avoid rushed patterns
        help='Repetition penalty (0.9-2.0). Higher = avoids rushed repetition. (default: 1.3)'
    )
    
    parser.add_argument(
        '--chunk-length',
        type=int,
        default=250,
        help='Text chunk length (100-300). Higher = longer natural pacing. (default: 250)'
    )
    
    args = parser.parse_args()
    
    # List targets and exit if requested
    if args.list_targets:
        list_targets()
        return 0
    
    # Validate target index
    if not 0 <= args.target < len(TARGET_TEXTS):
        print(f"Error: Target index must be between 0 and {len(TARGET_TEXTS)-1}")
        list_targets()
        return 1
    
    # Get target text
    target_text = TARGET_TEXTS[args.target]
    
    # Repository paths
    REPO_ROOT = Path.cwd()
    CHECKPOINTS_DIR = REPO_ROOT / "checkpoints" / "openaudio-s1-mini"
    CODEC_CHECKPOINT = CHECKPOINTS_DIR / "codec.pth"
    REFERENCE_AUDIO_PATH = REPO_ROOT / args.reference
    OUTPUTS_DIR = REPO_ROOT / "outputs"
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Output path
    if args.output:
        OUTPUT_WAV_PATH = OUTPUTS_DIR / args.output
    else:
        OUTPUT_WAV_PATH = OUTPUTS_DIR / f"neymar_zero_shot_{args.target}.wav"
    
    # Intermediate files
    VQ_TOKENS_FILE = REPO_ROOT / "fake.npy"
    SEMANTIC_TOKENS_FILE = REPO_ROOT / "temp" / "codes_0.npy"
    
    # Print environment info
    print_environment_info()
    
    # Verify paths
    verify_paths(REPO_ROOT, CHECKPOINTS_DIR, CODEC_CHECKPOINT, REFERENCE_AUDIO_PATH)
    
    # Print configuration
    LANG_NAMES = ["🇧🇷 Portuguese", "🇺🇸 English", "🇪🇸 Spanish", "🇫🇷 French", "🇩🇪 German"]
    lang = LANG_NAMES[args.target] if args.target < len(LANG_NAMES) else "Unknown"
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Reference Audio: {args.reference}")
    print(f"Reference Transcript: '{REFERENCE_TRANSCRIPT}'")
    print(f"\nTarget Index: {args.target} ({lang})")
    print(f"Target Text: '{target_text}'")
    print(f"\nOutput File: {OUTPUT_WAV_PATH}")
    print(f"\nPerformance Settings:")
    print(f"  Compile Enabled: {args.compile}")
    print(f"  Half Precision: {args.half}")
    print(f"\nVoice Tuning Parameters (for natural, slower speech):")
    print(f"  Temperature: {args.temperature} (lower = slower/consistent)")
    print(f"  Top-P: {args.top_p} (lower = more focused)")
    print(f"  Repetition Penalty: {args.repetition_penalty} (higher = less rushed)")
    print(f"  Chunk Length: {args.chunk_length}")
    print("=" * 60)
    
    try:
        # ============================================================
        # STEP 1: Extract VQ tokens from reference audio
        # ============================================================
        cmd = [
            sys.executable,
            "fish_speech/models/dac/inference.py",
            "-i", str(REFERENCE_AUDIO_PATH),
            "--checkpoint-path", str(CODEC_CHECKPOINT)
        ]
        
        run_command(
            cmd,
            "STEP 1: Extracting VQ tokens from reference audio",
            REPO_ROOT
        )
        
        if not VQ_TOKENS_FILE.exists():
            raise FileNotFoundError(
                f"VQ tokens file not created at {VQ_TOKENS_FILE}. "
                "Codec inference may have failed."
            )
        
        print(f"✓ VQ tokens saved to: {VQ_TOKENS_FILE}")
        
        # ============================================================
        # STEP 2: Generate semantic tokens from text
        # ============================================================
        cmd = [
            sys.executable,
            "fish_speech/models/text2semantic/inference.py",
            "--text", target_text,
            "--prompt-text", REFERENCE_TRANSCRIPT,
            "--prompt-tokens", str(VQ_TOKENS_FILE),
            # Voice tuning parameters for natural, slower speech
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--repetition-penalty", str(args.repetition_penalty),
            "--chunk-length", str(args.chunk_length),
        ]
        
        if args.compile:
            cmd.append("--compile")
            print("⚡ Compile acceleration enabled")
        
        if args.half:
            cmd.append("--half")
            print("⚡ Half precision (FP16) enabled")
        
        run_command(
            cmd,
            "STEP 2: Generating semantic tokens from text",
            REPO_ROOT
        )
        
        if not SEMANTIC_TOKENS_FILE.exists():
            raise FileNotFoundError(
                f"Semantic tokens file not created at {SEMANTIC_TOKENS_FILE}. "
                "Text2semantic inference may have failed."
            )
        
        print(f"✓ Semantic tokens saved to: {SEMANTIC_TOKENS_FILE}")
        
        # ============================================================
        # STEP 3: Decode semantic tokens to waveform
        # ============================================================
        cmd = [
            sys.executable,
            "fish_speech/models/dac/inference.py",
            "-i", str(SEMANTIC_TOKENS_FILE),
            "--checkpoint-path", str(CODEC_CHECKPOINT)
        ]
        
        run_command(
            cmd,
            "STEP 3: Decoding semantic tokens to waveform",
            REPO_ROOT
        )
        
        # Move generated audio to output location
        generated_wav = REPO_ROOT / "fake.wav"
        
        if not generated_wav.exists():
            raise FileNotFoundError(
                f"Generated audio not found at {generated_wav}. "
                "Decoder inference may have failed."
            )
        
        shutil.move(str(generated_wav), str(OUTPUT_WAV_PATH))
        
        # ============================================================
        # SUCCESS
        # ============================================================
        print("\n" + "=" * 60)
        print("✓ SYNTHESIS COMPLETE - NEYMAR VOICE CLONE")
        print("=" * 60)
        print(f"Target Index: {args.target} ({lang})")
        print(f"Target Text: '{target_text}'")
        print(f"Output File: {OUTPUT_WAV_PATH}")
        print(f"File Size: {OUTPUT_WAV_PATH.stat().st_size / 1024:.2f} KB")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
