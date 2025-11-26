import os
import argparse

from huggingface_hub import hf_hub_download


# Download
def check_and_download_files(repo_id, file_list, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for file in file_list:
        file_path = os.path.join(local_dir, file)
        if not os.path.exists(file_path):
            print(f"{file} does not exist, downloading from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                resume_download=True,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{file} already exists, skipping.")


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

# S1-mini (0.5B params) - Fast, good quality - PUBLICLY AVAILABLE
repo_id_s1_mini = "fishaudio/openaudio-s1-mini"
local_dir_s1_mini = "./checkpoints/openaudio-s1-mini"
files_s1_mini = [
    ".gitattributes",
    "model.pth",
    "README.md",
    "special_tokens.json",
    "tokenizer.tiktoken",
    "config.json",
    "codec.pth",
]

# Fish-Speech 1.5 (previous generation, still good quality)
repo_id_fish_1_5 = "fishaudio/fish-speech-1.5"
local_dir_fish_1_5 = "./checkpoints/fish-speech-1.5"
files_fish_1_5 = [
    "model.pth",
    "README.md",
    "special_tokens.json", 
    "tokenizer.tiktoken",
    "config.json",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
]

# FFmpeg utilities
repo_id_ffmpeg = "fishaudio/fish-speech-1"
local_dir_ffmpeg = "./"
files_ffmpeg = [
    "ffmpeg.exe",
    "ffprobe.exe",
]

# ASR labeling tool
repo_id_asr = "SpicyqSama007/fish-speech-packed"
local_dir_asr = "./"
files_asr = [
    "asr-label-win-x64.exe",
]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Fish Speech models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/download_models.py              # Download S1-mini only (default)
  python tools/download_models.py --all        # Download all available models
  python tools/download_models.py --fish-1.5   # Download Fish-Speech 1.5

Available Models:
  S1-mini:       ~2GB  (0.5B params, OpenAudio latest, recommended)
  Fish-Speech-1.5: ~3GB  (older version, still good quality)

Note: The full OpenAudio S1 (4B params) is not publicly available yet.
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--s1-mini", action="store_true", help="Download S1-mini model (0.5B params, ~2GB)")
    parser.add_argument("--fish-1.5", action="store_true", dest="fish_1_5", help="Download Fish-Speech 1.5")
    parser.add_argument("--skip-tools", action="store_true", help="Skip downloading ffmpeg and ASR tools")
    
    args = parser.parse_args()
    
    # Default: download S1-mini if no model specified
    if not args.all and not args.s1_mini and not args.fish_1_5:
        args.s1_mini = True
    
    print("=" * 60)
    print("Fish Speech Model Downloader")
    print("=" * 60)
    
    # Download S1-mini
    if args.all or args.s1_mini:
        print("\n📦 Downloading OpenAudio S1-mini (0.5B params, ~2GB)...")
        print("-" * 40)
        check_and_download_files(repo_id_s1_mini, files_s1_mini, local_dir_s1_mini)
        print("✅ S1-mini download complete!")
    
    # Download Fish-Speech 1.5
    if args.all or args.fish_1_5:
        print("\n📦 Downloading Fish-Speech 1.5 (~3GB)...")
        print("-" * 40)
        check_and_download_files(repo_id_fish_1_5, files_fish_1_5, local_dir_fish_1_5)
        print("✅ Fish-Speech 1.5 download complete!")
    
    # Download tools
    if not args.skip_tools:
        print("\n🔧 Downloading utilities (ffmpeg, ASR tools)...")
        print("-" * 40)
        check_and_download_files(repo_id_ffmpeg, files_ffmpeg, local_dir_ffmpeg)
        check_and_download_files(repo_id_asr, files_asr, local_dir_asr)
        print("✅ Utilities download complete!")
    
    print("\n" + "=" * 60)
    print("✅ All downloads complete!")
    print("=" * 60)
    
    # Show what's available
    print("\n📋 Available models:")
    if os.path.exists(local_dir_s1_mini):
        print(f"   ✅ S1-mini:        {local_dir_s1_mini}")
    else:
        print(f"   ❌ S1-mini:        not downloaded")
    
    if os.path.exists(local_dir_fish_1_5):
        print(f"   ✅ Fish-Speech-1.5: {local_dir_fish_1_5}")
    else:
        print(f"   ❌ Fish-Speech-1.5: not downloaded (use --fish-1.5 or --all)")
    
    print("\nℹ️  Note: The full OpenAudio S1 (4B params) is not publicly available yet.")

