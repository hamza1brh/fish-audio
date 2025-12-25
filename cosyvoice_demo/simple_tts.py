"""Simple script to use Fun-CosyVoice3-0.5B-2512 model for text-to-speech."""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install dependencies:")
    print("  pip install huggingface_hub torchaudio")
    print("\nAnd install CosyVoice:")
    print("  git clone https://github.com/FunAudioLLM/CosyVoice.git")
    print("  cd CosyVoice")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def download_model(model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B") -> str:
    """Download the Fun-CosyVoice3-0.5B-2512 model if not already present."""
    if os.path.exists(model_dir):
        print(f"Model already exists at {model_dir}")
        return model_dir
    
    print(f"Downloading model to {model_dir}...")
    snapshot_download(
        'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
        local_dir=model_dir
    )
    print("Model download complete!")
    return model_dir


def find_prompt_audio() -> Optional[str]:
    """Try to find a reference audio file."""
    possible_paths = [
        './asset/zero_shot_prompt.wav',
        '../asset/zero_shot_prompt.wav',
        'zero_shot_prompt.wav',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Run simple TTS inference."""
    model_dir = download_model()
    
    print("Loading model...")
    cosyvoice = AutoModel(model_dir=model_dir)
    
    text = "Hello, this is a test of the CosyVoice text-to-speech model."
    prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    
    prompt_audio = find_prompt_audio()
    if not prompt_audio:
        print("Warning: Reference audio file not found.")
        print("Please provide a reference audio file or download one from:")
        print("  https://github.com/FunAudioLLM/CosyVoice")
        print("\nYou can also modify the script to use your own audio file.")
        return
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cosyvoice_output.wav"
    
    print(f"Generating speech for: {text}")
    print(f"Using reference audio: {prompt_audio}")
    print("This may take a moment...")
    
    for i, result in enumerate(cosyvoice.inference_zero_shot(
        text=text,
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        stream=False
    )):
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"Saved audio to {output_path}")
        break


if __name__ == "__main__":
    main()

