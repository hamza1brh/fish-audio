"""Test script to verify CosyVoice model works after download completes."""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add CosyVoice to path
cosyvoice_path = Path(__file__).parent / "CosyVoice"
sys.path.insert(0, str(cosyvoice_path))

try:
    from cosyvoice.cli.cosyvoice import AutoModel
    import torchaudio
    
    MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
    
    print("Loading model...")
    model = AutoModel(model_dir=MODEL_DIR)
    print("✅ Model loaded successfully!")
    
    # Test with provided reference audio
    ref_audio = "CosyVoice/asset/zero_shot_prompt.wav"
    if not Path(ref_audio).exists():
        print(f"❌ Reference audio not found: {ref_audio}")
        sys.exit(1)
    
    text = "Hello, this is a test of the CosyVoice model."
    prompt = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    
    print(f"Generating speech for: '{text}'")
    print("This may take a moment...")
    
    for i, result in enumerate(model.inference_zero_shot(
        text=text,
        prompt_text=prompt,
        prompt_audio=ref_audio,
        stream=False
    )):
        output_path = "test_output.wav"
        torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
        print(f"✅ Speech generated successfully: {output_path}")
        break
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)



