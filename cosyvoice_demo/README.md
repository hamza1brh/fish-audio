# CosyVoice3 Demo

Simple script and Streamlit UI to use the Fun-CosyVoice3-0.5B-2512 model for text-to-speech.

## Important Notes

### GPU Compatibility

If you have an **RTX 5070 Ti** (or other Blackwell architecture GPU with sm_120), the current PyTorch 2.5.1 does not support it. The app is configured to run on **CPU mode** which is slower but works.

**Typical generation times:**
- CPU: 1-3 minutes per sentence
- GPU (when supported): 5-15 seconds per sentence

**To enable GPU in the future:**
1. Wait for PyTorch 2.6+ with Blackwell support
2. Or use PyTorch nightly builds: `pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126`
3. Remove the `CUDA_VISIBLE_DEVICES` line from `app.py`

## Setup

All setup commands have been run automatically. The following was done:

1. ✅ Cloned CosyVoice repository
2. ✅ Initialized Matcha-TTS submodule (required dependency)
3. ✅ Installed CosyVoice dependencies
4. ✅ Installed Python dependencies (huggingface_hub, torchaudio, streamlit)
5. ✅ Upgraded PyTorch to 2.5.1 (fixes compatibility issues)
6. ✅ Fixed torchvision compatibility
7. ✅ Downloaded model files (~6GB)

## Usage

### Streamlit UI (Recommended)

The Streamlit app is already running! Open your browser and go to:

**http://localhost:8501**

To start it manually:
```bash
cd cosyvoice_demo
streamlit run app.py
```

Or use the convenience scripts:
- Windows: `run_app.bat`
- Linux/Mac: `bash run_app.sh`

### Command Line Script

```bash
cd cosyvoice_demo
python simple_tts.py
```

The script will:
- Download the model automatically if not present
- Generate speech from the text
- Save the output to `outputs/cosyvoice_output.wav`

## Features

- **Zero-shot voice cloning**: Upload any reference audio to clone the voice
- **Multilingual support**: Supports 9 languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)
- **High-quality synthesis**: State-of-the-art speech quality
- **Easy to use**: Simple web interface with audio upload and playback

## Customization

### Streamlit App
- Upload reference audio through the web interface
- Enter custom text to synthesize
- Adjust prompt text in the sidebar
- Download generated audio files

### Command Line Script
Edit `simple_tts.py` to change:
- The text to synthesize
- The prompt text
- The reference audio file path
- The output location

## Model Download

The model will be automatically downloaded on first use to:
`pretrained_models/Fun-CosyVoice3-0.5B`

This is a one-time download (~2GB) and will be cached for future use.

