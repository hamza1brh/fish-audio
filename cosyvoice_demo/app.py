"""Streamlit UI for Fun-CosyVoice3-0.5B-2512 text-to-speech."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Force CPU - RTX 5070 Ti (sm_120) not supported even by PyTorch nightly
# Waiting for PyTorch with CUDA 12.8+ or 13.0+ support
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import torchaudio

# Add CosyVoice to path if it exists
cosyvoice_path = Path(__file__).parent / "CosyVoice"
if cosyvoice_path.exists():
    sys.path.insert(0, str(cosyvoice_path))
    # Add Matcha-TTS third_party dependency
    matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
    if matcha_path.exists():
        sys.path.insert(0, str(matcha_path))

# Workaround for torchvision compatibility issue with transformers
import warnings
warnings.filterwarnings('ignore')

try:
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import AutoModel
except (ImportError, RuntimeError) as e:
    st.error(f"Missing required package: {e}")
    st.info("""
    **Setup Instructions:**
    
    1. Install dependencies:
    ```bash
    pip install huggingface_hub torchaudio streamlit
    ```
    
    2. Install CosyVoice:
    ```bash
    git clone https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    pip install -r requirements.txt
    ```
    """)
    st.stop()


st.set_page_config(
    page_title="CosyVoice3 TTS",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ CosyVoice3 Text-to-Speech")
st.markdown("Generate high-quality speech using Fun-CosyVoice3-0.5B-2512 model")

st.warning("⚠️ Running on CPU (RTX 5070 Ti sm_120 not yet supported by PyTorch - needs CUDA 12.8+/13.0+). Generation: ~1-2 min/sentence.")

MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@st.cache_resource
def load_model():
    """Load the CosyVoice model."""
    if not os.path.exists(MODEL_DIR):
        st.info("📥 Downloading model files (~6GB). This is a one-time download and may take 5-15 minutes depending on your connection...")
        progress_bar = st.progress(0, text="Starting download...")
        try:
            snapshot_download(
                'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                local_dir=MODEL_DIR
            )
            progress_bar.progress(100, text="Download complete!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            raise
    
    with st.spinner("🔄 Loading model into memory... This may take a minute."):
        try:
            model = AutoModel(model_dir=MODEL_DIR)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            raise


def save_uploaded_audio(uploaded_file) -> Optional[str]:
    """Save uploaded audio file to temporary location."""
    if uploaded_file is None:
        return None
    
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def generate_speech(
    model: AutoModel,
    tts_text: str,
    prompt_text: str,
    prompt_audio_path: str,
    output_path: Path
) -> Optional[str]:
    """Generate speech using CosyVoice."""
    try:
        st.info("🎵 Generating speech on CPU... This may take 1-3 minutes depending on text length.")
        for i, result in enumerate(model.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_wav=prompt_audio_path,
            stream=False
        )):
            torchaudio.save(str(output_path), result['tts_speech'], model.sample_rate)
            return str(output_path)
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def main():
    """Main Streamlit app."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Model Info")
        st.info(f"**Model:** Fun-CosyVoice3-0.5B-2512\n\n**Status:** {'✅ Loaded' if os.path.exists(MODEL_DIR) else '⏳ Not downloaded'}")
        
        st.subheader("Reference Audio")
        st.markdown("Upload a reference audio file for voice cloning (WAV format recommended)")
        
        uploaded_audio = st.file_uploader(
            "Upload Reference Audio",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a reference audio file to clone the voice"
        )
        
        reference_audio_path = None
        if uploaded_audio:
            reference_audio_path = save_uploaded_audio(uploaded_audio)
            st.success(f"✅ Uploaded: {uploaded_audio.name}")
            st.audio(uploaded_audio, format='audio/wav')
        else:
            default_paths = [
                './asset/zero_shot_prompt.wav',
                '../asset/zero_shot_prompt.wav',
                'zero_shot_prompt.wav',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    reference_audio_path = path
                    st.info(f"Using default: {path}")
                    break
        
        st.subheader("Prompt Text")
        default_prompt = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
        prompt_text = st.text_area(
            "Prompt Text",
            value=default_prompt,
            help="Prompt text for the model"
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Text to Synthesize")
        text_input = st.text_area(
            "Enter your text",
            value="Hello, this is a test of the CosyVoice text-to-speech model.",
            height=150,
            help="Enter the text you want to convert to speech"
        )
        
        if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Please enter some text to synthesize.")
            elif not reference_audio_path:
                st.warning("Please upload a reference audio file or ensure default audio exists.")
            else:
                with st.spinner("Generating speech... This may take a moment."):
                    model = load_model()
                    timestamp = int(st.session_state.get('timestamp', 0))
                    output_path = OUTPUT_DIR / f"cosyvoice_output_{timestamp}.wav"
                    st.session_state['timestamp'] = timestamp + 1
                    
                    result_path = generate_speech(
                        model=model,
                        tts_text=text_input,
                        prompt_text=prompt_text,
                        prompt_audio_path=reference_audio_path,
                        output_path=output_path
                    )
                    
                    if result_path and os.path.exists(result_path):
                        st.success("✅ Speech generated successfully!")
                        st.audio(result_path, format='audio/wav')
                        
                        with open(result_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Audio",
                                data=f.read(),
                                file_name=f"cosyvoice_output_{timestamp}.wav",
                                mime="audio/wav"
                            )
    
    with col2:
        st.subheader("ℹ️ Information")
        st.markdown("""
        **Features:**
        - Zero-shot voice cloning
        - Multilingual support (9 languages)
        - High-quality speech synthesis
        
        **Supported Languages:**
        - Chinese, English, Japanese, Korean
        - German, Spanish, French, Italian, Russian
        
        **Tips:**
        - Use clear reference audio (3-10 seconds)
        - WAV format recommended for reference
        - Longer texts may take more time
        """)
        
        if 'timestamp' not in st.session_state:
            st.session_state['timestamp'] = 0


if __name__ == "__main__":
    main()

