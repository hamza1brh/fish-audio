"""
Neymar Voice Cloning - Streamlit Chat Interface

A ChatGPT-like interface for generating voice samples with Neymar's voice.
Supports both Fish Speech OpenAudio S1 and S1-mini models.

Usage:
    streamlit run neymar_voice_app.py
"""

import streamlit as st
import subprocess
import sys
import time
import shutil
import base64
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Neymar Voice Cloning",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .audio-container {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .download-btn {
        margin-top: 0.5rem;
    }
    .status-generating {
        color: #ff9800;
        font-weight: bold;
    }
    .status-complete {
        color: #4caf50;
        font-weight: bold;
    }
    .emotion-tag {
        background-color: #e3f2fd;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }
    .model-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .ref-audio-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "streamlit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# History file for persistent chat history
HISTORY_FILE = OUTPUT_DIR / "chat_history.json"

# Available models configuration
AVAILABLE_MODELS = {
    "openaudio-s1-mini": {
        "name": "OpenAudio S1-mini",
        "description": "0.5B params, fast, great quality (recommended)",
        "path": PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini",
        "icon": "⚡"
    },
    "fish-speech-1.5": {
        "name": "Fish-Speech 1.5",
        "description": "Previous generation, still good quality",
        "path": PROJECT_ROOT / "checkpoints" / "fish-speech-1.5",
        "icon": "🐟"
    }
}
# Note: Full OpenAudio S1 (4B params) is not publicly available yet

# Reference audio presets
REFERENCE_PRESETS = {
    "NeymarVO.mp3": {
        "name": "🎬 Trailer Voiceover",
        "description": "29s dramatic trailer narration",
        "text": """Eles me chamam de famoso, mas meus fãs não são mais meus. Algoritmos decidem quem me vê. Agentes decidem quem lucra comigo. As mídias e as plataformas possuem a minha voz, não você. A fome é passageira. O holofote de hoje é o silêncio de amanhã. Mas a minha história merece mais do que uma manchete. Meu espírito, meu amor, minha arte podem viver além do jogo."""
    }
}

# Emotion tags reference
EMOTION_TAGS = {
    "Basic": ["(angry)", "(sad)", "(excited)", "(surprised)", "(satisfied)", "(delighted)", 
              "(scared)", "(worried)", "(upset)", "(nervous)", "(frustrated)", "(depressed)",
              "(empathetic)", "(embarrassed)", "(disgusted)", "(moved)", "(proud)", "(relaxed)",
              "(grateful)", "(confident)", "(interested)", "(curious)", "(confused)", "(joyful)"],
    "Advanced": ["(disdainful)", "(unhappy)", "(anxious)", "(hysterical)", "(indifferent)",
                 "(impatient)", "(guilty)", "(scornful)", "(panicked)", "(furious)", "(reluctant)",
                 "(keen)", "(disapproving)", "(negative)", "(denying)", "(astonished)", "(serious)",
                 "(sarcastic)", "(conciliative)", "(comforting)", "(sincere)", "(sneering)",
                 "(hesitating)", "(yielding)", "(painful)", "(awkward)", "(amused)"],
    "Tones": ["(in a hurry tone)", "(shouting)", "(screaming)", "(whispering)", "(soft tone)"],
    "Effects": ["(laughing)", "(chuckling)", "(sobbing)", "(crying loudly)", "(sighing)", 
                "(panting)", "(groaning)"]
}

# Default inference parameters
DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.7,
    "repetition_penalty": 1.3,
    "chunk_length": 250,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_device():
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except:
        pass
    return "cpu"

def load_chat_history():
    """Load chat history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate audio paths still exist
                for msg in data.get('messages', []):
                    if 'audio_path' in msg and not Path(msg['audio_path']).exists():
                        msg['audio_missing'] = True
                return data
        except Exception as e:
            print(f"Error loading history: {e}")
    return {'messages': [], 'generation_count': 0}

def save_chat_history(messages, generation_count):
    """Save chat history to JSON file."""
    try:
        data = {
            'messages': messages,
            'generation_count': generation_count,
            'last_updated': datetime.now().isoformat()
        }
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def get_available_models():
    """Check which models are actually downloaded."""
    available = {}
    for model_id, model_info in AVAILABLE_MODELS.items():
        codec_path = model_info["path"] / "codec.pth"
        if codec_path.exists():
            available[model_id] = model_info
    return available

def extract_vq_tokens(reference_audio: Path, model_path: Path) -> bool:
    """Extract VQ tokens from reference audio."""
    codec_path = model_path / "codec.pth"
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "fish_speech" / "models" / "dac" / "inference.py"),
        "-i", str(reference_audio),
        "--checkpoint-path", str(codec_path),
        "--device", get_device(),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return result.returncode == 0

def generate_audio(
    text: str, 
    output_name: str,
    model_path: Path,
    reference_audio: Path,
    reference_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_length: int = 250
) -> tuple[bool, str, float]:
    """
    Generate audio from text.
    
    Returns:
        (success, output_path, duration)
    """
    device = get_device()
    codec_path = model_path / "codec.pth"
    
    # Step 1: Extract VQ tokens from reference audio
    # Use a unique filename based on reference audio to cache tokens
    ref_hash = hash(str(reference_audio)) % 10000
    vq_tokens_file = PROJECT_ROOT / f"ref_tokens_{ref_hash}.npy"
    
    if not vq_tokens_file.exists():
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "fish_speech" / "models" / "dac" / "inference.py"),
            "-i", str(reference_audio),
            "--checkpoint-path", str(codec_path),
            "--device", device,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"❌ VQ extraction failed: {result.stderr}")
            print(f"Command: {' '.join(cmd)}")
            return False, "", 0
        
        # Rename fake.npy to our cached filename
        fake_npy = PROJECT_ROOT / "fake.npy"
        if fake_npy.exists():
            shutil.copy(str(fake_npy), str(vq_tokens_file))
        else:
            print(f"⚠️ Warning: fake.npy not found after VQ extraction")
    
    # Step 2: Generate semantic tokens
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "fish_speech" / "models" / "text2semantic" / "inference.py"),
        "--text", text,
        "--prompt-text", reference_text,
        "--prompt-tokens", str(vq_tokens_file),
        "--checkpoint-path", str(model_path),
        "--device", device,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
        "--chunk-length", str(chunk_length),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"❌ Semantic generation failed: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return False, "", 0
    
    # Step 3: Decode to audio
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "fish_speech" / "models" / "dac" / "inference.py"),
        "-i", str(PROJECT_ROOT / "temp" / "codes_0.npy"),
        "--checkpoint-path", str(codec_path),
        "--device", device,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"❌ DAC decoding failed: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return False, "", 0
    
    # Move output file
    output_path = OUTPUT_DIR / f"{output_name}.wav"
    fake_wav = PROJECT_ROOT / "fake.wav"
    if fake_wav.exists():
        shutil.move(str(fake_wav), str(output_path))
        
        # Get duration
        try:
            import soundfile as sf
            info = sf.info(str(output_path))
            return True, str(output_path), info.duration
        except:
            return True, str(output_path), 0
    
    return False, "", 0

# ============================================================
# SESSION STATE INITIALIZATION (with persistent history)
# ============================================================

if "history_loaded" not in st.session_state:
    # Load history from file on first run
    saved_history = load_chat_history()
    st.session_state.messages = saved_history.get('messages', [])
    st.session_state.generation_count = saved_history.get('generation_count', 0)
    st.session_state.history_loaded = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generation_count" not in st.session_state:
    st.session_state.generation_count = 0

if "inference_params" not in st.session_state:
    st.session_state.inference_params = DEFAULT_INFERENCE_PARAMS.copy()

if "custom_ref_text" not in st.session_state:
    st.session_state.custom_ref_text = ""

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎤 Neymar Voice Clone")
    st.markdown("---")
    
    # ===== MODEL SELECTION =====
    st.subheader("🤖 Model Selection")
    
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No models found! Please download a model first.")
        st.code("python tools/download_models.py")
        st.stop()
    
    # If only one model available, just show it (no dropdown needed)
    if len(available_models) == 1:
        selected_model_id = list(available_models.keys())[0]
        selected_model = available_models[selected_model_id]
        st.markdown(f"**{selected_model['icon']} {selected_model['name']}**")
        st.caption(f"📝 {selected_model['description']}")
    else:
        # Multiple models - show dropdown
        model_options = {
            model_id: f"{info['icon']} {info['name']}" 
            for model_id, info in available_models.items()
        }
        
        selected_model_id = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="S1 = better quality, slower. S1-mini = faster, good quality."
        )
        
        selected_model = available_models[selected_model_id]
        st.caption(f"📝 {selected_model['description']}")
    
    # Show download hint for missing models
    missing_models = set(AVAILABLE_MODELS.keys()) - set(available_models.keys())
    if missing_models:
        with st.expander("📥 Download more models"):
            st.markdown("**Available for download:**")
            for model_id in missing_models:
                info = AVAILABLE_MODELS[model_id]
                st.markdown(f"- **{info['icon']} {info['name']}**: {info['description']}")
            st.markdown("---")
            st.code("python tools/download_models.py --all")
            st.caption("ℹ️ Note: Full OpenAudio S1 (4B) is not publicly available yet")
    
    st.markdown("---")
    
    # ===== REFERENCE AUDIO SELECTION =====
    st.subheader("🎵 Reference Audio")
    
    # File uploader for reference audio
    uploaded_ref = st.file_uploader(
        "📤 Upload or Browse Reference Audio",
        type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        help="Upload an audio file with clear speech to clone. This is the voice that will be cloned.",
        key="ref_audio_uploader"
    )
    
    # Default reference audio option
    use_default = st.checkbox(
        "🎬 Use NeymarVO.mp3 (default trailer)",
        value=True if not uploaded_ref else False,
        help="Use the pre-loaded Neymar trailer voiceover as reference"
    )
    
    # Determine which reference audio to use
    if uploaded_ref and not use_default:
        # Save uploaded file temporarily
        upload_path = PROJECT_ROOT / f"uploaded_ref_{uploaded_ref.name}"
        with open(upload_path, "wb") as f:
            f.write(uploaded_ref.getbuffer())
        ref_audio_path = upload_path
        ref_info = {
            "name": f"📤 {uploaded_ref.name}",
            "description": f"Uploaded audio ({Path(uploaded_ref.name).suffix})",
            "text": ""
        }
        st.success(f"✅ Using: {uploaded_ref.name}")
    else:
        # Use default NeymarVO.mp3
        ref_audio_path = PROJECT_ROOT / "NeymarVO.mp3"
        ref_info = REFERENCE_PRESETS.get("NeymarVO.mp3", {
            "name": "🎬 Trailer Voiceover",
            "description": "29s dramatic trailer narration",
            "text": """Eles me chamam de famoso, mas meus fãs não são mais meus. Algoritmos decidem quem me vê. Agentes decidem quem lucra comigo. As mídias e as plataformas possuem a minha voz, não você. A fome é passageira. O holofote de hoje é o silêncio de amanhã. Mas a minha história merece mais do que uma manchete. Meu espírito, meu amor, minha arte podem viver além do jogo."""
        })
        st.info(f"📝 Using default: NeymarVO.mp3")
    
    # Show audio player for reference
    if ref_audio_path.exists():
        with st.expander("🎧 Listen to Reference", expanded=False):
            st.audio(str(ref_audio_path))
    
    # Reference transcript
    st.markdown("**📝 Reference Transcript:**")
    
    if ref_info.get("text"):
        # Pre-filled transcript for default
        reference_text = st.text_area(
            "Transcript (what's spoken in the audio)",
            value=ref_info["text"],
            height=100,
            key="ref_transcript",
            help="This must match what's spoken in the reference audio!"
        )
    else:
        # Custom audio needs user-provided transcript
        st.warning("⚠️ You MUST provide the transcript of what's spoken in this audio:")
        reference_text = st.text_area(
            "Transcript (required for uploaded audio)",
            value=st.session_state.custom_ref_text,
            height=100,
            key="ref_transcript_custom",
            placeholder="Type EXACTLY what is spoken in the reference audio..."
        )
        st.session_state.custom_ref_text = reference_text
    
    st.markdown("---")
    
    # ===== INFERENCE PARAMETERS =====
    st.subheader("🎛️ Voice Parameters")
    st.caption("Adjust for more natural speech")
    
    # Use keys for sliders to prevent lag
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.1, max_value=1.0, 
        value=st.session_state.inference_params.get("temperature", 0.6),
        step=0.1,
        help="Lower = slower, more consistent. Higher = more varied.",
        key="temp_slider"
    )
    
    top_p = st.slider(
        "🎯 Top-P (Focus)",
        min_value=0.1, max_value=1.0,
        value=st.session_state.inference_params.get("top_p", 0.7),
        step=0.1,
        help="Lower = more focused/deliberate. Higher = more diverse.",
        key="top_p_slider"
    )
    
    repetition_penalty = st.slider(
        "🔄 Repetition Penalty",
        min_value=0.9, max_value=2.0,
        value=st.session_state.inference_params.get("repetition_penalty", 1.3),
        step=0.1,
        help="Higher = avoids rushed repetition.",
        key="rep_pen_slider"
    )
    
    # Update session state (without triggering rerun)
    st.session_state.inference_params = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "chunk_length": DEFAULT_INFERENCE_PARAMS["chunk_length"],
    }
    
    # Preset buttons (instead of selectbox which causes reruns)
    st.caption("Quick Presets:")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🔄", help="Reset to defaults"):
            st.session_state.inference_params = DEFAULT_INFERENCE_PARAMS.copy()
            st.rerun()
    with preset_cols[1]:
        if st.button("🐢", help="Slow/Dramatic"):
            st.session_state.inference_params = {"temperature": 0.5, "top_p": 0.6, "repetition_penalty": 1.4, "chunk_length": 250}
            st.rerun()
    with preset_cols[2]:
        if st.button("⚖️", help="Natural/Balanced"):
            st.session_state.inference_params = {"temperature": 0.6, "top_p": 0.7, "repetition_penalty": 1.2, "chunk_length": 250}
            st.rerun()
    with preset_cols[3]:
        if st.button("🏃", help="Fast/Energetic"):
            st.session_state.inference_params = {"temperature": 0.8, "top_p": 0.8, "repetition_penalty": 1.1, "chunk_length": 250}
            st.rerun()
    
    st.markdown("---")
    
    # ===== EMOTION TAGS =====
    st.subheader("🎭 Emotion Tags")
    st.info("💡 Use `(soft tone)` for slower speech")
    
    for category, tags in EMOTION_TAGS.items():
        with st.expander(f"{category} ({len(tags)} tags)"):
            st.markdown(" ".join([f"`{tag}`" for tag in tags[:8]]))
            if len(tags) > 8:
                st.markdown(" ".join([f"`{tag}`" for tag in tags[8:]]))
    
    st.markdown("---")
    
    # ===== QUICK EXAMPLES =====
    st.subheader("💡 Quick Examples")
    if st.button("🇧🇷 Portuguese greeting"):
        st.session_state.example_text = "(excited) Olá pessoal! Como vocês estão? É muito bom estar aqui!"
    if st.button("🇺🇸 English motivation"):
        st.session_state.example_text = "(confident) Never give up on your dreams! Every challenge makes you stronger!"
    if st.button("🎬 Dramatic trailer"):
        st.session_state.example_text = "(serious) (soft tone) Eles me chamam de famoso. Mas meus fãs não são mais meus."
    if st.button("🎤 Natural/Slow"):
        st.session_state.example_text = "(soft tone) Obrigado pelo carinho de vocês... É muito especial pra mim."
    
    st.markdown("---")
    
    # ===== CLEAR CHAT =====
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.messages = []
            st.session_state.generation_count = 0
            save_chat_history([], 0)  # Clear saved history too
            st.rerun()
    with col_clear2:
        if st.button("💾 Save History"):
            save_chat_history(st.session_state.messages, st.session_state.generation_count)
            st.success("✅ Saved!")
    
    # Stats
    st.subheader("📈 Session Stats")
    st.markdown(f"""
    - **Generations:** {st.session_state.generation_count}
    - **History:** {len(st.session_state.messages)} messages
    - **Model:** {selected_model['name']}
    - **Device:** {get_device().upper()}
    """)
    
    # Show history file location
    with st.expander("📂 History Location"):
        st.code(str(HISTORY_FILE), language=None)
        st.caption("History is auto-saved after each generation")

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.title("🎤 Neymar Voice Cloning Chat")

# Show current config
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Model:** {selected_model['icon']} {selected_model['name']}")
with col2:
    st.markdown(f"**Reference:** {ref_info['name']}")
with col3:
    params = st.session_state.inference_params
    st.markdown(f"**Params:** T={params['temperature']}, P={params['top_p']}")

st.markdown("Type any text and I'll generate it in Neymar's voice. Use emotion tags for expressive speech!")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🎤"):
        st.markdown(message["content"])
        
        # If assistant message has audio, display it
        if message["role"] == "assistant" and "audio_path" in message:
            if Path(message["audio_path"]).exists():
                st.audio(message["audio_path"])
                
                # Download button
                col1, col2 = st.columns([1, 4])
                with col1:
                    with open(message["audio_path"], "rb") as f:
                        st.download_button(
                            label="📥 Download",
                            data=f,
                            file_name=Path(message["audio_path"]).name,
                            mime="audio/wav",
                            key=f"download_{message.get('timestamp', '')}_{message['audio_path']}"
                        )
                with col2:
                    # Display persistent metrics
                    metrics_text = []
                    if "duration" in message:
                        metrics_text.append(f"Duration: {message['duration']:.2f}s")
                    if "generation_time" in message:
                        metrics_text.append(f"Gen Time: {message['generation_time']:.1f}s")
                        if "duration" in message:
                            rtf = message['duration'] / message['generation_time']
                            metrics_text.append(f"RTF: {rtf:.2f}x")
                    if "model" in message:
                        metrics_text.append(f"Model: {message['model']}")
                    
                    if metrics_text:
                        st.caption(" | ".join(metrics_text))

# Check for example text
if "example_text" in st.session_state:
    example = st.session_state.example_text
    del st.session_state.example_text
    st.session_state.pending_input = example

# Chat input
prompt = st.chat_input("Enter text to generate in Neymar's voice...")

# Handle pending input from sidebar
if "pending_input" in st.session_state:
    prompt = st.session_state.pending_input
    del st.session_state.pending_input

if prompt:
    # Validate reference text
    if not reference_text.strip():
        st.error("❌ Please provide a transcript for the reference audio in the sidebar!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    save_chat_history(st.session_state.messages, st.session_state.generation_count)
    
    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="🎤"):
        status_placeholder = st.empty()
        status_placeholder.markdown(f"🔄 *Generating with {selected_model['name']}...*")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"neymar_chat_{timestamp}_{st.session_state.generation_count}"
        
        # Get inference parameters
        params = st.session_state.inference_params
        
        start_time = time.time()
        success, audio_path, duration = generate_audio(
            text=prompt,
            output_name=output_name,
            model_path=selected_model["path"],
            reference_audio=ref_audio_path,
            reference_text=reference_text,
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            chunk_length=params["chunk_length"]
        )
        generation_time = time.time() - start_time
        
        if success:
            st.session_state.generation_count += 1
            
            status_placeholder.markdown(f"✅ Generated in {generation_time:.1f}s with {selected_model['name']}")
            
            # Display audio
            st.audio(audio_path)
            
            # Download button
            col1, col2 = st.columns([1, 4])
            with col1:
                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="📥 Download",
                        data=f,
                        file_name=f"{output_name}.wav",
                        mime="audio/wav",
                        key=f"download_new_{timestamp}"
                    )
            with col2:
                st.caption(f"Duration: {duration:.2f}s | RTF: {duration/generation_time:.2f}x | Model: {selected_model['name']}")
            
            # Save to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ Generated in {generation_time:.1f}s",
                "audio_path": audio_path,
                "duration": duration,
                "generation_time": generation_time,
                "model": selected_model['name'],
                "timestamp": datetime.now().isoformat()
            })
            save_chat_history(st.session_state.messages, st.session_state.generation_count)
        else:
            status_placeholder.markdown("❌ Failed to generate audio. Check console for errors.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Failed to generate audio. Please try again.",
                "timestamp": datetime.now().isoformat()
            })
            save_chat_history(st.session_state.messages, st.session_state.generation_count)

# Footer
st.markdown("---")
st.caption(f"Powered by Fish Speech | Model: {selected_model['name']} | Device: {get_device().upper()}")
