"""
Streamlit UI for testing different quantization levels of Fish-Speech models.

Run with: streamlit run tools/testing/streamlit_quant_test.py

This allows you to:
- Select different quantization levels (BF16, INT8, INT4)
- Enter text and generate speech
- Zero-shot voice cloning with reference audio
- Compare audio quality across different quantization levels
"""

import io
import os
import sys
import time
import wave
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import streamlit as st
import torch
import numpy as np

# Page config
st.set_page_config(
    page_title="Fish-Speech Quantization Tester",
    page_icon="🐟",
    layout="wide",
)

# Constants
CHECKPOINTS_DIR = Path("checkpoints")
REFERENCES_DIR = Path("references")
# Format: (checkpoint_name, runtime_int4, runtime_int8)
AVAILABLE_MODELS = {
    "BF16 (Best Quality)": ("openaudio-s1-mini", False, False),
    "INT8 Runtime (Recommended)": ("openaudio-s1-mini", False, True),  # Good quality + VRAM savings
    "INT8 TorchAO": ("openaudio-s1-mini-int8-torchao-20260116_182651", False, False),
    "INT4 Runtime": ("openaudio-s1-mini", True, False),  # Max VRAM savings, lower quality
    "INT4 TorchAO (g128)": ("openaudio-s1-mini-int4-g128-torchao-20260116_182842", False, False),
}

SAMPLE_TEXTS = {
    "English": "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet.",
    "Chinese": "今天天气真好，阳光明媚，微风轻拂。我决定出去散步，享受这美好的一天。",
    "Japanese": "今日はとても良い天気です。公園を散歩して、桜の花を見ました。春は本当に美しい季節ですね。",
    "Korean": "안녕하세요! 오늘 날씨가 정말 좋네요. 좋은 하루 되세요.",
    "German": "Guten Tag! Das Wetter ist heute wunderbar. Ich hoffe, Sie haben einen schönen Tag.",
}


@st.cache_resource
def load_model_and_decoder(
    _checkpoint_path: str,
    _runtime_int4: bool = False,
    _runtime_int8: bool = False,
    _dac_int8: bool = False,
    device: str = "cuda"
):
    """Load model and decoder with caching."""
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_decoder_model

    precision = torch.bfloat16

    if _runtime_int8:
        st.info(f"Loading BF16 model from {_checkpoint_path} + applying runtime INT8...")
    elif _runtime_int4:
        st.info(f"Loading BF16 model from {_checkpoint_path} + applying runtime INT4...")
    else:
        st.info(f"Loading LLaMA model from {_checkpoint_path}...")

    # Launch LLaMA queue (waits for init internally)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=_checkpoint_path,
        device=device,
        precision=precision,
        compile=False,  # Disable compile for faster startup
        runtime_int4=_runtime_int4,
        runtime_int8=_runtime_int8,
    )

    if _dac_int8:
        st.info("Loading DAC decoder with INT8 quantization...")
    else:
        st.info("Loading DAC decoder...")

    # Load DAC decoder (always from original checkpoint)
    decoder = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(CHECKPOINTS_DIR / "openaudio-s1-mini" / "codec.pth"),
        device=device,
        quantize_int8=_dac_int8,
    )

    return llama_queue, decoder, precision


def generate_speech(
    llama_queue,
    decoder,
    precision,
    text: str,
    reference_audio: bytes | None = None,
    reference_text: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
):
    """Generate speech from text, optionally with voice cloning."""
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    # Create engine
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=precision,
        compile=False,
    )

    # Build references list for voice cloning
    references = []
    if reference_audio is not None and reference_text:
        references = [
            ServeReferenceAudio(audio=reference_audio, text=reference_text)
        ]

    # Create request
    req = ServeTTSRequest(
        text=text,
        references=references,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streaming=False,
    )

    # Generate
    start_time = time.time()
    audio_result = None
    error = None

    try:
        for result in engine.inference(req):
            if result.code == "final":
                audio_result = result.audio
            elif result.code == "error":
                error = str(result.error)
    except Exception as e:
        error = str(e)

    generation_time = time.time() - start_time

    return audio_result, generation_time, error


def get_available_references():
    """Get list of available pre-stored reference voices."""
    references = []
    if REFERENCES_DIR.exists():
        for ref_dir in REFERENCES_DIR.iterdir():
            if ref_dir.is_dir():
                # Check if it has audio files
                audio_files = list(ref_dir.glob("*.wav")) + list(ref_dir.glob("*.mp3")) + list(ref_dir.glob("*.flac"))
                if audio_files:
                    references.append(ref_dir.name)
    return sorted(references)


def main():
    st.title("🐟 Fish-Speech Quantization Tester")
    st.markdown("Compare audio quality across different quantization levels with optional voice cloning")

    # Initialize session state
    if "text_input" not in st.session_state:
        st.session_state.text_input = "Hello! This is a test of the Fish Speech text to speech system. How does it sound?"
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None
    if "generation_info" not in st.session_state:
        st.session_state.generation_info = None
    if "reference_text" not in st.session_state:
        st.session_state.reference_text = ""

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Settings")

        selected_model_name = st.selectbox(
            "Select Quantization",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
        )

        checkpoint_name, runtime_int4, runtime_int8 = AVAILABLE_MODELS[selected_model_name]
        checkpoint_path = str(CHECKPOINTS_DIR / checkpoint_name)

        # Check if model exists
        if not Path(checkpoint_path).exists():
            st.error(f"Model not found: {checkpoint_path}")
            st.stop()

        if runtime_int8:
            st.success(f"Model: {checkpoint_name} + Runtime INT8")
        elif runtime_int4:
            st.success(f"Model: {checkpoint_name} + Runtime INT4")
        else:
            st.success(f"Model: {checkpoint_name}")

        # DAC decoder quantization option
        st.header("DAC Decoder Settings")
        dac_int8 = st.checkbox(
            "DAC INT8 Quantization",
            value=True,
            help="Apply INT8 quantization to DAC decoder (~1.87 GB -> ~0.66 GB, saves ~1.2 GB)"
        )

        # Generation settings
        st.header("Generation Settings")
        max_new_tokens = st.slider("Max New Tokens", 256, 2048, 1024, 128)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.8, 0.05)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.5, 1.1, 0.05)

        # VRAM info
        if torch.cuda.is_available():
            st.header("GPU Info")
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.metric("VRAM Used", f"{vram_used:.2f} GB")
            st.metric("VRAM Total", f"{vram_total:.2f} GB")

    # Main content - Use tabs for organization
    tab_default, tab_clone = st.tabs(["Default Voice", "Voice Cloning (Zero-Shot)"])

    # Reference audio variables
    reference_audio_bytes = None
    reference_text_value = None

    with tab_default:
        st.header("Text Input")

        # Sample text buttons
        st.subheader("Quick Load Sample Text")
        sample_cols = st.columns(len(SAMPLE_TEXTS))
        for i, (lang, text) in enumerate(SAMPLE_TEXTS.items()):
            with sample_cols[i]:
                if st.button(lang, use_container_width=True, key=f"sample_{lang}"):
                    st.session_state.text_input = text

        # Text area
        text_input_default = st.text_area(
            "Enter text to synthesize",
            value=st.session_state.text_input,
            height=150,
            key="text_area_default",
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            generate_default = st.button(
                "🎵 Generate (Default Voice)",
                type="primary",
                use_container_width=True,
                key="gen_default"
            )

    with tab_clone:
        st.header("Zero-Shot Voice Cloning")
        st.markdown("""
        Upload a reference audio file and provide its exact transcription.
        The model will clone the voice characteristics and speak your target text.
        """)

        col_ref, col_target = st.columns(2)

        with col_ref:
            st.subheader("Reference Voice")

            # File uploader for reference audio
            uploaded_file = st.file_uploader(
                "Upload reference audio (WAV, MP3, FLAC)",
                type=["wav", "mp3", "flac", "ogg"],
                key="ref_audio_upload"
            )

            if uploaded_file is not None:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
                reference_audio_bytes = uploaded_file.getvalue()

            # Reference text (transcription)
            reference_text_input = st.text_area(
                "Reference audio transcription (MUST match exactly what is said)",
                value=st.session_state.reference_text,
                height=100,
                key="ref_text_area",
                placeholder="Enter the exact words spoken in the reference audio..."
            )
            st.session_state.reference_text = reference_text_input
            reference_text_value = reference_text_input

            # Show pre-stored references if available
            available_refs = get_available_references()
            if available_refs:
                st.markdown("---")
                st.subheader("Or use pre-stored reference")
                selected_ref = st.selectbox(
                    "Select reference voice",
                    options=["(None)"] + available_refs,
                    key="preset_ref"
                )
                if selected_ref != "(None)":
                    st.info(f"Using pre-stored reference: {selected_ref}")

        with col_target:
            st.subheader("Target Text")

            # Sample text buttons for cloning
            st.markdown("**Quick Load:**")
            clone_sample_cols = st.columns(3)
            for i, (lang, text) in enumerate(list(SAMPLE_TEXTS.items())[:3]):
                with clone_sample_cols[i]:
                    if st.button(lang, use_container_width=True, key=f"clone_sample_{lang}"):
                        st.session_state.text_input = text

            # Text to synthesize
            text_input_clone = st.text_area(
                "Text to synthesize with cloned voice",
                value=st.session_state.text_input,
                height=150,
                key="text_area_clone",
            )

            # Validation
            can_clone = uploaded_file is not None and reference_text_input.strip()
            if not can_clone:
                if uploaded_file is None:
                    st.warning("Please upload a reference audio file")
                elif not reference_text_input.strip():
                    st.warning("Please provide the reference audio transcription")

            generate_clone = st.button(
                "🎭 Generate (Cloned Voice)",
                type="primary",
                use_container_width=True,
                disabled=not can_clone,
                key="gen_clone"
            )

    # Determine which text to use and whether to clone
    use_cloning = False
    text_input = text_input_default

    if generate_clone and can_clone:
        use_cloning = True
        text_input = text_input_clone
        st.session_state.text_input = text_input_clone
    elif generate_default:
        use_cloning = False
        text_input = text_input_default
        st.session_state.text_input = text_input_default
        reference_audio_bytes = None
        reference_text_value = None

    # Generation
    if (generate_default or generate_clone) and text_input.strip():
        # Clear previous audio
        st.session_state.generated_audio = None
        st.session_state.generation_info = None

        progress_placeholder = st.empty()

        with progress_placeholder.container():
            if use_cloning:
                st.info(f"Loading {selected_model_name} model for voice cloning...")
            else:
                st.info(f"Loading {selected_model_name} model...")

        try:
            llama_queue, decoder, precision = load_model_and_decoder(
                checkpoint_path, runtime_int4, runtime_int8, dac_int8
            )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

        with progress_placeholder.container():
            if use_cloning:
                st.info("Generating speech with cloned voice... This may take a moment.")
            else:
                st.info("Generating speech... This may take a moment.")

        audio_result, gen_time, error = generate_speech(
            llama_queue=llama_queue,
            decoder=decoder,
            precision=precision,
            text=text_input,
            reference_audio=reference_audio_bytes if use_cloning else None,
            reference_text=reference_text_value if use_cloning else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        progress_placeholder.empty()

        if error:
            st.error(f"Generation failed: {error}")
        elif audio_result:
            sample_rate, audio_data = audio_result
            st.session_state.generated_audio = (sample_rate, audio_data)
            st.session_state.generation_info = {
                "gen_time": gen_time,
                "model": selected_model_name,
                "cloned": use_cloning,
            }
        else:
            st.error("No audio generated. Please try again.")

    # Display results
    if st.session_state.generated_audio is not None:
        st.markdown("---")
        st.header("Generated Audio")

        sample_rate, audio_data = st.session_state.generated_audio
        gen_info = st.session_state.generation_info

        # Calculate metrics
        audio_duration = len(audio_data) / sample_rate
        rtf = gen_info['gen_time'] / audio_duration if audio_duration > 0 else 0
        speed_multiplier = audio_duration / gen_info['gen_time'] if gen_info['gen_time'] > 0 else 0

        # Success message
        clone_msg = " (voice cloned)" if gen_info.get("cloned") else ""
        st.success(f"Generated with {gen_info['model']}{clone_msg} in {gen_info['gen_time']:.2f}s")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Generation Time", f"{gen_info['gen_time']:.2f}s")
        with col2:
            st.metric("Audio Duration", f"{audio_duration:.2f}s")
        with col3:
            # Speed multiplier: how much faster than real-time (higher is better)
            st.metric(
                "Speed",
                f"{speed_multiplier:.2f}x real-time",
                help="How many seconds of audio generated per second of compute. Higher is faster."
            )
        with col4:
            # RTF: lower is better (< 1 means faster than real-time)
            rtf_color = "normal" if rtf < 1 else "off"
            st.metric(
                "RTF",
                f"{rtf:.3f}",
                delta=f"{'faster' if rtf < 1 else 'slower'} than real-time",
                delta_color=rtf_color,
                help="Real-Time Factor = generation_time / audio_duration. Lower is better. RTF < 1 means faster than real-time."
            )

        # Audio player
        st.audio(audio_data, sample_rate=sample_rate)

        # Download button
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())

        model_name_clean = gen_info['model'].replace(' ', '_').replace('(', '').replace(')', '').lower()
        clone_suffix = "_cloned" if gen_info.get("cloned") else ""
        st.download_button(
            label="📥 Download WAV",
            data=buffer.getvalue(),
            file_name=f"tts_{model_name_clean}{clone_suffix}.wav",
            mime="audio/wav",
        )

    # Footer
    st.markdown("---")

    # Model availability
    with st.expander("Available Models"):
        for name, (ckpt, rt_int4, rt_int8) in AVAILABLE_MODELS.items():
            exists = "✅" if (CHECKPOINTS_DIR / ckpt).exists() else "❌"
            if rt_int8:
                suffix = " (+ Runtime INT8)"
            elif rt_int4:
                suffix = " (+ Runtime INT4)"
            else:
                suffix = ""
            st.markdown(f"- {exists} **{name}**: `{ckpt}`{suffix}")

    # Help section
    with st.expander("Help & Tips"):
        st.markdown("""
        ### Voice Cloning Tips
        - **Reference audio**: Use 5-15 seconds of clear speech
        - **Transcription**: Must match EXACTLY what is said in the reference audio
        - **Quality**: Better reference audio = better cloning results

        ### Performance Metrics
        - **Speed (x real-time)**: How fast audio is generated. 2x means 2 seconds of audio per 1 second of compute.
        - **RTF (Real-Time Factor)**: Compute time / audio duration. RTF < 1 means faster than real-time.

        ### VRAM Usage Guide
        | Configuration | LLaMA | DAC | Total (baseline) |
        |---------------|-------|-----|------------------|
        | BF16 | ~1.8 GB | ~1.87 GB | ~3.7 GB |
        | BF16 + DAC INT8 | ~1.8 GB | ~0.66 GB | ~2.5 GB |
        | INT8 Runtime | ~1.1 GB | ~1.87 GB | ~3.0 GB |
        | INT8 Runtime + DAC INT8 | ~1.1 GB | ~0.66 GB | ~1.8 GB |
        | INT4 Runtime + DAC INT8 | ~0.85 GB | ~0.66 GB | ~1.5 GB |

        **Note:** During generation, KV cache adds ~2 GB at full sequence length.

        ### Quantization Options (Quality Order)
        1. **BF16 (Best Quality)**: Full precision
        2. **INT8 Runtime (Recommended)**: Good quality, ~36% LLaMA VRAM savings
        3. **INT8 TorchAO**: Pre-quantized INT8, similar to Runtime INT8
        4. **INT4 Runtime/TorchAO**: Max VRAM savings but lower audio quality

        ### DAC Decoder Quantization
        - **DAC INT8**: Reduces DAC from ~1.87 GB to ~0.66 GB (~65% savings!)
        - Minimal impact on audio quality
        - Enabled by default for maximum VRAM savings
        """)

    if "INT4" in selected_model_name:
        st.warning("⚠️ INT4 may have reduced audio quality. Use INT8 or BF16 for best results.")


if __name__ == "__main__":
    main()
