"""Streamlit UI for Voice Service API."""

import io
import base64
from pathlib import Path

import httpx
import streamlit as st

API_BASE_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.set_page_config(
    page_title="Voice Service",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Voice Service")
st.markdown("Generate speech using OpenAudio S1 Mini with zero-shot voice cloning.")


@st.cache_data(ttl=60)
def get_health():
    """Check API health."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=10)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@st.cache_data(ttl=60)
def get_voices():
    """Get available voices."""
    try:
        response = httpx.get(f"{API_BASE_URL}/v1/voices", timeout=10)
        return response.json().get("voices", [])
    except Exception:
        return []


def generate_speech(
    text: str,
    voice: str = "default",
    response_format: str = "wav",
    reference_audio: bytes | None = None,
    reference_text: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    streaming: bool = False,
) -> bytes | None:
    """Generate speech via API."""
    payload = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "stream": streaming,
    }

    # Use extended endpoint if we have extra params
    endpoint = "/v1/audio/speech"
    if temperature != 0.7 or top_p != 0.8:
        endpoint = "/v1/audio/speech/extended"
        payload["temperature"] = temperature
        payload["top_p"] = top_p

    try:
        with st.spinner("Generating audio..."):
            response = httpx.post(
                f"{API_BASE_URL}{endpoint}",
                json=payload,
                timeout=120,
            )

        if response.status_code == 200:
            return response.content
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def add_reference_voice(voice_id: str, audio_data: bytes, transcript: str) -> bool:
    """Add a reference voice to the server."""
    try:
        # For now, we'll use base64 encoding since the API accepts bytes
        files = {
            "audio": ("reference.wav", audio_data, "audio/wav"),
        }
        data = {
            "id": voice_id,
            "text": transcript,
        }

        response = httpx.post(
            f"{API_BASE_URL}/v1/references/add",
            files=files,
            data=data,
            timeout=30,
        )

        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to add reference: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error adding reference: {e}")
        return False


# Sidebar - API Status
with st.sidebar:
    st.header("API Status")
    health = get_health()

    if health.get("status") == "healthy":
        st.success(f"Connected: {health.get('provider', 'unknown')}")
        st.caption(f"Models loaded: {health.get('models_loaded', False)}")
        st.caption(f"Streaming: {health.get('streaming_enabled', False)}")
        if health.get("lora_active"):
            st.caption(f"LoRA: {health.get('lora_active')}")
    elif health.get("status") == "error":
        st.error(f"Connection failed: {health.get('error', 'Unknown')}")
    else:
        st.warning(f"Status: {health.get('status', 'unknown')}")

    st.divider()

    # Available voices
    st.header("Available Voices")
    voices = get_voices()
    if voices:
        for v in voices:
            st.caption(f"- {v.get('name', v.get('voice_id'))}")
    else:
        st.caption("No voices available")

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


# Main content - tabs
tab1, tab2, tab3 = st.tabs(["Generate Speech", "Upload Reference", "Settings"])


# Tab 1: Generate Speech
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area(
            "Text to synthesize",
            value="Hello! This is a test of the voice synthesis system.",
            height=150,
            help="Enter the text you want to convert to speech.",
        )

        # Emotion tags helper
        with st.expander("Emotion Tags (click to expand)"):
            st.markdown("""
            **Basic emotions:**
            `(angry)` `(sad)` `(excited)` `(surprised)` `(satisfied)` `(delighted)`
            `(scared)` `(worried)` `(upset)` `(nervous)` `(frustrated)` `(depressed)`

            **Advanced emotions:**
            `(sincere)` `(sarcastic)` `(hesitating)` `(confident)` `(curious)`

            **Tone markers:**
            `(in a hurry tone)` `(shouting)` `(whispering)` `(soft tone)`

            **Audio effects:**
            `(laughing)` `(sighing)` `(crying loudly)`
            """)

            if st.button("Add (sincere)"):
                text_input = "(sincere) " + text_input

    with col2:
        voice_options = ["default"] + [v.get("voice_id") for v in voices if v.get("voice_id") != "default"]
        selected_voice = st.selectbox("Voice", voice_options)

        audio_format = st.selectbox(
            "Format",
            ["wav", "mp3", "flac"],
            index=0,
        )

        st.subheader("Advanced")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
        top_p = st.slider("Top P", 0.1, 1.0, 0.8, 0.05)

    st.divider()

    col_gen1, col_gen2 = st.columns(2)

    with col_gen1:
        generate_btn = st.button("🎵 Generate Speech", type="primary", use_container_width=True)

    with col_gen2:
        stream_btn = st.button("📡 Generate (Streaming)", use_container_width=True)

    if generate_btn or stream_btn:
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            audio_data = generate_speech(
                text=text_input,
                voice=selected_voice,
                response_format=audio_format,
                temperature=temperature,
                top_p=top_p,
                streaming=stream_btn,
            )

            if audio_data:
                st.success(f"Generated {len(audio_data):,} bytes of audio")

                # Audio player
                st.audio(audio_data, format=f"audio/{audio_format}")

                # Download button
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_data,
                    file_name=f"generated_speech.{audio_format}",
                    mime=f"audio/{audio_format}",
                )


# Tab 2: Upload Reference
with tab2:
    st.header("Upload Reference Audio")
    st.markdown("""
    Upload a reference audio file to create a new voice for zero-shot cloning.
    The audio should be 10-30 seconds of clear speech.
    """)

    col_ref1, col_ref2 = st.columns(2)

    with col_ref1:
        voice_id = st.text_input(
            "Voice ID",
            placeholder="e.g., my_voice",
            help="Unique identifier for this voice (alphanumeric, hyphens, underscores)",
        )

        uploaded_file = st.file_uploader(
            "Reference Audio",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Upload 10-30 seconds of clear speech",
        )

        if uploaded_file:
            st.audio(uploaded_file)
            st.caption(f"File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

    with col_ref2:
        transcript = st.text_area(
            "Transcript",
            placeholder="Enter the exact words spoken in the audio...",
            height=150,
            help="Accurate transcript of what is said in the reference audio",
        )

    st.divider()

    if st.button("📤 Upload Reference", type="primary"):
        if not voice_id:
            st.warning("Please enter a Voice ID.")
        elif not uploaded_file:
            st.warning("Please upload an audio file.")
        elif not transcript:
            st.warning("Please enter a transcript.")
        else:
            audio_bytes = uploaded_file.read()
            success = add_reference_voice(voice_id, audio_bytes, transcript)

            if success:
                st.success(f"Reference voice '{voice_id}' uploaded successfully!")
                st.cache_data.clear()
                st.balloons()
            else:
                st.error("Failed to upload reference. Check the API logs.")

    st.divider()

    # Quick test with uploaded reference
    st.subheader("Quick Test (Local)")
    st.markdown("Test with uploaded audio without adding to server:")

    if uploaded_file:
        test_text = st.text_input(
            "Test text",
            value="This is a quick test of the uploaded voice.",
        )

        if st.button("🎵 Test Generate"):
            st.info("Note: Direct reference audio injection requires the extended API endpoint.")
            st.caption("For now, upload the reference first, then use it from the Generate tab.")


# Tab 3: Settings
with tab3:
    st.header("API Configuration")

    st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        key="settings_api_url",
        help="Change this if running the API on a different host/port",
    )

    st.divider()

    st.subheader("API Information")
    health = get_health()

    if health.get("status") != "error":
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Provider", health.get("provider", "unknown"))
        with info_cols[1]:
            st.metric("Device", health.get("device", "unknown"))
        with info_cols[2]:
            st.metric("Status", health.get("status", "unknown"))

    st.divider()

    st.subheader("Quick Reference")
    st.markdown("""
    **API Endpoints:**
    - `POST /v1/audio/speech` - Generate speech (OpenAI compatible)
    - `POST /v1/audio/speech/extended` - Generate with advanced params
    - `GET /v1/voices` - List available voices
    - `GET /health` - Health check
    - `GET /docs` - Swagger UI

    **Environment Variables:**
    - `VOICE_TTS_PROVIDER` - s1_mini, elevenlabs, or mock
    - `VOICE_S1_COMPILE` - Enable torch.compile (Linux only)
    - `VOICE_REFERENCE_STORAGE` - huggingface, local, or s3
    """)


# Footer
st.divider()
st.caption("Voice Service - Production TTS with OpenAI-compatible API")
















