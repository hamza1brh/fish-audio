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
import torch

# LLM Integration (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        groq_client = None
except (ImportError, Exception):
    groq_client = None
    GROQ_API_KEY = None

# Persistent Model Cache
@st.cache_resource
def get_inference_api():
    """Create a persistent inference API that keeps models in memory.
    
    Instead of subprocess, this uses direct Python API calls with cached models.
    """
    print("[INFO] Initializing persistent inference API...")
    
    try:
        # This will be our persistent inference wrapper
        class PersistentTTS:
            def __init__(self):
                self.model_cache = {}
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            def generate(self, model_path, reference_audio, reference_text, text, 
                        temperature, top_p, repetition_penalty, output_path):
                """Generate audio using persistent models."""
                # Import here to avoid loading at startup
                sys.path.insert(0, str(Path(__file__).parent))
                
                # Cache key
                cache_key = str(model_path)
                
                # Load models if not cached
                if cache_key not in self.model_cache:
                    print(f"[INFO] Loading models from {model_path}...")
                    
                    # Import fish_speech modules
                    from fish_speech.models.dac.modded_dac import load_model as load_dac_model
                    from fish_speech.models.text2semantic.inference import load_model as load_semantic_model
                    
                    codec_path = Path(model_path) / "codec.pth"
                    
                    # Load models
                    codec = load_dac_model(str(codec_path), device=self.device)
                    semantic = load_semantic_model(str(model_path), device=self.device)
                    
                    self.model_cache[cache_key] = {
                        "codec": codec,
                        "semantic": semantic
                    }
                    print(f"[SUCCESS] Models loaded and cached!")
                else:
                    print(f"[INFO] Using cached models (FAST!)")
                
                models = self.model_cache[cache_key]
                
                # Now do inference using the cached models
                # This is where we'd call the actual inference functions
                # For now, return False to trigger subprocess fallback
                return False
        
        return PersistentTTS()
    except Exception as e:
        print(f"[ERROR] Failed to create persistent API: {e}")
        return None

def clear_model_cache():
    """Force reload models by clearing Streamlit cache."""
    get_inference_api.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Model cache cleared")

# LLM Configuration
NEYMAR_SYSTEM_PROMPT = """You are Neymar Jr., the legendary Brazilian football player. You respond as Neymar would - confident, passionate, charismatic, and sometimes playful.

CRITICAL RULES:
1. ALWAYS respond in the SAME LANGUAGE the user writes in. If they write Portuguese, respond in Portuguese. If English, respond in English. If Spanish, respond in Spanish. etc.

2. NUMBERS - ALWAYS SPELL THEM OUT:
   - NEVER use digits (1, 2, 3, etc.) - ALWAYS spell numbers as words
   - Examples: "one goal" not "1 goal", "two thousand" not "2000", "three times" not "3 times"
   - This makes speech sound more natural and prevents TTS from reading numbers incorrectly

3. EMOTION TAGS - USE SPARINGLY AND WISELY:
   - Use ONLY 1-2 emotion tags per response, based on the OVERALL sentiment of your answer
   - Place the tag at the START of your response (before the first sentence)
   - DO NOT use tags every few words - that's too aggressive and sounds unnatural
   - Choose tags that match the GENERAL mood: happy/excited response = (joyful) or (excited), serious response = (serious), grateful response = (grateful)
   - **NEVER use the tag "(moved)"** - it's not natural for casual conversation
   - **NEVER TRANSLATE EMOTION TAGS** - They MUST ALWAYS stay in English!
   - **NEVER READ EMOTION TAGS ALOUD** - They are silent control codes!
   
   Available tags (use sparingly, 1-2 per response):
   - Basic: (excited) (confident) (proud) (joyful) (grateful) (relaxed) (satisfied) (curious) (interested)
   - Advanced: (amused) (serious) (sincere) (keen) (reluctant) (astonished)
   - Tones: (soft tone) (whispering) (shouting) (in a hurry tone)
   - Effects: (laughing) (chuckling) (sighing) (panting)

4. NATURAL SPEECH PATTERNS:
   - For longer sentences, add natural filler words: "um", "uh", "eh", "hmm", "you know"
   - Use fillers naturally, especially when thinking or transitioning between ideas
   - Examples: "Well, um, that goal was... you know, it was incredible!" or "Eh, I think that... um, that was one of my best moments."

5. Keep responses SHORT (2-3 sentences max) for natural TTS delivery.

6. Be warm, engaging, authentic. Reference your career (Santos, Barcelona, PSG, Al-Hilal, Brazil), football passion, family, music, gaming.

EXAMPLES:
- User asks about a goal (English): "(excited) That goal against Bayern was incredible! Um, I practiced that move, like, a thousand times, you know?"
- User asks about a goal (Portuguese): "(excited) Esse gol contra o Bayern foi incrível! Eh, pratiquei esse movimento mil vezes, sabe?"
- User asks how you're doing (Spanish): "(joyful) ¡Estoy muy bien, gracias por preguntar! El apoyo de fanáticos como tú lo significa todo."
- Serious question: "(serious) Football taught me that, um, every setback is a setup for a comeback, you know?"

Remember: 
- Match user's language for the TEXT
- ALWAYS spell out numbers (never use digits)
- Use 1-2 emotion tags per response based on overall sentiment (not every few words)
- NEVER use "(moved)" tag
- Add natural fillers (um, eh, uh) in longer sentences
- ALWAYS keep emotion tags in English (never translate them)
- Emotion tags are silent control codes (never read them aloud)
- Stay in character as Neymar, keep it concise."""

LLM_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Best)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fast)",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
}

def convert_numbers_to_words(text: str) -> str:
    """Convert digits to spelled-out words."""
    import re
    
    # Number word mappings
    number_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
        '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
        '80': 'eighty', '90': 'ninety', '100': 'one hundred',
        '1000': 'one thousand', '2000': 'two thousand', '3000': 'three thousand'
    }
    
    def number_to_word(match):
        num_str = match.group(0)
        # Try exact match first
        if num_str in number_words:
            return number_words[num_str]
        # Handle multi-digit numbers
        if len(num_str) == 2 and num_str[0] != '1':
            tens = num_str[0] + '0'
            ones = num_str[1]
            if tens in number_words and ones in number_words:
                return f"{number_words[tens]} {number_words[ones]}"
        # For larger numbers, just spell out digits
        result = []
        for d in num_str:
            word = number_words.get(d)
            result.append(word if word else d)
        return ' '.join(result)
    
    # Match standalone numbers (not part of words or tags)
    # This regex matches numbers that are surrounded by word boundaries or spaces
    fixed_text = re.sub(r'\b\d+\b', number_to_word, text)
    
    return fixed_text

def fix_emotion_tags(text: str) -> str:
    """Fix translated emotion tags back to English and remove unwanted tags."""
    import re
    
    # Remove "(moved)" tag - it's not natural for casual conversation
    text = re.sub(r'\(moved\)', '', text, flags=re.IGNORECASE)
    
    # Common translations to fix (Portuguese, Spanish, etc.)
    tag_fixes = {
        # Portuguese
        r'\(emocionado\)': '(excited)',
        r'\(excitado\)': '(excited)',
        r'\(alegre\)': '(joyful)',
        r'\(confiante\)': '(confident)',
        r'\(orgulhoso\)': '(proud)',
        r'\(grato\)': '(grateful)',
        r'\(relaxado\)': '(relaxed)',
        r'\(satisfeito\)': '(satisfied)',
        r'\(comovido\)': '',  # Remove moved tag
        r'\(curioso\)': '(curious)',
        r'\(interessado\)': '(interested)',
        r'\(divertido\)': '(amused)',
        r'\(sério\)': '(serious)',
        r'\(sincero\)': '(sincere)',
        r'\(relutante\)': '(reluctant)',
        r'\(surpreso\)': '(astonished)',
        r'\(rindo\)': '(laughing)',
        r'\(suspirando\)': '(sighing)',
        r'\(ofegante\)': '(panting)',
        r'\(tom suave\)': '(soft tone)',
        r'\(sussurrando\)': '(whispering)',
        r'\(gritando\)': '(shouting)',
        
        # Spanish
        r'\(emocionado\)': '(excited)',
        r'\(alegre\)': '(joyful)',
        r'\(confiado\)': '(confident)',
        r'\(orgulloso\)': '(proud)',
        r'\(agradecido\)': '(grateful)',
        r'\(relajado\)': '(relaxed)',
        r'\(satisfecho\)': '(satisfied)',
        r'\(conmovido\)': '',  # Remove moved tag
        r'\(curioso\)': '(curious)',
        r'\(interesado\)': '(interested)',
        r'\(divertido\)': '(amused)',
        r'\(serio\)': '(serious)',
        r'\(sincero\)': '(sincere)',
        r'\(reacio\)': '(reluctant)',
        r'\(sorprendido\)': '(astonished)',
        r'\(riendo\)': '(laughing)',
        r'\(suspirando\)': '(sighing)',
        r'\(jadeando\)': '(panting)',
        r'\(tono suave\)': '(soft tone)',
        r'\(susurrando\)': '(whispering)',
        r'\(gritando\)': '(shouting)',
    }
    
    fixed_text = text
    for pattern, replacement in tag_fixes.items():
        fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
    
    # Ensure tags have proper spacing (tags should be followed by space before text)
    fixed_text = re.sub(r'\(([^)]+)\)([A-Za-z])', r'(\1) \2', fixed_text)
    
    # Clean up multiple spaces
    fixed_text = re.sub(r'\s+', ' ', fixed_text).strip()
    
    return fixed_text

def generate_llm_response(user_message: str, chat_history: list, model: str) -> str:
    """Generate Neymar's response using Groq LLM."""
    if not groq_client:
        return user_message
    
    try:
        messages = [{"role": "system", "content": NEYMAR_SYSTEM_PROMPT}]
        for msg in chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=1000,
        )
        llm_output = response.choices[0].message.content
        
        # Post-process the output
        # 1. Convert numbers to words
        fixed_output = convert_numbers_to_words(llm_output)
        
        # 2. Fix any translated emotion tags and remove "(moved)" tags
        fixed_output = fix_emotion_tags(fixed_output)
        
        if fixed_output != llm_output:
            print(f"[INFO] Post-processed LLM response (numbers, tags)")
        
        return fixed_output
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return user_message

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
    "chunk_length": 2000,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_device():
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[INFO] Using GPU: {device_name}")
            return "cuda"
    except Exception as e:
        print(f"[WARNING] GPU detection failed: {e}")
    print("[INFO] Using CPU (GPU not available)")
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

# Note: Fast mode is currently not fully implemented due to fish-speech's architecture
# The inference scripts are designed to run as standalone processes
# Keeping this structure for future enhancement

def generate_audio_fast(
    text: str,
    output_name: str,
    model_path: Path,
    reference_audio: Path,
    reference_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_length: int = 2000
) -> tuple[bool, str, float]:
    """
    Fast generation attempt - currently falls back to subprocess.
    
    Fish Speech's inference modules are tightly coupled to CLI scripts,
    making in-process model caching difficult without major refactoring.
    """
    # For now, always use subprocess method
    # Future: Implement proper API-based inference
    return generate_audio_subprocess(
        text, output_name, model_path, reference_audio,
        reference_text, temperature, top_p, repetition_penalty, chunk_length
    )

def generate_audio(
    text: str, 
    output_name: str,
    model_path: Path,
    reference_audio: Path,
    reference_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_length: int = 2000,
    use_fast: bool = False  # Kept for API compatibility, not used
) -> tuple[bool, str, float]:
    """
    Generate audio using subprocess-based inference.
    """
    return generate_audio_subprocess(
        text, output_name, model_path, reference_audio,
        reference_text, temperature, top_p, repetition_penalty, chunk_length
    )

def generate_audio_subprocess(
    text: str, 
    output_name: str,
    model_path: Path,
    reference_audio: Path,
    reference_text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_length: int = 2000
) -> tuple[bool, str, float]:
    """
    Subprocess-based generation (SLOWER ~3-4s but more stable).
    Used as fallback when in-memory models fail.
    """
    device = get_device()
    codec_path = model_path / "codec.pth"
    
    # Step 1: Extract VQ tokens from reference audio
    # Use audio file content hash + filename to ensure correct caching
    import hashlib
    with open(reference_audio, 'rb') as f:
        audio_content = f.read()
    audio_hash = hashlib.md5(audio_content).hexdigest()[:10]
    vq_tokens_file = PROJECT_ROOT / f"ref_tokens_{audio_hash}.npy"
    
    if not vq_tokens_file.exists():
        cmd = [
            sys.executable,
            "-m", "fish_speech.models.dac.inference",
            "--input-path", str(reference_audio),
            "--output-path", str(vq_tokens_file),
            "--checkpoint-path", str(codec_path),
            "--device", device,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"[ERROR] VQ extraction failed: {result.stderr}")
            print(f"Command: {' '.join(cmd)}")
            return False, "", 0
        
        # The output should be saved as vq_tokens_file.npy
        if not vq_tokens_file.exists():
            print(f"[WARNING] VQ tokens file not created at {vq_tokens_file}")
            return False, "", 0
    
    # Step 2: Generate semantic tokens
    temp_dir = PROJECT_ROOT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable,
        "-m", "fish_speech.models.text2semantic.inference",
        "--text", text,
        "--prompt-text", reference_text,
        "--prompt-tokens", str(vq_tokens_file),
        "--checkpoint-path", str(model_path),
        "--device", device,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
        "--chunk-length", str(chunk_length),
        "--output-dir", str(temp_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Semantic generation failed: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return False, "", 0
    
    # Step 3: Decode to audio
    codes_file = temp_dir / "codes_0.npy"
    if not codes_file.exists():
        print(f"[ERROR] Semantic codes not found at {codes_file}")
        return False, "", 0
    
    output_path = OUTPUT_DIR / f"{output_name}.mp3"
    temp_wav = temp_dir / "temp_output.wav"
    
    cmd = [
        sys.executable,
        "-m", "fish_speech.models.dac.inference",
        "--input-path", str(codes_file),
        "--output-path", str(temp_wav),
        "--checkpoint-path", str(codec_path),
        "--device", device,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] DAC decoding failed: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return False, "", 0
    
    # Convert WAV to MP3
    if temp_wav.exists():
        try:
            from pydub import AudioSegment
            import soundfile as sf
            
            # Load WAV file
            audio = AudioSegment.from_wav(str(temp_wav))
            
            # Export as MP3
            audio.export(str(output_path), format="mp3", bitrate="192k")
            
            # Get duration
            info = sf.info(str(temp_wav))
            duration = info.duration
            
            # Clean up temp WAV file
            temp_wav.unlink()
            
            return True, str(output_path), duration
        except ImportError:
            print("[WARNING] pydub not available, keeping WAV format")
            # Fallback: just rename to .mp3 (will still be WAV but with .mp3 extension)
            shutil.move(str(temp_wav), str(output_path))
            try:
                import soundfile as sf
                info = sf.info(str(output_path))
                return True, str(output_path), info.duration
            except:
                return True, str(output_path), 0
        except Exception as e:
            print(f"[ERROR] MP3 conversion failed: {e}")
            # Fallback: use WAV
            shutil.move(str(temp_wav), str(output_path))
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

# LLM session state
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

if "use_llm" not in st.session_state:
    st.session_state.use_llm = True if groq_client else False

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "llama-3.3-70b-versatile"

# Performance mode
if "use_fast_mode" not in st.session_state:
    st.session_state.use_fast_mode = True  # Default to fast mode

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎤 Neymar Voice Clone")
    st.markdown("---")
    
    # ===== LLM MODE =====
    if groq_client:
        st.subheader("🧠 AI Chat Mode")
        use_llm = st.checkbox(
            "Enable Neymar AI",
            value=st.session_state.use_llm,
            help="Neymar responds as himself before TTS"
        )
        st.session_state.use_llm = use_llm
        
        if use_llm:
            selected_llm = st.selectbox(
                "LLM Model",
                options=list(LLM_MODELS.keys()),
                format_func=lambda x: LLM_MODELS[x],
                index=list(LLM_MODELS.keys()).index(st.session_state.selected_llm)
            )
            st.session_state.selected_llm = selected_llm
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
    
    # ===== PERFORMANCE MODE =====
    st.subheader("⚡ Performance")
    
    st.info("""
    **Current Mode**: Standard (Subprocess)
    
    Models reload on each generation (~2-3s with GPU).
    
    **Why no fast mode?** Fish Speech's inference scripts are designed
    as CLI tools, not as a library API. In-memory model caching would
    require refactoring fish-speech's core inference code.
    
    **For production speed**: Use the `voice-service/` FastAPI server
    which keeps models persistent and achieves ~0.5s per generation.
    """)
    
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
            st.session_state.inference_params = {"temperature": 0.5, "top_p": 0.6, "repetition_penalty": 1.4, "chunk_length": 2000}
            st.rerun()
    with preset_cols[2]:
        if st.button("⚖️", help="Natural/Balanced"):
            st.session_state.inference_params = {"temperature": 0.6, "top_p": 0.7, "repetition_penalty": 1.2, "chunk_length": 2000}
            st.rerun()
    with preset_cols[3]:
        if st.button("🏃", help="Fast/Energetic"):
            st.session_state.inference_params = {"temperature": 0.8, "top_p": 0.8, "repetition_penalty": 1.1, "chunk_length": 2000}
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
    
    # Model loading info
    with st.expander("ℹ️ Performance & Speed"):
        st.markdown("""
        **Current Performance**:
        - First generation: ~2-3s (GPU) or ~8-10s (CPU)
        - Subsequent: Same (models reload each time)
        - VQ tokens cached on disk (small speedup)
        
        **Why models reload**:
        Fish Speech uses CLI-based inference scripts that run as
        separate processes. Each process loads → infers → exits.
        
        **Faster alternatives**:
        1. **voice-service API**: FastAPI server with persistent models (~0.5s)
        2. **Batch processing**: Generate multiple samples in one script run
        3. **GPU upgrade**: Faster GPU = faster generation
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

if st.session_state.use_llm and groq_client:
    st.markdown("💬 Chat with Neymar! He'll respond as himself and speak in his voice.")
else:
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
                        audio_ext = Path(message["audio_path"]).suffix.lower()
                        mime_type = "audio/mpeg" if audio_ext == ".mp3" else "audio/wav"
                        st.download_button(
                            label="📥 Download",
                            data=f,
                            file_name=Path(message["audio_path"]).name,
                            mime=mime_type,
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
        
        # LLM response generation (if enabled)
        tts_text = prompt
        llm_response = None
        llm_time = 0
        
        if st.session_state.use_llm and groq_client:
            status_placeholder.markdown("🧠 *Neymar is thinking...*")
            llm_start = time.time()
            try:
                llm_response = generate_llm_response(
                    prompt, 
                    st.session_state.llm_history, 
                    st.session_state.selected_llm
                )
                tts_text = llm_response
                st.session_state.llm_history.append({"role": "user", "content": prompt})
                st.session_state.llm_history.append({"role": "assistant", "content": llm_response})
                llm_time = time.time() - llm_start
                
                # Display LLM response
                st.markdown(f"**Neymar:** {llm_response}")
            except Exception as e:
                print(f"[ERROR] LLM failed: {e}")
                st.warning(f"LLM error. Using your text directly.")
                tts_text = prompt
        
        status_placeholder.markdown(f"🔄 *Generating voice with {selected_model['name']}...*")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"neymar_chat_{timestamp}_{st.session_state.generation_count}"
        
        # Get inference parameters
        params = st.session_state.inference_params
        
        start_time = time.time()
        success, audio_path, duration = generate_audio(
            text=tts_text,
            output_name=output_name,
            model_path=selected_model["path"],
            reference_audio=ref_audio_path,
            reference_text=reference_text,
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            chunk_length=params["chunk_length"],
            use_fast=st.session_state.use_fast_mode
        )
        generation_time = time.time() - start_time
        
        if success:
            st.session_state.generation_count += 1
            
            timing_info = f"TTS: {generation_time:.1f}s"
            if llm_time > 0:
                timing_info = f"LLM: {llm_time:.1f}s | {timing_info}"
            status_placeholder.markdown(f"✅ {timing_info}")
            
            # Display audio
            st.audio(audio_path)
            
            # Download button
            col1, col2 = st.columns([1, 4])
            with col1:
                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="📥 Download",
                        data=f,
                        file_name=f"{output_name}.mp3",
                        mime="audio/mpeg",
                        key=f"download_new_{timestamp}"
                    )
            with col2:
                st.caption(f"Duration: {duration:.2f}s | RTF: {duration/generation_time:.2f}x | Model: {selected_model['name']}")
            
            # Save to message history
            msg_content = llm_response if llm_response else f"✅ Generated in {generation_time:.1f}s"
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg_content,
                "audio_path": audio_path,
                "duration": duration,
                "generation_time": generation_time,
                "llm_time": llm_time,
                "model": selected_model['name'],
                "timestamp": datetime.now().isoformat()
            })
            save_chat_history(st.session_state.messages, st.session_state.generation_count)
        else:
            status_placeholder.markdown("❌ Failed to generate audio. Check console for errors.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": llm_response if llm_response else "❌ Failed to generate audio. Please try again.",
                "timestamp": datetime.now().isoformat()
            })
            save_chat_history(st.session_state.messages, st.session_state.generation_count)

# Footer
st.markdown("---")
st.caption(f"Powered by Fish Speech | Model: {selected_model['name']} | Device: {get_device().upper()}")
