# Voice Service

Production TTS service using OpenAudio S1 Mini with OpenAI-compatible API.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint
- Abstract TTS provider interface (S1 Mini, ElevenLabs, Mock)
- Zero-shot voice cloning with HuggingFace/local/S3 reference storage
- Streaming audio generation
- LoRA adapter hot-swapping
- Docker deployment with GPU support

## Quick Start

### Standalone Installation

The `voice-service` is a **standalone service** with no dependencies on parent directories:

```bash
# Copy voice-service to your project
cp -r voice-service /path/to/your/project/

cd /path/to/your/project/voice-service

# Install dependencies
pip install poetry
poetry install

# Download model (inside voice-service directory)
mkdir -p checkpoints
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Configure
cp env.example.txt .env
# Edit .env:
#   - VOICE_S1_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini (relative) or absolute path
#   - VOICE_FISH_SPEECH_PATH=/absolute/path/to/fish-speech (if not installed as package)

# Run
python -m uvicorn src.main:app --reload
```

**Important:** For S1 Mini provider, you need `fish_speech`. Options:

1. **Install fish_speech as package** (recommended): `pip install fish-speech` (if available)
2. **Set VOICE_FISH_SPEECH_PATH** to absolute path of fish-speech repository root
3. **Use mock provider** for testing without S1 Mini dependencies

### Docker Deployment

```bash
# Build
docker build -t voice-service .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v ./checkpoints:/app/checkpoints \
  -e VOICE_S1_COMPILE=true \
  voice-service
```

## API Endpoints

### POST /v1/audio/speech

OpenAI-compatible TTS endpoint.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "default"}' \
  --output speech.wav
```

### GET /v1/voices

List available voices.

```bash
curl http://localhost:8000/v1/voices
```

### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_TTS_PROVIDER` | `s1_mini` | Provider: `s1_mini`, `elevenlabs`, `mock` |
| `VOICE_S1_CHECKPOINT_PATH` | - | **Required** Model checkpoint path (absolute recommended) |
| `VOICE_S1_COMPILE` | `false` | Enable torch.compile (Linux only) |
| `VOICE_S1_DEVICE` | `cuda` | Device: `cuda`, `cpu` |
| `VOICE_FISH_SPEECH_PATH` | - | Path to fish-speech repo (if not installed as package) |
| `VOICE_REFERENCE_STORAGE` | `local` | Storage: `huggingface`, `local`, `s3` |
| `VOICE_HF_REFERENCE_REPO` | - | HuggingFace repo for references |
| `VOICE_LORA_ENABLED` | `false` | Enable LoRA adapter |
| `VOICE_LORA_PATH` | - | Path to LoRA checkpoint |

## Architecture

```
src/
  api/           # FastAPI routes and schemas
  providers/     # TTS provider implementations
    base.py      # Abstract TTSProvider interface
    s1_mini.py   # Fish Audio S1 Mini
    elevenlabs.py # ElevenLabs API
    mock.py      # Mock for testing
  storage/       # Reference audio storage
    huggingface.py
    local.py
    s3.py
  inference/     # S1 Mini inference engine
    engine.py
    lora_manager.py
  config.py      # Settings
  main.py        # Application entry
```

## Finetuning

See `notebooks/finetune_s1_mini.ipynb` for finetuning on custom voice data.

Requirements:
- Linux with CUDA (Triton required)
- 16GB+ VRAM
- Dataset in LJSpeech format

## Deployment

### AWS (Async Voice Notes)

```bash
VOICE_S1_COMPILE=false  # Disable compile for batch processing
VOICE_STREAMING_ENABLED=false
```

### RunPod (Real-time Calls)

```bash
VOICE_S1_COMPILE=true  # Enable for 10x speed
VOICE_STREAMING_ENABLED=true
```

## License

See main repository LICENSE.



