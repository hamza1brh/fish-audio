# Quickstart

Get voice-service running in 5 minutes.

## Requirements

- Python 3.10+
- CUDA GPU with 8GB+ VRAM (16GB+ recommended)
- Model checkpoint (~2GB)

## Installation

```bash
cd voice-service

# Install dependencies
poetry install

# Or with pip
pip install -e .
```

## Download Model

```bash
huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir checkpoints/openaudio-s1-mini
```

## Configure

Create `.env`:

```bash
VOICE_TTS_PROVIDER=s1_mini
VOICE_S1_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini
VOICE_FISH_SPEECH_PATH=/path/to/fish-speech  # If not installed as package
```

## Run

```bash
# Development
uvicorn src.main:app --reload

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "default"}' \
  --output test.wav
```

## Docker

```bash
docker-compose up -d
```

## Next Steps

- [API Reference](api.md)
- [Deployment Guide](deployment.md)
- [Performance Tuning](performance.md)






