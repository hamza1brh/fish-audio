# Production Deployment Checklist

## Pre-Deployment

### 1. Environment Variables
- [ ] `VOICE_S1_CHECKPOINT_PATH` - Absolute path to model checkpoint (or mount volume)
- [ ] `VOICE_FISH_SPEECH_PATH` - Absolute path to fish-speech repo (if not installed as package)
- [ ] `VOICE_TTS_PROVIDER=s1_mini` - Set to production provider
- [ ] `VOICE_REFERENCE_STORAGE` - Set to `huggingface` or `s3` for production
- [ ] `VOICE_HF_REFERENCE_REPO` - If using HuggingFace storage
- [ ] `VOICE_HF_TOKEN` - If using HuggingFace storage
- [ ] `VOICE_S3_BUCKET` - If using S3 storage
- [ ] `VOICE_S1_COMPILE=true` - Enable for production (Linux only)

### 2. Model Checkpoints
- [ ] Model downloaded to `/app/checkpoints/openaudio-s1-mini` (or mounted volume)
- [ ] Verify all required files: `model.pth`, `codec.pth`, `tokenizer.tiktoken`, `config.json`

### 3. Fish Speech Dependency
**CRITICAL**: The service requires `fish_speech` module. Options:
- [ ] Option A: Install fish-speech as package (if available): `pip install fish-speech`
- [ ] Option B: Copy fish-speech repo into container and set `VOICE_FISH_SPEECH_PATH=/path/to/fish-speech`
- [ ] Option C: Mount fish-speech as volume and set `VOICE_FISH_SPEECH_PATH`

### 4. Docker Build
- [ ] Build from parent directory: `docker build -f voice-service/Dockerfile -t voice-service .`
- [ ] Or copy fish-speech into voice-service before building
- [ ] Verify container starts: `docker run --gpus all voice-service`

### 5. Health Checks
- [ ] `/health` endpoint returns 200
- [ ] Provider initializes successfully
- [ ] Models load without errors

### 6. Testing
- [ ] Run all tests: `pytest tests/`
- [ ] Test API endpoints locally
- [ ] Verify streaming works
- [ ] Test with production-like load

## Known Issues Fixed

✅ Module path in Dockerfile: Fixed `voice_service.src.main` → `src.main`
✅ Sample rate test: Now accepts both 24000 and 44100
✅ Empty text validation: Added proper error handling

## Deployment Notes

### Docker Build Context
The Dockerfile expects to be built from the **parent directory** (fish-speech root) to access fish-speech:

```bash
# From fish-speech root
docker build -f voice-service/Dockerfile -t voice-service .
```

Or build from voice-service with fish-speech copied in:

```bash
cd voice-service
cp -r ../fish_speech ./fish_speech  # Copy fish-speech
docker build -t voice-service .
```

### Required Volumes
- `/app/checkpoints` - Model checkpoints (read-only)
- `/app/references` - Reference audio storage (read-write for local, read-only for HF/S3)

### GPU Requirements
- CUDA 12.1+
- 8GB+ VRAM (16GB+ recommended)
- NVIDIA driver compatible with CUDA 12.1

## Production Configuration

### Recommended Settings
```bash
VOICE_TTS_PROVIDER=s1_mini
VOICE_S1_DEVICE=cuda
VOICE_S1_COMPILE=true  # Linux only, requires Triton
VOICE_REFERENCE_STORAGE=huggingface  # or s3
VOICE_LOG_LEVEL=info
VOICE_STREAMING_ENABLED=true
```

### Not for Production
- ❌ `VOICE_TTS_PROVIDER=mock` - Development only
- ❌ `VOICE_S1_COMPILE=false` - Slower inference
- ❌ `VOICE_REFERENCE_STORAGE=local` - Use HF or S3 for multi-instance



