# S1-Mini Production Inference Engine

A production-ready inference engine for Fish-Speech S1-Mini TTS model.

## Features

- **Persistent Model Loading**: Models stay in VRAM between requests (no reload per request)
- **Optimized VRAM Management**: No aggressive cache clearing after each generation
- **Platform-Aware Compilation**: Triton on Linux/SageMaker, eager fallback on Windows
- **Production Server**: FastAPI-based REST API with health checks and metrics
- **Request Streaming**: Real-time audio streaming for low-latency playback
- **Comprehensive Monitoring**: Prometheus-compatible metrics endpoint

## Quick Start

### Python API

```python
from s1_mini import ProductionTTSEngine, EngineConfig

# Create configuration
config = EngineConfig(
    checkpoint_path="checkpoints/openaudio-s1-mini",
    device="cuda",
    precision="float16",
    compile_model=True,  # Uses Triton on Linux, eager on Windows
)

# Create and start engine
engine = ProductionTTSEngine(config)
engine.start()

# Generate audio
response = engine.generate(
    text="Hello, this is a test of the text to speech system.",
    temperature=0.7,
    top_p=0.8,
)

if response.success:
    sample_rate, audio = response.audio
    # Save or play audio
    import soundfile as sf
    sf.write("output.wav", audio, sample_rate)

# Clean up
engine.stop()
```

### Context Manager

```python
from s1_mini import ProductionTTSEngine, EngineConfig

config = EngineConfig(checkpoint_path="checkpoints/openaudio-s1-mini")

with ProductionTTSEngine(config) as engine:
    response = engine.generate("Hello world!")
    # Engine automatically starts and stops
```

### REST API Server

```bash
# Start the server
python -m s1_mini.server --checkpoint checkpoints/openaudio-s1-mini --port 8080

# Or with environment variables
export S1_MINI_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini
export S1_MINI_DEVICE=cuda
python -m s1_mini.server
```

API endpoints:
- `POST /v1/tts` - Generate audio (returns WAV file)
- `POST /v1/tts/stream` - Streaming audio generation
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

### Example API Request

```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}' \
  --output speech.wav
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `S1_MINI_CHECKPOINT_PATH` | Model checkpoint directory | `checkpoints/openaudio-s1-mini` |
| `S1_MINI_DEVICE` | Device (cuda, cpu, mps) | `cuda` |
| `S1_MINI_PRECISION` | Precision (float16, bfloat16, float32) | `float16` |
| `S1_MINI_COMPILE` | Enable torch.compile | `true` |
| `S1_MINI_COMPILE_BACKEND` | Compile backend (auto, inductor, eager) | `auto` |
| `S1_MINI_REQUEST_TIMEOUT` | Request timeout in seconds | `60` |

### EngineConfig Options

```python
config = EngineConfig(
    # Model
    checkpoint_path="checkpoints/openaudio-s1-mini",
    device="cuda",
    precision="float16",

    # Compilation
    compile_model=True,
    compile_backend="auto",  # auto, inductor, eager, disabled
    compile_mode="reduce-overhead",

    # VRAM Management
    vram_clear_on_oom_only=True,  # Only clear cache on OOM
    vram_clear_threshold_gb=2.0,

    # Timeouts
    request_timeout_seconds=60.0,

    # Generation Defaults
    default_temperature=0.7,
    default_top_p=0.8,
    default_repetition_penalty=1.1,
)
```

## Platform Support

### Linux (Recommended for Production)

- Full Triton support for maximum performance
- Use `compile_backend="inductor"` for best results
- Ideal for AWS SageMaker deployment

### Windows (Development)

- Triton is NOT supported on Windows
- Automatic fallback to `eager` backend
- Still functional but ~2-3x slower than Linux

To check your platform capabilities:

```python
from s1_mini.compilation import get_platform_info, get_compilation_recommendations

info = get_platform_info()
print(f"Triton available: {info.triton_available}")
print(f"Recommended backend: {info.recommended_backend}")

for rec in get_compilation_recommendations():
    print(f"  - {rec}")
```

## Key Improvements Over Original

### 1. No Aggressive VRAM Clearing

Original code cleared VRAM after every request:
```python
# Original - SLOW
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

New code only clears on OOM:
```python
# New - FAST
# Cache is kept hot for subsequent requests
# Only cleared on OOM or explicit request
```

### 2. Platform-Aware Compilation

```python
# Automatically detects platform
if platform.system() == "Linux":
    backend = "inductor"  # Triton available
else:
    backend = "eager"  # Windows fallback
```

### 3. Persistent Model Loading

Models are loaded once and stay in VRAM:
- No reload per request
- KV cache persists between requests
- Warmup ensures JIT compilation completes

## Performance Comparison

| Metric | Original | S1-Mini Production |
|--------|----------|-------------------|
| First request | 10-15s | 10-15s |
| Subsequent requests | 5-15s | 1-5s |
| VRAM clearing | Every request | OOM only |
| Windows support | Very slow | Reasonable |

## Architecture

```
ProductionTTSEngine
├── ModelManager
│   ├── LLAMA Model (persistent in VRAM)
│   │   ├── 32 transformer layers
│   │   └── 4 fast layers
│   └── DAC Decoder (persistent in VRAM)
│
├── Reference Cache
│   └── Pre-encoded VQ tokens
│
└── Worker Thread
    └── Sequential request processing
```

## Monitoring

### Health Check

```python
health = engine.health_check()
# {
#     "healthy": True,
#     "models_loaded": True,
#     "device": "cuda",
#     "vram": {"used_gb": 8.5, "total_gb": 24.0}
# }
```

### Prometheus Metrics

```
s1_mini_requests_total 1234
s1_mini_requests_success_total 1230
s1_mini_requests_error_total 4
s1_mini_request_duration_seconds_bucket{le="1.0"} 100
s1_mini_vram_utilization 0.35
```

## Deployment

### AWS SageMaker

1. Package the model:
```bash
tar -czvf model.tar.gz checkpoints/openaudio-s1-mini
```

2. Deploy with SageMaker:
```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="s1_mini/server.py",
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
)
```

### Docker

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY . /app
WORKDIR /app

RUN pip install -e .

EXPOSE 8080

CMD ["python", "-m", "s1_mini.server", "--port", "8080"]
```

## Troubleshooting

### VRAM Issues

```python
# Check VRAM usage
from s1_mini.utils import get_vram_info
print(get_vram_info())

# Manually clear cache if needed
engine.model_manager.clear_cache(force=True)
```

### Compilation Issues

```python
# Disable compilation if having issues
config = EngineConfig(compile_model=False)

# Or force specific backend
config = EngineConfig(compile_backend="eager")
```

### Windows-Specific Issues

If you see Triton errors on Windows:
1. The engine should automatically fall back to eager mode
2. If not, explicitly set `compile_backend="eager"`
3. Consider using WSL2 for better performance

## License

Apache 2.0

## Contributing

See CONTRIBUTING.md for guidelines.
