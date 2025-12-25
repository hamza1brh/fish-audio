# Voice Service Production Dockerfile
# Optimized for Linux with Triton/torch.compile support
#
# Build args:
#   CUDA_VERSION: 12.1 (default), 12.4, 12.8
#   PYTORCH_CUDA: cu121 (default), cu124, cu128
#
# Example:
#   docker build --build-arg CUDA_VERSION=12.4 --build-arg PYTORCH_CUDA=cu124 -t voice-service .

ARG CUDA_VERSION=12.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel


FROM base AS builder

ARG PYTORCH_CUDA=cu121

WORKDIR /build

# Install core dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install PyTorch with CUDA (version determined by build arg)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

# Install Triton for torch.compile
RUN pip install triton

# Install fish-speech from pip
RUN pip install fish-speech


FROM base AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy voice-service source
COPY src/ ./src/

# Create mount points for models and voices
RUN mkdir -p /app/checkpoints /app/references /app/voices

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment configuration
ENV VOICE_HOST=0.0.0.0 \
    VOICE_PORT=8000 \
    VOICE_TTS_PROVIDER=s1_mini \
    VOICE_S1_DEVICE=cuda \
    VOICE_S1_COMPILE=true \
    VOICE_S1_CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini \
    VOICE_REFERENCE_STORAGE=local \
    VOICE_LOCAL_REFERENCE_PATH=/app/references \
    VOICE_VOICES_PATH=/app/voices \
    VOICE_LOG_LEVEL=info \
    HF_HOME=/app/.cache/huggingface

# Mount points:
# /app/checkpoints - Model weights (from persistent volume)
# /app/references - Reference audio files
# /app/voices - Pre-encoded voice tokens (.npy)
VOLUME ["/app/checkpoints", "/app/references", "/app/voices"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
