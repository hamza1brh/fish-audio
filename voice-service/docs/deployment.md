# Deployment

Production deployment options.

## Docker

### Build

```bash
docker build -t voice-service:latest .
```

### Run

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ./checkpoints:/app/checkpoints:ro \
  -e VOICE_TTS_PROVIDER=s1_mini \
  -e VOICE_S1_CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini \
  voice-service:latest
```

### Compose

```bash
docker-compose up -d
```

## SageMaker

### Endpoint Configuration

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/voice-service.tar.gz",
    role=role,
    framework_version="2.1",
    py_version="py310",
    entry_point="src/main.py",
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
)
```

### Instance Types

| Instance | GPU | VRAM | Use Case |
|----------|-----|------|----------|
| ml.g5.xlarge | A10G | 24GB | Production |
| ml.g5.2xlarge | A10G | 24GB | High throughput |
| ml.p4d.24xlarge | A100 | 320GB | Multi-model |

## EKS

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-service
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: voice-service
        image: voice-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: VOICE_TTS_PROVIDER
          value: s1_mini
```

### GPU Node Pool

```bash
eksctl create nodegroup \
  --cluster my-cluster \
  --name gpu-nodes \
  --node-type g5.xlarge \
  --nodes-min 1 \
  --nodes-max 10
```

### Autoscaling

Use KEDA for queue-based scaling:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: voice-service-scaler
spec:
  scaleTargetRef:
    name: voice-service
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      query: voice_service_pending_requests
      threshold: "10"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| VOICE_TTS_PROVIDER | mock | s1_mini, elevenlabs, mock |
| VOICE_S1_CHECKPOINT_PATH | - | Model checkpoint path |
| VOICE_S1_DEVICE | cuda | cuda, cpu |
| VOICE_S1_COMPILE | false | Enable torch.compile |
| VOICE_FISH_SPEECH_PATH | - | Path to fish-speech repo |
| VOICE_HOST | 0.0.0.0 | Server host |
| VOICE_PORT | 8000 | Server port |

## Health Checks

```bash
curl http://localhost:8000/health
```

Kubernetes probe:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
```








