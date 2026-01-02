---
name: Voice Service Production Ready
overview: Make voice-service production-ready with dynamic request batching, real GPU tests for SageMaker validation, and clear concise documentation. No external queue or storage dependencies - those will be handled by the backend team.
todos:
  - id: batcher
    content: Create request batcher for GPU efficiency (src/inference/batcher.py)
    status: completed
  - id: engine-batch
    content: Add generate_batch() to S1MiniEngine
    status: completed
  - id: gpu-tests
    content: Create real GPU tests (no mocks) for SageMaker validation
    status: completed
  - id: benchmark-tests
    content: Add throughput benchmark tests
    status: completed
  - id: docs
    content: Create docs/ folder with quickstart, api, deployment, performance guides
    status: completed
---

# Voice Service Production Ready

## Scope

**In scope:**

- Dynamic request batching for GPU efficiency
- Real GPU integration tests (SageMaker-compatible)
- Concise, beginner-friendly documentation
- Performance benchmarks

**Out of scope (backend team):**

- Redis/Celery job queue
- S3 audio storage
- Webhook callbacks

## Architecture

```mermaid
flowchart LR
    Client --> FastAPI
    FastAPI --> Batcher[Request Batcher]
    Batcher --> Engine[S1MiniEngine]
    Engine --> GPU[GPU Inference]
    GPU --> Audio[Audio Response]
```



## Capacity Reference (A10G 24GB)

| Config | Throughput ||--------|------------|| batch=1 | ~15 req/min || batch=4 | ~45 req/min || batch=8 | ~70 req/min |

## Implementation

### 1. Request Batching

**New file**: `src/inference/batcher.py`Collects requests within a time window, batches them, processes together.

```python
class RequestBatcher:
    def __init__(self, max_batch_size: int = 4, max_wait_ms: int = 100): ...
    async def submit(self, request: TTSRequest) -> bytes: ...
```



### 2. Engine Batch Support

**Update**: [`src/inference/engine.py`](voice-service/src/inference/engine.py)Add `generate_batch()` method for processing multiple requests.

### 3. Real GPU Tests