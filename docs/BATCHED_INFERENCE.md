# Batched Inference Implementation

## Overview

This document describes the batched inference implementation for Fish-Speech S1-Mini, which enables processing multiple TTS requests simultaneously on a single GPU for improved throughput.

## Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         HTTP API Layer              │
                                    │  POST /v1/tts/batch                 │
                                    └──────────────┬──────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              BatchQueue                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Request 1       │    │ Request 2       │    │ Request 3       │          │
│  │ text: "Hello"   │    │ text: "World"   │    │ text: "Test"    │          │
│  │ ref_hash: abc   │    │ ref_hash: abc   │    │ ref_hash: xyz   │          │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘          │
│           │                      │                      │                    │
│           └──────────┬───────────┘                      │                    │
│                      │                                  │                    │
│           ┌──────────▼──────────┐          ┌───────────▼──────────┐         │
│           │ Batch A (ref: abc)  │          │ Batch B (ref: xyz)   │         │
│           │ [Request 1, 2]      │          │ [Request 3]          │         │
│           └──────────┬──────────┘          └───────────┬──────────┘         │
└──────────────────────┼─────────────────────────────────┼────────────────────┘
                       │                                 │
                       ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BatchedModelWorker                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    DualARTransformer                                │     │
│  │                                                                     │     │
│  │  KV Cache: [batch_size=4, seq_len=4096, ...]                       │     │
│  │                                                                     │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │  decode_one_token_ar_batched()                              │   │     │
│  │  │  - Process all samples in parallel                          │   │     │
│  │  │  - Handle [B, codebook_dim, seq_len] tensors                │   │     │
│  │  │  - Per-sample EOS detection with active_mask                │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                       DAC Decoder                                   │     │
│  │  decode_vq_tokens_batched() - Batch VQ → Audio conversion          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Audio Results │
              │  [audio1, ...] │
              └────────────────┘
```

## Components

### 1. BatchQueue (`s1_mini/batch_queue.py`)

The BatchQueue collects incoming requests and groups them into batches for efficient processing.

**Key Features:**
- Configurable batch timeout (default: 200ms)
- Maximum batch size limit (default: 4)
- Request grouping by reference audio hash
- Statistics tracking (wait times, batch sizes)

**Configuration:**
```python
queue = BatchQueue(
    max_batch_size=4,        # Max requests per batch
    batch_timeout_ms=200,    # Wait time for batch to fill
    enable_grouping=True,    # Group by reference audio
    process_callback=fn,     # Callback to process batches
)
```

**Request Grouping:**
Requests are grouped by their reference audio hash to maximize cache efficiency:
- Same reference audio → same batch (shared prompt encoding)
- Different reference audio → separate batches
- Within each group, requests are sorted by text length for efficient padding

### 2. Batched Inference Functions (`fish_speech/models/text2semantic/inference_batched.py`)

Core functions for GPU-batched token generation:

#### `sample_batched(logits, temperature, top_p, repetition_penalty, previous_tokens)`
Parallel sampling from logits for all batch elements.

```python
# Input: logits [batch_size, seq_len, vocab_size]
# Output: (sampled_indices [batch_size, 1], probs [batch_size, vocab_size])
```

#### `decode_one_token_ar_batched(model, x, input_pos, ...)`
Single-step decode for all samples in batch.

```python
# Input: x [batch_size, codebook_dim, seq_len]
# Output: next_tokens [batch_size, codebook_dim+1, 1]
```

Key differences from sequential version:
- Processes all batch elements simultaneously
- Uses `active_mask` to track which samples are still generating
- Handles variable-length generation per sample

#### `decode_n_tokens_batched(model, cur_token, input_pos, num_new_tokens, ...)`
Multi-step generation with per-sample EOS detection.

```python
# Returns: (generated_tokens, lengths)
# generated_tokens: [codebook_dim+1, batch_size, max_gen_len]
# lengths: [batch_size] - actual length per sample
```

Features:
- Early termination per sample on EOS token
- Windowed repetition penalty
- Progress tracking with tqdm

#### `generate_batched(model, prompts, max_new_tokens, ...)`
Full generation pipeline for multiple prompts.

```python
results = generate_batched(
    model=model,
    prompts=[prompt1, prompt2, prompt3],  # List of [codebook_dim, seq_len]
    max_new_tokens=2048,
    audio_masks_list=[mask1, mask2, mask3],
    audio_parts_list=[parts1, parts2, parts3],
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.1,
)
# Returns: List of generated sequences
```

### 3. BatchedModelWorker (`s1_mini/batch_worker.py`)

Worker thread that processes batches using the batched inference functions.

**Responsibilities:**
- Receives batches from the queue
- Initializes KV cache with appropriate batch size
- Calls batched inference functions
- Distributes results to per-request response queues
- Handles OOM errors gracefully

```python
worker = BatchedModelWorker(
    model=model,
    decode_one_token=decode_fn,
    device="cuda",
    max_batch_size=4,
)
worker.start()
```

### 4. Batched Reference Encoding (`fish_speech/inference_engine/vq_manager.py`)

New methods for batch processing reference audios:

#### `encode_references_batched(reference_audios, enable_reference_audio)`
Encode multiple reference audios in a single forward pass.

```python
# Input: List of audio bytes
# Output: List of encoded tokens [num_codebooks, seq_len]
```

Process:
1. Load all audio files
2. Pad to same length
3. Stack into batch tensor
4. Single DAC encode forward pass
5. Split results back to individual tensors

#### `decode_vq_tokens_batched(codes_list)`
Decode multiple VQ token sequences in a single forward pass.

```python
# Input: List of code tensors [num_codebooks, seq_len]
# Output: List of audio tensors
```

### 5. Configuration (`s1_mini/config.py`)

New batch-related settings:

```python
@dataclass
class EngineConfig:
    # Batching Configuration
    enable_batching: bool = True       # Enable batched inference
    max_batch_size: int = 4            # Max requests per batch
    batch_timeout_ms: int = 200        # Time to wait for batch to fill
    batch_grouping: bool = True        # Group by reference audio
```

Environment variables:
- `S1_MINI_ENABLE_BATCHING` - Enable/disable batching
- `S1_MINI_BATCH_SIZE` - Maximum batch size
- `S1_MINI_BATCH_TIMEOUT` - Batch timeout in milliseconds
- `S1_MINI_BATCH_GROUPING` - Enable/disable reference grouping

### 6. Engine Integration (`s1_mini/engine.py`)

New batch generation methods:

#### `generate_batch(texts, reference_audios, reference_texts, ...)`
Synchronous batch generation.

```python
response = engine.generate_batch(
    texts=["Hello", "World", "Test"],
    reference_audios=[audio1, audio2, None],
    reference_texts=["ref1", "ref2", None],
    max_new_tokens=2048,
    temperature=0.7,
    return_bytes=True,
)

# Response:
# BatchGenerationResponse(
#     success=True,
#     results=[GenerationResponse(...), ...],
#     total_time=5.2,
# )
```

#### `generate_batch_async(texts, ...)`
Async version for integration with async web frameworks.

```python
response = await engine.generate_batch_async(
    texts=["Hello", "World"],
    ...
)
```

### 7. API Endpoint (`s1_mini/server.py`)

New batch endpoint:

```
POST /v1/tts/batch
```

**Request:**
```json
{
    "items": [
        {
            "text": "Hello, this is the first message.",
            "reference_audio": "base64...",
            "reference_text": "Reference text here"
        },
        {
            "text": "And this is the second message."
        }
    ],
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "chunk_length": 200
}
```

**Response:**
```json
{
    "success": true,
    "results": [
        {
            "success": true,
            "audio_base64": "UklGR...",
            "generation_time": 2.5
        },
        {
            "success": true,
            "audio_base64": "UklGR...",
            "generation_time": 2.3
        }
    ],
    "total_time": 4.8
}
```

## Memory Budget

For a T4 GPU with 16GB VRAM:

| Component | Memory |
|-----------|--------|
| Model weights | ~4.3 GB |
| KV Cache (per batch item) | ~537 MB |
| Available for batching | ~11.7 GB |
| **Recommended max_batch_size** | **4-8** |

Memory formula:
```
Total = Model + (KV_Cache_Per_Item × batch_size) + Activations
```

## Performance Expectations

| Batch Size | Throughput Improvement | Latency Increase |
|------------|----------------------|------------------|
| 1 | 1x (baseline) | 0ms |
| 2 | ~1.8x | +50-100ms |
| 4 | ~3.2x | +100-200ms |
| 8 | ~5x | +200-400ms |

Note: Actual performance depends on:
- Text lengths (more similar = better batching)
- Reference audio (shared = better efficiency)
- GPU memory bandwidth
- Model precision (fp16 vs bf16)

## Usage Examples

### Python API

```python
from s1_mini import ProductionTTSEngine, EngineConfig

# Configure with batching
config = EngineConfig(
    checkpoint_path="checkpoints/openaudio-s1-mini",
    device="cuda",
    enable_batching=True,
    max_batch_size=4,
    batch_timeout_ms=200,
)

engine = ProductionTTSEngine(config)
engine.start()

# Generate batch
response = engine.generate_batch(
    texts=[
        "Welcome to our service.",
        "Your order has been confirmed.",
        "Thank you for your purchase.",
        "Have a great day!",
    ],
    temperature=0.7,
    return_bytes=True,
)

# Process results
for i, result in enumerate(response.results):
    if result.success:
        with open(f"output_{i}.wav", "wb") as f:
            f.write(result.audio_bytes)
    else:
        print(f"Failed: {result.error}")

print(f"Total time: {response.total_time:.2f}s")
```

### HTTP API

```bash
curl -X POST "http://localhost:8080/v1/tts/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"text": "Hello world"},
            {"text": "Goodbye world"}
        ],
        "temperature": 0.7
    }' | jq '.results[].audio_base64' | head -1 | base64 -d > output.wav
```

### Async Integration

```python
import asyncio
from s1_mini import ProductionTTSEngine, EngineConfig

async def process_batch():
    config = EngineConfig(enable_batching=True)
    engine = ProductionTTSEngine(config)
    engine.start()

    response = await engine.generate_batch_async(
        texts=["Hello", "World"],
        return_bytes=True,
    )

    return response

asyncio.run(process_batch())
```

## Troubleshooting

### OOM Errors

If you encounter out-of-memory errors:

1. Reduce `max_batch_size`:
   ```python
   config = EngineConfig(max_batch_size=2)
   ```

2. Use smaller texts or split long texts

3. Monitor VRAM usage:
   ```python
   health = engine.health_check()
   print(f"VRAM: {health['vram']['used_gb']:.1f}/{health['vram']['total_gb']:.1f} GB")
   ```

### Slow Batch Formation

If batches are too slow to form:

1. Reduce `batch_timeout_ms`:
   ```python
   config = EngineConfig(batch_timeout_ms=100)
   ```

2. Disable grouping if references vary:
   ```python
   config = EngineConfig(batch_grouping=False)
   ```

### Inconsistent Results

For reproducible results:

1. Set a seed:
   ```python
   response = engine.generate_batch(texts=texts, seed=42)
   ```

2. Use the same reference audio for all items in batch

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_batch_inference.py -v

# Integration tests
pytest tests/test_batch_inference.py::TestBatchIntegration -v

# Performance tests
pytest tests/test_batch_inference.py::TestBatchPerformance -v
```
