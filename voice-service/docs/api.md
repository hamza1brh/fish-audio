# API Reference

OpenAI-compatible TTS API.

## Endpoints

### POST /v1/audio/speech

Generate speech from text.

**Request:**

```json
{
  "model": "s1-mini",
  "input": "Text to synthesize",
  "voice": "default",
  "response_format": "wav",
  "stream": false
}
```

**Parameters:**

| Field           | Type   | Default  | Description               |
| --------------- | ------ | -------- | ------------------------- |
| model           | string | s1-mini  | Model ID                  |
| input           | string | required | Text (1-4096 chars)       |
| voice           | string | default  | Voice ID                  |
| response_format | string | wav      | wav, mp3, opus, flac, pcm |
| stream          | bool   | false    | Enable streaming          |

**Response:** Audio file

### POST /v1/audio/speech/extended

Extended endpoint with additional parameters.

**Additional Parameters:**

| Field              | Type  | Default | Description                    |
| ------------------ | ----- | ------- | ------------------------------ |
| temperature        | float | 0.7     | Sampling temperature (0.1-1.0) |
| top_p              | float | 0.8     | Nucleus sampling (0.1-1.0)     |
| repetition_penalty | float | 1.1     | Repetition penalty (0.9-2.0)   |
| max_new_tokens     | int   | 1024    | Max tokens (100-4096)          |
| chunk_length       | int   | 200     | Chars per chunk (100-300)      |

### GET /v1/voices

List available voices.

**Response:**

```json
{
  "voices": [
    {
      "voice_id": "default",
      "name": "Default",
      "description": "Base model voice"
    }
  ]
}
```

### POST /v1/references/add

Add reference audio for voice cloning.

**Request:** multipart/form-data

| Field | Type   | Description          |
| ----- | ------ | -------------------- |
| id    | string | Voice ID             |
| text  | string | Transcript of audio  |
| audio | file   | Reference audio file |

### GET /v1/references

List reference voices.

### DELETE /v1/references/{voice_id}

Delete a reference voice.

### GET /health

Health check.

**Response:**

```json
{
  "status": "healthy",
  "provider": "s1_mini",
  "models_loaded": true,
  "device": "cuda",
  "streaming_enabled": true
}
```

## Streaming

Set `stream: true` for chunked audio streaming:

```python
import httpx

with httpx.stream("POST", url, json={"input": "...", "stream": True}) as r:
    for chunk in r.iter_bytes():
        play_audio(chunk)
```

## Error Codes

| Code | Description        |
| ---- | ------------------ |
| 400  | Invalid request    |
| 500  | Generation failed  |
| 503  | Provider not ready |






