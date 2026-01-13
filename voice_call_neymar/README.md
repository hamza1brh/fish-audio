# Voice Call Neymar

Real-time voice conversation with Neymar's AI clone using LiveKit Agents.

## Architecture

- **Transport**: LiveKit (self-hosted WebRTC)
- **STT**: Groq Whisper
- **LLM**: Groq Llama 3.3 70B
- **TTS**: ElevenLabs (cloned Neymar voice)
- **VAD**: Silero (voice activity detection)

## Setup

### 1. Install Dependencies

```bash
cd voice_call_neymar
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `env.example.txt` to `.env` in the project root and fill in your API keys:

```bash
cp env.example.txt ../.env
```

Required keys:
- `GROQ_API_KEY` - Get from console.groq.com
- `ELEVEN_LABS_KEY` - Get from elevenlabs.io
- `ELEVEN_LABS_VOICE_ID` - Your cloned Neymar voice ID

### 3. Start LiveKit Server

Using Docker:
```bash
docker compose up livekit
```

Or run directly:
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```

### 4. Run the Agent

```bash
python -m voice_call_neymar.agent dev
```

### 5. Connect via Browser

1. Go to: https://agents-playground.livekit.io
2. Click "Connect"
3. Enter your LiveKit URL: `ws://localhost:7880`
4. Use API Key: `devkey` and Secret: `secret`
5. Start talking!

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LIVEKIT_URL` | LiveKit server WebSocket URL | `ws://localhost:7880` |
| `LIVEKIT_API_KEY` | LiveKit API key | `devkey` |
| `LIVEKIT_API_SECRET` | LiveKit API secret | `secret` |
| `GROQ_API_KEY` | Groq API key | (required) |
| `GROQ_LLM_MODEL` | LLM model | `llama-3.3-70b-versatile` |
| `GROQ_STT_MODEL` | STT model | `whisper-large-v3` |
| `ELEVEN_LABS_KEY` | ElevenLabs API key | (required) |
| `ELEVEN_LABS_VOICE_ID` | Voice clone ID | (required) |
| `ELEVEN_LABS_MODEL` | TTS model | `eleven_turbo_v2_5` |

## Production Deployment

1. Deploy LiveKit server on your infrastructure
2. Update `LIVEKIT_URL` to point to your server
3. Generate proper API keys (not devkey/secret)
4. Run the agent with production settings:
   ```bash
   python -m voice_call_neymar.agent start
   ```
