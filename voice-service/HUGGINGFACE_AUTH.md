# HuggingFace Authentication for Model Download

## Problem

The `fishaudio/openaudio-s1-mini` model is a **gated repository** on HuggingFace, meaning it requires authentication to download.

## Solution Options

### Option 1: HuggingFace CLI Login (Recommended)

```bash
# Install huggingface-cli if not already installed
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Enter your token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

### Option 2: Environment Variable

```bash
# Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here

# Or
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Then download
python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini', token='$HF_TOKEN')
"
```

### Option 3: Use Script with Token

```bash
# Set token
export HF_TOKEN=your_token_here

# Run GPU tests script (will use token)
./run-gpu-tests.sh
```

## Getting a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "voice-service")
4. Select "Read" permissions
5. Copy the token
6. Use it in one of the methods above

## Verify Authentication

```bash
# Check if logged in
huggingface-cli whoami

# Should show your username, or "Not logged in"
```

## Download Model Manually

After authentication:

```bash
python -c "
from huggingface_hub import snapshot_download
from pathlib import Path

checkpoint_dir = Path('checkpoints/openaudio-s1-mini')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    'fishaudio/openaudio-s1-mini',
    local_dir=str(checkpoint_dir),
)
print('Model downloaded successfully')
"
```

## Alternative: Use Pre-downloaded Model

If you already have the model elsewhere:

```bash
# Set path to existing checkpoint
export VOICE_S1_CHECKPOINT_PATH=/path/to/openaudio-s1-mini

# Run tests
pytest tests/ --gpu -v
```

## Troubleshooting

### "401 Unauthorized"
- Make sure you're logged in: `huggingface-cli whoami`
- Check token is valid: `huggingface-cli login`
- Verify token has read access to the repo

### "GatedRepoError"
- The model requires access approval
- Request access at: https://huggingface.co/fishaudio/openaudio-s1-mini
- Wait for approval, then try again

### "Token not found"
- Set environment variable: `export HF_TOKEN=your_token`
- Or use CLI: `huggingface-cli login`

