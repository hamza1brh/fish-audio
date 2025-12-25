#!/bin/bash
set -e

MODEL_PATH="${VOICE_S1_CHECKPOINT_PATH:-/app/checkpoints/openaudio-s1-mini}"

# Download model if not present
if [ ! -f "$MODEL_PATH/model.pth" ]; then
    echo "Model not found at $MODEL_PATH, downloading from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
import os
path = os.environ.get('VOICE_S1_CHECKPOINT_PATH', '/app/checkpoints/openaudio-s1-mini')
snapshot_download('fishaudio/openaudio-s1-mini', local_dir=path)
print('Model downloaded successfully')
"
fi

# Execute the main command
exec "$@"


