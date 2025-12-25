#!/bin/bash
# Setup HuggingFace authentication for model download
set -e

echo "=========================================="
echo "HuggingFace Authentication Setup"
echo "=========================================="
echo ""

echo "The openaudio-s1-mini model requires HuggingFace authentication."
echo ""
echo "Option 1: Use HuggingFace CLI (Recommended)"
echo "  Run: huggingface-cli login"
echo "  Then enter your token when prompted"
echo ""
echo "Option 2: Set environment variable"
echo "  export HF_TOKEN=your_token_here"
echo ""
echo "Option 3: Get token from https://huggingface.co/settings/tokens"
echo "  Create a token with 'read' permissions"
echo "  Then: export HF_TOKEN=your_token_here"
echo ""

# Check if already logged in
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "✓ Already authenticated as: $(huggingface-cli whoami)"
    echo ""
    echo "You can now download the model:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')\""
else
    echo "Not authenticated. Please run: huggingface-cli login"
fi

echo ""
echo "After authentication, run: ./run-gpu-tests.sh"

