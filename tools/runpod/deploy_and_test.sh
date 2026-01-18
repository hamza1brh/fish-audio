#!/bin/bash
#
# Fish-Speech Quick Deploy and Test Script
#
# Syncs local changes to RunPod and runs INT8 vs BF16 comparison.
#
# Usage:
#   ./tools/runpod/deploy_and_test.sh                    # Standard test
#   ./tools/runpod/deploy_and_test.sh --heavy            # Heavy test (20 samples)
#   ./tools/runpod/deploy_and_test.sh --skip-sync        # Skip sync, just run
#
# Prerequisites:
#   - SSH config with 'runpod' host defined
#   - RunPod instance running with fish-speech at /workspace/fish-audio
#
# Setup SSH config (~/.ssh/config):
#   Host runpod
#       HostName <your-runpod-ip>
#       User root
#       Port <your-port>
#       IdentityFile ~/.ssh/id_rsa

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
RUNPOD_HOST="${RUNPOD_HOST:-runpod}"
REMOTE_PATH="${REMOTE_PATH:-/workspace/fish-audio}"
NUM_SAMPLES=10
SKIP_SYNC=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --heavy)
            NUM_SAMPLES=20
            shift
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --host)
            RUNPOD_HOST="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================="
echo "  Fish-Speech Deploy and Test"
echo "==============================================${NC}"
echo ""
echo "  RunPod Host: $RUNPOD_HOST"
echo "  Remote Path: $REMOTE_PATH"
echo "  Num Samples: $NUM_SAMPLES"
echo ""

# Step 1: Sync files
if [ "$SKIP_SYNC" = false ]; then
    echo -e "${YELLOW}Step 1: Syncing files to RunPod...${NC}"

    # Files to sync
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'checkpoints' \
        --exclude '*.egg-info' \
        tools/runpod/ \
        tools/llama/quantize*.py \
        fish_speech/models/text2semantic/inference.py \
        fish_speech/models/dac/inference.py \
        "$RUNPOD_HOST:$REMOTE_PATH/"

    echo -e "${GREEN}  Files synced!${NC}"
else
    echo -e "${YELLOW}Step 1: Skipping sync (--skip-sync)${NC}"
fi

# Step 2: Run comparison on RunPod
echo ""
echo -e "${YELLOW}Step 2: Running comparison benchmark on RunPod...${NC}"
echo ""

ssh "$RUNPOD_HOST" << EOF
    cd $REMOTE_PATH

    # Activate environment if needed
    source /workspace/venv/bin/activate 2>/dev/null || true

    # Clear any existing CUDA processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true

    # Run the comparison
    python tools/runpod/run_comparison.py \
        --num-samples $NUM_SAMPLES \
        --warmup-runs 3 \
        --output comparison_report.html

    echo ""
    echo "Done! Fetching results..."
EOF

# Step 3: Fetch results
echo ""
echo -e "${YELLOW}Step 3: Fetching results...${NC}"

mkdir -p benchmark_results

scp "$RUNPOD_HOST:$REMOTE_PATH/comparison_report.html" benchmark_results/
scp "$RUNPOD_HOST:$REMOTE_PATH/bf16_results.json" benchmark_results/
scp "$RUNPOD_HOST:$REMOTE_PATH/int8_results.json" benchmark_results/

echo ""
echo -e "${GREEN}=============================================="
echo "  Complete!"
echo "==============================================${NC}"
echo ""
echo "  Results saved to: benchmark_results/"
echo "    - comparison_report.html"
echo "    - bf16_results.json"
echo "    - int8_results.json"
echo ""
echo "  Open the HTML report:"
echo "    open benchmark_results/comparison_report.html"
echo ""
