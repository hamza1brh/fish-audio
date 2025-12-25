# Running GPU Tests on SageMaker

## Quick Start

```bash
# Make script executable
chmod +x run-gpu-tests.sh

# Run GPU tests (will check CUDA, download model if needed, then run tests)
./run-gpu-tests.sh
```

## Manual Steps

### 1. Verify CUDA is Available

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2. Ensure Model Checkpoint Exists

```bash
# Check if model exists
ls -la checkpoints/openaudio-s1-mini/model.pth

# If not, download it
python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
checkpoint_dir = Path('checkpoints/openaudio-s1-mini')
checkpoint_dir.mkdir(parents=True, exist_ok=True)
snapshot_download('fishaudio/openaudio-s1-mini', local_dir=str(checkpoint_dir))
print('Model downloaded')
"
```

### 3. Set Environment Variables (Optional)

```bash
export VOICE_S1_CHECKPOINT_PATH="checkpoints/openaudio-s1-mini"
export VOICE_TTS_PROVIDER="s1_mini"
export VOICE_S1_DEVICE="cuda"
```

### 4. Run GPU Tests

```bash
# Run all GPU tests
pytest tests/ --gpu -v

# Run specific GPU test file
pytest tests/test_gpu_inference.py --gpu -v

# Run with more verbose output
pytest tests/ --gpu -v --tb=short

# Run with coverage
pytest tests/ --gpu -v --cov=src
```

## Expected Results

When running GPU tests, you should see:

- ✅ `test_engine_initializes` - Engine loads successfully
- ✅ `test_simple_generation` - Basic audio generation works
- ✅ `test_generation_parameters` - Different parameters work
- ✅ `test_longer_text` - Longer text handled correctly
- ✅ `test_streaming_generation` - Streaming works
- ✅ `test_batch_generation` - Batch processing works
- ✅ `test_audio_range` - Audio values in valid range
- ✅ `test_reproducibility_with_seed` - Generation is consistent

Plus benchmark and batch scaling tests.

## Troubleshooting

### "GPU not available"
- Check: `nvidia-smi`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Fix: Ensure PyTorch was installed with CUDA support

### "Model checkpoint not found"
- Check: `ls checkpoints/openaudio-s1-mini/model.pth`
- Fix: Download model using HuggingFace (see step 2 above)
- Or set: `export VOICE_S1_CHECKPOINT_PATH="/path/to/checkpoint"`

### "CUDA out of memory"
- Reduce batch size in tests
- Close other GPU processes
- Use smaller model if available

### Tests hang or timeout
- Check GPU utilization: `nvidia-smi`
- Increase timeout: `pytest tests/ --gpu -v --timeout=300`
- Run tests individually to isolate issues

## Test Files That Require GPU

- `test_gpu_inference.py` - Core GPU inference tests
- `test_benchmark.py` - Performance benchmarks
- `test_batch_scaling.py` - Batch size scaling tests
- `test_batching.py::TestBatcherWithGPU` - GPU batcher tests

## Notes

- GPU tests take longer to run (model loading + inference)
- First test run will be slower (model loading)
- Subsequent tests reuse the loaded model (faster)
- Tests use `--gpu` flag to skip on systems without GPU

