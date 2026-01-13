# PyTorch Adaptability Test - Verification Checklist

## Test Objective

Verify that `setup_pytorch.py` automatically installs the correct PyTorch version based on GPU type across different environments.

## Test Environments

### Environment 1: Local (RTX 5070 Ti)
- **Location**: Your desktop PC
- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **Expected Compute Capability**: sm_120
- **Expected PyTorch**: Nightly (2.11.0.dev+cu128)
- **Status**: ✅ Already confirmed working

### Environment 2: AWS SageMaker
- **Location**: Cloud
- **GPU**: Tesla T4 / V100 / A100 (varies by instance)
- **Expected Compute Capability**: sm_75 (T4) / sm_70 (V100) / sm_80 (A100)
- **Expected PyTorch**: Stable (2.5.1+cu124)
- **Status**: 🔄 Ready to test

## Pre-Test Setup

### On SageMaker:

1. **Open Terminal in SageMaker Studio**
   ```bash
   # Check you're in a terminal, not a notebook
   pwd  # Should show your home directory
   ```

2. **Clone/Upload Repository**
   ```bash
   cd ~
   # Option A: Git clone
   git clone <your-repo-url>
   
   # Option B: Upload files manually
   # Use SageMaker's file upload feature
   
   cd fish-speech
   ```

3. **Run Quick Setup**
   ```bash
   chmod +x setup_sagemaker.sh
   ./setup_sagemaker.sh
   ```

## Verification Checklist

### Phase 1: Environment Detection

**On SageMaker**, verify GPU detection:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

- [ ] Command shows GPU name (e.g., "Tesla T4")
- [ ] Compute capability shown (e.g., "7.5" for sm_75)
- [ ] NOT sm_120 (that's only RTX 50-series)

### Phase 2: PyTorch Installation

**On SageMaker**, check what PyTorch was installed:

```bash
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'Type: {\"Nightly\" if \"dev\" in torch.__version__ else \"Stable\"}')"
```

- [ ] Shows PyTorch version
- [ ] Type is **"Stable"** (not Nightly)
- [ ] Version is 2.5.1 or similar (NOT 2.11.0.dev)
- [ ] Has `+cu124` or `+cu118` suffix

### Phase 3: CUDA Functionality

**On SageMaker**, test GPU operations:

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test GPU computation
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = torch.matmul(x, y)
print('✓ GPU computation works')
"
```

- [ ] CUDA Available: True
- [ ] GPU name displayed correctly
- [ ] No errors during computation
- [ ] GPU computation works

### Phase 4: Application Dependencies

**On SageMaker**, verify imports:

```bash
python -c "
import torch
import fish_speech
import streamlit
import soundfile
print('✓ All imports successful')
"
```

- [ ] No import errors
- [ ] All packages load correctly

### Phase 5: Voice Generation Test

**On SageMaker**, try generating audio:

```bash
streamlit run neymar_voice_app.py
```

- [ ] App launches without errors
- [ ] Can access UI (via ngrok or port forwarding)
- [ ] Can generate voice samples
- [ ] Audio plays correctly
- [ ] No CUDA errors

### Phase 6: Cross-Environment Comparison

**Compare outputs between environments**:

| Check | Local (RTX 5070 Ti) | SageMaker (T4/V100/A100) | Match? |
|-------|---------------------|--------------------------|--------|
| **Installation Command** | `python setup_pytorch.py` | `python setup_pytorch.py` | Should match ✅ |
| **PyTorch Type** | Nightly (dev) | Stable | Should differ ✅ |
| **PyTorch Version** | 2.11.0.dev | 2.5.1 | Should differ ✅ |
| **CUDA Works** | Yes | Yes | Should match ✅ |
| **Compute Capability** | sm_120 | sm_75/70/80 | Should differ ✅ |
| **Voice App Works** | Yes | Yes | Should match ✅ |
| **Code Changed** | No | No | Should match ✅ |

**Key Success Criteria**:
- ✅ Same command (`setup_pytorch.py`) used on both
- ✅ Different PyTorch versions installed (Nightly vs Stable)
- ✅ Both environments work correctly
- ✅ No code changes needed

## Test Results Template

Copy and fill this out after testing:

```markdown
## Test Results

### Local Environment (Baseline)
- Date: 2026-01-02
- GPU: NVIDIA GeForce RTX 5070 Ti
- Compute Cap: sm_120
- PyTorch: 2.11.0.dev20260101+cu128
- Type: Nightly
- CUDA: 12.8
- Voice Gen: ✅ Works
- Status: ✅ PASS

### SageMaker Environment
- Date: [FILL IN]
- GPU: [FILL IN - e.g., Tesla T4]
- Compute Cap: [FILL IN - e.g., sm_75]
- PyTorch: [FILL IN]
- Type: [FILL IN - Should be "Stable"]
- CUDA: [FILL IN]
- Voice Gen: [FILL IN]
- Status: [FILL IN]

### Comparison
- Same setup command: [YES/NO]
- Different PyTorch versions: [YES/NO]
- Both work correctly: [YES/NO]
- Auto-detection worked: [YES/NO]

### Conclusion
[Your assessment of whether the adaptability test passed]
```

## Common Issues & Solutions

### Issue: SageMaker installs Nightly instead of Stable

**Problem**: System didn't detect GPU correctly.

**Debug**:
```bash
python -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                       capture_output=True, text=True)
print(f'Compute Cap: {result.stdout}')
"
```

**Expected**: Should show 7.5, 7.0, or 8.0 (NOT 12.0)

### Issue: Both environments install same PyTorch

**Problem**: Not using the new setup system.

**Solution**: Make sure you ran `python setup_pytorch.py` on SageMaker, not just `poetry install` or `pip install torch`.

### Issue: Can't access Streamlit on SageMaker

**Solutions**:

1. **Use ngrok** (easiest):
```bash
pip install pyngrok
streamlit run neymar_voice_app.py &
python -c "from pyngrok import ngrok; url = ngrok.connect(8501); print(f'\n🌐 Access at: {url}\n')"
```

2. **Use SageMaker proxy**: Check SageMaker docs for port forwarding

3. **Test in notebook**: Run a quick generation test in a Jupyter cell instead of full UI

### Issue: Model download fails

**Problem**: Large model download (~2GB) on slow connection.

**Solution**: Use HuggingFace cache or pre-download models:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')
"
```

## Success Definition

The test is **SUCCESSFUL** if:

1. ✅ Local uses PyTorch Nightly (2.11.0.dev+cu128)
2. ✅ SageMaker uses PyTorch Stable (2.5.1+cu124)  
3. ✅ Same `setup_pytorch.py` command on both
4. ✅ Both environments work correctly
5. ✅ Voice generation works on both
6. ✅ No manual version management needed
7. ✅ No code changes between environments

## Next Steps After Successful Test

1. **Document results** - Fill out the test results template
2. **Save SageMaker setup** - Can use as cloud deployment
3. **Update README** - Add SageMaker instructions
4. **Share findings** - Confirm portable setup works
5. **Deploy to production** - Use same setup for prod servers

## Files to Upload to SageMaker

Minimum required files:
```
fish-speech/
├── setup_pytorch.py         # Auto-detection script
├── setup_sagemaker.sh       # SageMaker quick setup
├── neymar_voice_app.py      # Main app
├── NeymarVO.mp3            # Reference audio
├── .env                     # Environment variables (create new)
└── fish_speech/            # Python package (if not using pip install)
```

Optional but recommended:
```
├── backup_pytorch.py        # Backup utility
├── restore_pytorch.py       # Restore utility
├── SAGEMAKER_TEST_GUIDE.md # This guide
└── .pytorch_backups/        # Your local backup (for reference)
```

Good luck with the test! Let me know the results! 🚀



