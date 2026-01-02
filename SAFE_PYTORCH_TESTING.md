# Safe PyTorch Testing Guide

## Overview

Test the new PyTorch setup system safely with automatic backup and restore.

## Files

- `backup_pytorch.py` - Backup your current PyTorch
- `restore_pytorch.py` - Restore from backup
- `test_pytorch_setup.py` - Automated safe testing
- `setup_pytorch.py` - New auto-detection system

## Quick Start (Recommended)

### Option 1: Automated Testing

Run the automated test script (safest):

```bash
python test_pytorch_setup.py
```

This will:
1. ✅ Backup current PyTorch automatically
2. ✅ Test new setup
3. ✅ Verify GPU works
4. ✅ Auto-restore on failure
5. ✅ Show rollback instructions

### Option 2: Manual Step-by-Step

If you prefer manual control:

**1. Backup Current Setup**
```bash
python backup_pytorch.py
```

**2. Test New Setup**
```bash
python setup_pytorch.py
```

**3. Verify It Works**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**4. Test with Your App**
```bash
streamlit run neymar_voice_app.py
```

**5. If Issues, Restore**
```bash
python restore_pytorch.py
```

## Restore Process

### Quick Restore (Latest Backup)

```bash
python restore_pytorch.py
```

### Choose Specific Backup

```bash
# List backups
ls .pytorch_backups/

# Restore specific backup
python restore_pytorch.py .pytorch_backups/pytorch_backup_20260102_123456.json
```

### Manual Restore

Check backup file:
```bash
cat .pytorch_backups/pytorch_backup_latest.json
```

Run the restore command shown in the backup file.

## Backup Details

### What Gets Backed Up

- PyTorch version
- TorchVision version
- TorchAudio version
- CUDA version
- GPU information
- Exact restore command

### Backup Location

```
.pytorch_backups/
├── pytorch_backup_20260102_120000.json
├── pytorch_backup_20260102_123000.json
└── pytorch_backup_latest.json  (symlink to most recent)
```

### Backup Format

```json
{
  "timestamp": "20260102_123456",
  "datetime": "2026-01-02T12:34:56",
  "torch_info": {
    "torch_version": "2.11.0.dev20260101+cu128",
    "torchvision_version": "0.21.0.dev20260101+cu128",
    "torchaudio_version": "2.6.0.dev20260101+cu128",
    "cuda_available": true,
    "cuda_version": "12.8",
    "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
    "compute_cap": "sm_120"
  },
  "restore_command": "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
}
```

## Testing Checklist

After switching to new setup:

- [ ] PyTorch imports successfully
- [ ] CUDA is available (if you have GPU)
- [ ] GPU operations work
- [ ] fish-speech imports work
- [ ] Streamlit app launches
- [ ] Voice generation works
- [ ] No error messages

## Common Scenarios

### Scenario 1: New Setup Works Perfectly

```bash
python test_pytorch_setup.py  # Test
streamlit run neymar_voice_app.py  # Verify
# Keep new setup!
```

### Scenario 2: Import Errors After Switch

```bash
python setup_pytorch.py  # New PyTorch
poetry install  # Reinstall dependencies
streamlit run neymar_voice_app.py  # Test
```

### Scenario 3: New Setup Doesn't Work

```bash
python restore_pytorch.py  # Back to working version
# You're back to normal!
```

### Scenario 4: Want to Try Again Later

```bash
python restore_pytorch.py  # Restore for now
# Later...
python setup_pytorch.py  # Try again
```

## Troubleshooting

### Issue: "No backups found"

**Solution**: Create a backup first:
```bash
python backup_pytorch.py
```

### Issue: Restore fails

**Solution**: Manual restore from backup file:
```bash
# 1. Check backup
cat .pytorch_backups/pytorch_backup_latest.json

# 2. Uninstall current
pip uninstall torch torchvision torchaudio

# 3. Run restore command from backup file
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: Both versions have issues

**Solution**: Clean install of known working version:
```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Best Practices

1. **Always backup before testing**
   ```bash
   python backup_pytorch.py
   ```

2. **Keep multiple backups**
   - Backups are small (~1KB)
   - Keep history of working configurations

3. **Test immediately after switch**
   ```bash
   python -c "import torch; torch.cuda.is_available()"
   ```

4. **Document what works**
   - Note PyTorch version that works
   - Keep backup files

5. **Clean up old backups occasionally**
   ```bash
   # Keep last 5 backups
   ls -t .pytorch_backups/*.json | tail -n +6 | xargs rm
   ```

## Summary

✅ **Safe**: Always have a backup
✅ **Fast**: Restore in seconds
✅ **Automated**: test_pytorch_setup.py does everything
✅ **Reversible**: Always go back to working version
✅ **No Risk**: Test without fear

**Golden Rule**: Backup before any PyTorch changes!

```bash
python backup_pytorch.py  # Always start here
python test_pytorch_setup.py  # Safe testing
python restore_pytorch.py  # If needed
```

