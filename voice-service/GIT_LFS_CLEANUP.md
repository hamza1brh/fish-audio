# Git LFS Cleanup Summary

## What Was Removed

The following large files have been removed from git tracking (but kept locally):

### Model Checkpoints & Weights
- `cosyvoice_demo/pretrained_models/Fun-CosyVoice3-0.5B/*.pt` (multiple model files)
- `cosyvoice_demo/pretrained_models/Fun-CosyVoice3-0.5B/*.onnx` (ONNX models)
- `cosyvoice_demo/pretrained_models/Fun-CosyVoice3-0.5B/*.safetensors` (model weights)

### Datasets
- `neymar_Dataset_enhanced/metadata.csv`

## Updated .gitignore Patterns

The following patterns are now ignored:

```
# Model files
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
*.h5
*.hdf5
*.onnx
pretrained_models/
models/
weights/
cosyvoice_demo/pretrained_models/
cosyvoice_demo/CosyVoice/

# Datasets
datasets/
dataset/
*_Dataset*/
*_dataset*/

# Audio files
*.wav
*.mp3
*.flac
*.ogg
*.m4a

# Data files
*.npy
*.npz
```

## Next Steps

1. **Commit the changes:**
   ```bash
   git add .gitignore voice-service/.gitignore
   git commit -m "Remove large model files and datasets from git tracking"
   ```

2. **Push to remote:**
   ```bash
   git push origin master
   ```

3. **Models will be downloaded automatically:**
   - S1-mini model: Downloaded via `install.py` or `entrypoint.sh`
   - Other models: Users can download manually or via HuggingFace

## Notes

- All files are **kept locally** - only removed from git tracking
- Models will be automatically downloaded when running `install.py`
- The voice-service uses HuggingFace Hub to download models at runtime


