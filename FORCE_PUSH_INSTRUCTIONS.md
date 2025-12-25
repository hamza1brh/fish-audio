# Force Push Instructions

## What Was Done

1. ✅ Removed `cosyvoice_demo/pretrained_models/` files from git history using `git filter-branch`
2. ✅ Removed LFS tracking for `*.pt`, `*.pth`, `*.safetensors`, `*.onnx` files
3. ✅ Cleaned up LFS cache (8 objects pruned)
4. ✅ Ran garbage collection to clean up git history

## Next Steps - Force Push Required

Since we rewrote git history, you **must** use `--force` to push:

```bash
git push --force origin master
```

**⚠️ WARNING:** Force pushing rewrites remote history. Make sure:
- You're the only one working on this branch
- You've backed up if needed
- You understand this will overwrite remote history

## Alternative: If Force Push Fails

If GitHub still rejects due to LFS budget, you may need to:

1. **Create a new repository** and push fresh:
   ```bash
   git remote set-url origin <new-repo-url>
   git push -u origin master
   ```

2. **Or contact GitHub support** to increase LFS quota

## Verification

After pushing, verify no LFS files:
```bash
git lfs ls-files
```

Should return nothing.

