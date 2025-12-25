# Installation Guide

## Important: Two Different Projects

- **Parent `fish-speech`**: Uses `uv` or `pip` (PEP 621 format)
- **`voice-service`**: Uses Poetry

## Installing Parent Project (fish-speech)

**DO NOT use Poetry** - it doesn't support PEP 621 format properly.

```bash
# Option 1: Use uv (recommended)
uv pip install -e .

# Option 2: Use pip
pip install -e .
```

## Installing voice-service

**Use Poetry** (or the install scripts):

```bash
cd voice-service

# Option 1: Use install script (recommended)
.\install.ps1  # Windows
./install.sh   # Linux/Mac

# Option 2: Manual Poetry install
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
poetry install --no-interaction

# Option 3: Use pip only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Why Poetry Fails in Parent Directory

The parent `fish-speech` uses PEP 621 format (`[project]`), not Poetry format (`[tool.poetry]`). Poetry misinterprets the Python version requirement and throws errors.

**Solution:** Always use `uv` or `pip` for the parent project, Poetry only for `voice-service`.





