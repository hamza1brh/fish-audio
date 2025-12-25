#!/bin/bash
# Fix pyrootutils issue when fish-speech is installed as package
set -e

echo "=========================================="
echo "Fixing pyrootutils Project Root Issue"
echo "=========================================="
echo ""

# Option 1: Create .project-root file in parent directory
echo "[1/3] Creating .project-root marker in parent directory..."
PARENT_DIR=$(cd .. && pwd)
touch "$PARENT_DIR/.project-root"
echo "  Created: $PARENT_DIR/.project-root"

# Option 2: Set PYTHONPATH to include parent directory
echo ""
echo "[2/3] Setting PYTHONPATH..."
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"
echo "  PYTHONPATH includes: $PARENT_DIR"

# Option 3: Create symlink or workaround
echo ""
echo "[3/3] Verifying fix..."
python -c "
import sys
from pathlib import Path

# Check if pyrootutils can find root now
try:
    import pyrootutils
    # Try to find root from parent
    parent = Path('..').resolve()
    if (parent / '.project-root').exists():
        print('  ✓ .project-root found in parent directory')
    else:
        print('  ⚠ .project-root not found, but PYTHONPATH is set')
except Exception as e:
    print(f'  ⚠ Warning: {e}')
"

echo ""
echo "=========================================="
echo "Fix applied!"
echo ""
echo "To make this permanent, add to your shell profile:"
echo "  export PYTHONPATH=\"$PARENT_DIR:\$PYTHONPATH\""
echo ""
echo "Or run tests with:"
echo "  PYTHONPATH=\"$PARENT_DIR\" pytest tests/ --gpu -v"
echo "=========================================="

