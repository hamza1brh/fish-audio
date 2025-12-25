#!/bin/bash
# Fix pytest-asyncio configuration for SageMaker
set -e

echo "Fixing pytest-asyncio configuration..."

# Ensure pytest-asyncio is installed
pip install pytest-asyncio>=0.25.0 --upgrade

# Verify installation
python -c "import pytest_asyncio; print(f'pytest-asyncio version: {pytest_asyncio.__version__}')"

echo ""
echo "✓ pytest-asyncio configured"
echo ""
echo "Run tests with:"
echo "  pytest tests/ -v"

