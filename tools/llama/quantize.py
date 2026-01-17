"""
Quantization script for Fish-Speech models.

This script uses TorchAO for efficient quantization by default.
For legacy quantization (deprecated), use --legacy flag.

Usage:
    # TorchAO quantization (recommended)
    python tools/llama/quantize.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int8
    python tools/llama/quantize.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int4 --group-size 128

    # Legacy quantization (deprecated)
    python tools/llama/quantize.py --legacy --checkpoint-path checkpoints/openaudio-s1-mini --mode int8
"""

import sys
import warnings
from pathlib import Path

import click


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--legacy", is_flag=True, help="Use legacy (deprecated) quantization")
@click.pass_context
def main(ctx, legacy):
    """Quantize a Fish-Speech model."""

    if legacy:
        warnings.warn(
            "Legacy quantization is deprecated and may produce slow/broken models. "
            "Consider using TorchAO quantization (remove --legacy flag).",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import and run legacy quantization
        from tools.llama.quantize_legacy import quantize
        # Pass remaining arguments to legacy script
        sys.argv = [sys.argv[0]] + ctx.args
        quantize()
    else:
        # Import and run TorchAO quantization
        from tools.llama.quantize_torchao import quantize
        # Pass remaining arguments to TorchAO script
        sys.argv = [sys.argv[0]] + ctx.args
        quantize()


if __name__ == "__main__":
    main()
