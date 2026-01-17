"""
TorchAO-based quantization for Fish-Speech models.

This script uses TorchAO (PyTorch's official quantization library) to create
quantized model checkpoints with optimized CUDA kernels for fast inference.

Supported modes:
- int8: INT8 weight-only quantization (~50% memory reduction)
- int4: INT4 weight-only quantization (~75% memory reduction)

Usage:
    python tools/llama/quantize_torchao.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int8
    python tools/llama/quantize_torchao.py --checkpoint-path checkpoints/openaudio-s1-mini --mode int4 --group-size 128

Requirements:
    pip install torchao>=0.15.0
"""

import datetime
import json
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import click
import torch
from loguru import logger

try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    logger.warning("TorchAO not available. Install with: pip install torchao>=0.15.0")


def generate_folder_name() -> str:
    """Generate timestamp-based folder name."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB."""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size / 1e6


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/openaudio-s1-mini",
    help="Path to the model checkpoint directory",
)
@click.option(
    "--mode",
    type=click.Choice(["int8", "int4"]),
    default="int8",
    help="Quantization mode: int8 or int4",
)
@click.option(
    "--group-size",
    type=click.Choice(["32", "64", "128", "256"]),
    default="128",
    help="Group size for int4 quantization",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (auto-generated if not specified)",
)
def quantize(
    checkpoint_path: Path,
    mode: str,
    group_size: str,
    output_dir: Path,
) -> None:
    """Quantize a Fish-Speech model using TorchAO."""

    if not TORCHAO_AVAILABLE:
        raise click.ClickException(
            "TorchAO is required for quantization. Install with: pip install torchao>=0.15.0"
        )

    group_size_int = int(group_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"TorchAO Quantization")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Group size: {group_size_int}" if mode == "int4" else "  Group size: N/A")
    logger.info(f"  Device: {device}")
    logger.info(f"  Checkpoint: {checkpoint_path}")

    # Import model classes
    from fish_speech.models.text2semantic.llama import DualARTransformer, BaseModelArgs
    from fish_speech.tokenizer import FishTokenizer

    # Load model configuration and tokenizer
    logger.info("Loading model configuration...")
    t0 = time.time()

    config = BaseModelArgs.from_pretrained(str(checkpoint_path))
    tokenizer = FishTokenizer.from_pretrained(checkpoint_path)

    # Create model
    logger.info("Creating model...")
    model = DualARTransformer(config, tokenizer=tokenizer)

    # Load weights
    logger.info("Loading weights...")
    weights_path = checkpoint_path / "model.pth"
    weights = torch.load(
        str(weights_path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )

    # Handle state dict format
    if "state_dict" in weights:
        logger.info("Extracting weights from state_dict")
        weights = weights["state_dict"]

    if next(iter(weights.keys())).startswith("model."):
        logger.info("Removing 'model.' prefix from keys")
        new_weights = OrderedDict()
        for k, v in weights.items():
            new_weights[k.replace("model.", "")] = v
        weights = new_weights

    # Remove audio related weights
    for k in list(weights.keys()):
        if "audio_" in k:
            weights.pop(k)

    # Load state dict
    err = model.load_state_dict(weights, strict=False, assign=True)
    logger.info(f"Loaded weights: {err}")

    # Move to device and set precision
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")

    # Get original model size
    original_size = get_model_size_mb(model)
    logger.info(f"Original model size: {original_size:.2f} MB")

    # Apply quantization
    logger.info(f"Applying {mode} quantization...")
    t1 = time.time()

    if mode == "int8":
        quantize_(model, Int8WeightOnlyConfig())
        suffix = "int8-torchao"
    else:  # int4
        # Use version 1 which doesn't require fbgemm-gpu-genai
        # Version 2 requires fbgemm which may not be available on all platforms
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*version 1.*deprecated.*")
            quantize_(model, Int4WeightOnlyConfig(group_size=group_size_int, version=1))
        suffix = f"int4-g{group_size_int}-torchao"

    quant_time = time.time() - t1
    logger.info(f"Quantization completed in {quant_time:.2f}s")

    # Get quantized model size (estimate from state dict)
    quantized_size = get_model_size_mb(model)
    logger.info(f"Quantized model size (in memory): {quantized_size:.2f} MB")

    # Determine output directory
    if output_dir is None:
        timestamp = generate_folder_name()
        output_dir = checkpoint_path.parent / f"{checkpoint_path.name}-{suffix}-{timestamp}"

    logger.info(f"Saving quantized model to {output_dir}...")

    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-model files (config, tokenizer, etc.)
    for item in checkpoint_path.iterdir():
        if item.name not in ("model.pth", ".cache", ".git", "__pycache__"):
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, output_dir / item.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Save quantized model state dict
    quantized_state_dict = model.state_dict()
    model_path = output_dir / "model.pth"
    torch.save(quantized_state_dict, str(model_path))

    # Get actual file size
    file_size = model_path.stat().st_size / 1e6
    logger.info(f"Saved model file size: {file_size:.2f} MB")

    # Update config.json with quantization metadata
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
    else:
        config_data = {}

    config_data["quantization"] = {
        "method": "torchao",
        "mode": mode,
        "group_size": group_size_int if mode == "int4" else None,
        "torchao_version": "0.15.0",
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4, sort_keys=True)

    # Summary
    total_time = time.time() - t0
    compression_ratio = original_size / file_size if file_size > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Quantization Summary")
    logger.info("=" * 60)
    logger.info(f"  Mode:              {mode}")
    logger.info(f"  Group size:        {group_size_int if mode == 'int4' else 'N/A'}")
    logger.info(f"  Original size:     {original_size:.2f} MB")
    logger.info(f"  Quantized size:    {file_size:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"  Total time:        {total_time:.2f}s")
    logger.info(f"  Output path:       {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    quantize()
