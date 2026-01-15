"""
S1-Mini ONNX Model Exporter
===========================

Exports the Fish-Speech DualARTransformer to ONNX format for
inference without Triton dependency.

The exporter creates wrapper models that:
1. Take explicit KV cache as input/output (not internal state)
2. Use manual attention instead of SDPA (ONNX compatible)
3. Handle prefill vs decode phases separately

Usage:
------
    from s1_mini.onnx_export import ONNXExporter

    exporter = ONNXExporter(
        checkpoint_path="checkpoints/openaudio-s1-mini",
        output_dir="checkpoints/openaudio-s1-mini/onnx",
    )
    exporter.export_all()
"""

import os
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    max_batch_size: int = 1
    max_seq_len: int = 2048
    num_codebooks: int = 8
    opset_version: int = 17
    use_fp16: bool = True


# =============================================================================
# ONNX-Compatible Attention Implementation
# =============================================================================


def onnx_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """
    ONNX-compatible scaled dot-product attention.

    This replaces F.scaled_dot_product_attention which is not
    fully ONNX exportable (especially with Flash Attention).

    Args:
        query: [B, H, Q, D]
        key: [B, H, K, D]
        value: [B, H, K, D]
        attn_mask: Optional mask [B, H, Q, K] or [1, 1, Q, K]
        dropout_p: Dropout probability (ignored in export)
        is_causal: Whether to apply causal masking

    Returns:
        Attention output [B, H, Q, D]
    """
    L, S = query.size(-2), key.size(-2)
    scale = 1.0 / math.sqrt(query.size(-1))

    # QK^T
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        # Create causal mask on-the-fly
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))

    # Apply attention mask if provided
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask

    # Softmax
    attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(query.dtype)

    # Attention @ Value
    return torch.matmul(attn_weight, value)


def apply_rotary_emb_onnx(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary position embeddings (ONNX compatible).

    This is the same as the original but ensures all ops are ONNX exportable.

    Args:
        x: Input tensor [B, S, H, D]
        freqs_cis: Rotary frequencies [S, D//2, 2] or [1, S, 1, D//2, 2]

    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape to separate real/imaginary pairs
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # Broadcast freqs_cis to match x
    if freqs_cis.dim() == 3:
        freqs_cis = freqs_cis.view(1, freqs_cis.size(0), 1, freqs_cis.size(1), 2)

    # Apply rotation
    x_out = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)

    # Reshape back
    x_out = x_out.flatten(-2)
    return x_out.type_as(x)


# =============================================================================
# ONNX Wrapper Models
# =============================================================================


class ONNXAttentionLayer(nn.Module):
    """
    ONNX-exportable attention layer with explicit KV cache I/O.
    """

    def __init__(
        self,
        original_attention: nn.Module,
        layer_idx: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len

        # Copy weights from original attention
        self.wqkv = original_attention.wqkv
        self.wo = original_attention.wo

        # Copy config
        self.n_head = original_attention.n_head
        self.n_local_heads = original_attention.n_local_heads
        self.head_dim = original_attention.head_dim
        self.dim = original_attention.dim

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_position: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with explicit KV cache.

        Args:
            x: Input [B, S, D]
            freqs_cis: Rotary embeddings [S, head_dim//2, 2]
            mask: Attention mask [B, 1, S, K]
            k_cache: Key cache [B, H, max_seq_len, head_dim]
            v_cache: Value cache [B, H, max_seq_len, head_dim]
            cache_position: Position to update in cache [S]

        Returns:
            output: Attention output [B, S, D]
            new_k_cache: Updated key cache
            new_v_cache: Updated value cache
        """
        bsz, seqlen, _ = x.shape

        # Project Q, K, V
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        # Reshape for multi-head attention
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb_onnx(q, freqs_cis)
        k = apply_rotary_emb_onnx(k, freqs_cis)

        # Transpose to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update KV cache using scatter
        # cache_position contains indices where to place new K, V
        new_k_cache = k_cache.clone()
        new_v_cache = v_cache.clone()

        # For each position in cache_position, update the cache
        for i, pos in enumerate(cache_position):
            new_k_cache[:, :, pos, :] = k[:, :, i, :]
            new_v_cache[:, :, pos, :] = v[:, :, i, :]

        # Use full cache for attention
        k_full = new_k_cache
        v_full = new_v_cache

        # Repeat K, V for grouped query attention
        if self.n_head != self.n_local_heads:
            k_full = k_full.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v_full = v_full.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        # Compute attention (ONNX compatible)
        y = onnx_scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)

        # Reshape and output projection
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)
        output = self.wo(y)

        return output, new_k_cache, new_v_cache


class ONNXTransformerBlock(nn.Module):
    """ONNX-exportable transformer block."""

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Wrap attention with explicit cache
        self.attention = ONNXAttentionLayer(
            original_block.attention,
            layer_idx,
            max_seq_len,
        )

        # Copy other components directly
        self.feed_forward = original_block.feed_forward
        self.ffn_norm = original_block.ffn_norm
        self.attention_norm = original_block.attention_norm

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_position: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with explicit KV cache.

        Returns:
            output: Block output
            new_k_cache: Updated key cache
            new_v_cache: Updated value cache
        """
        # Pre-norm attention
        h = x + self._attention_with_cache(
            self.attention_norm(x),
            freqs_cis,
            mask,
            k_cache,
            v_cache,
            cache_position,
        )[0]

        # Get updated cache
        _, new_k_cache, new_v_cache = self._attention_with_cache(
            self.attention_norm(x),
            freqs_cis,
            mask,
            k_cache,
            v_cache,
            cache_position,
        )

        # Pre-norm FFN
        out = h + self.feed_forward(self.ffn_norm(h))

        return out, new_k_cache, new_v_cache

    def _attention_with_cache(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_position: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.attention(x, freqs_cis, mask, k_cache, v_cache, cache_position)


class ONNXSlowTransformerPrefill(nn.Module):
    """
    ONNX wrapper for slow transformer prefill phase.

    Processes the full input sequence and outputs KV cache.
    """

    def __init__(self, original_model: nn.Module, config: ExportConfig):
        super().__init__()
        self.config = config

        # Copy embedding layer
        self.embeddings = original_model.embeddings

        # Copy and wrap transformer blocks
        self.layers = nn.ModuleList([
            ONNXTransformerBlock(block, i, config.max_seq_len)
            for i, block in enumerate(original_model.layers)
        ])

        # Copy final components
        self.norm = original_model.norm
        self.output = original_model.output

        # Copy precomputed constants
        self.register_buffer('freqs_cis', original_model.freqs_cis.clone())
        self.register_buffer('causal_mask', original_model.causal_mask.clone())

        # Store dimensions
        self.n_layers = len(self.layers)
        self.n_heads = original_model.config.n_local_heads
        self.head_dim = original_model.config.head_dim

    def forward(
        self,
        input_ids: Tensor,  # [B, num_codebooks+1, seq_len]
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Prefill forward pass.

        Args:
            input_ids: Input token IDs [B, num_codebooks+1, seq_len]

        Returns:
            logits: Output logits [B, 1, vocab_size]
            hidden_states: Hidden representation [B, 1, dim]
            k_caches: List of key caches per layer
            v_caches: List of value caches per layer
        """
        B, C, T = input_ids.shape

        # Embed tokens
        x = self.embeddings(input_ids)  # [B, T, D]

        # Get position indices
        positions = torch.arange(T, device=input_ids.device, dtype=torch.long)

        # Get rotary embeddings and mask for these positions
        freqs_cis = self.freqs_cis[positions]
        mask = self.causal_mask[None, None, :T, :T]

        # Initialize empty KV caches
        k_caches = []
        v_caches = []

        # Process through layers
        for layer in self.layers:
            # Create empty cache for this layer
            k_cache = torch.zeros(
                B, self.n_heads, self.config.max_seq_len, self.head_dim,
                dtype=x.dtype, device=x.device
            )
            v_cache = torch.zeros(
                B, self.n_heads, self.config.max_seq_len, self.head_dim,
                dtype=x.dtype, device=x.device
            )

            x, k_cache, v_cache = layer(x, freqs_cis, mask, k_cache, v_cache, positions)
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        # Final norm and output
        x = self.norm(x)
        logits = self.output(x[:, -1:, :])  # Only last position

        return logits, x[:, -1:, :], k_caches, v_caches


class ONNXSlowTransformerDecode(nn.Module):
    """
    ONNX wrapper for slow transformer decode phase.

    Processes single token with KV cache.
    """

    def __init__(self, original_model: nn.Module, config: ExportConfig):
        super().__init__()
        self.config = config

        # Copy embedding layer
        self.embeddings = original_model.embeddings

        # Wrap transformer blocks (same as prefill)
        self.layers = nn.ModuleList([
            ONNXTransformerBlock(block, i, config.max_seq_len)
            for i, block in enumerate(original_model.layers)
        ])

        # Copy final components
        self.norm = original_model.norm
        self.output = original_model.output

        # Copy precomputed constants
        self.register_buffer('freqs_cis', original_model.freqs_cis.clone())
        self.register_buffer('causal_mask', original_model.causal_mask.clone())

        self.n_layers = len(self.layers)
        self.n_heads = original_model.config.n_local_heads
        self.head_dim = original_model.config.head_dim

    def forward(
        self,
        input_ids: Tensor,  # [B, num_codebooks+1, 1]
        position: Tensor,   # [1] - current position
        k_caches: List[Tensor],  # List of [B, H, max_seq_len, D]
        v_caches: List[Tensor],  # List of [B, H, max_seq_len, D]
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Decode forward pass (single token).

        Args:
            input_ids: Single token [B, num_codebooks+1, 1]
            position: Current sequence position [1]
            k_caches: Key caches from previous steps
            v_caches: Value caches from previous steps

        Returns:
            logits: Output logits [B, 1, vocab_size]
            hidden_states: Hidden representation [B, 1, dim]
            new_k_caches: Updated key caches
            new_v_caches: Updated value caches
        """
        B = input_ids.shape[0]

        # Embed single token
        x = self.embeddings(input_ids)  # [B, 1, D]

        # Get rotary embeddings and mask for this position
        freqs_cis = self.freqs_cis[position]
        mask = self.causal_mask[None, None, position, :self.config.max_seq_len]

        # Process through layers with cache
        new_k_caches = []
        new_v_caches = []

        for i, layer in enumerate(self.layers):
            x, k_cache, v_cache = layer(
                x, freqs_cis, mask,
                k_caches[i], v_caches[i],
                position
            )
            new_k_caches.append(k_cache)
            new_v_caches.append(v_cache)

        # Final norm and output
        x = self.norm(x)
        logits = self.output(x)

        return logits, x, new_k_caches, new_v_caches


# =============================================================================
# ONNX Exporter
# =============================================================================


class ONNXExporter:
    """
    Exports Fish-Speech models to ONNX format.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/openaudio-s1-mini",
        output_dir: Optional[str] = None,
        config: Optional[ExportConfig] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir or checkpoint_path) / "onnx"
        self.config = config or ExportConfig()

        self._original_model = None

    def _load_original_model(self):
        """Load the original PyTorch model."""
        if self._original_model is not None:
            return self._original_model

        logger.info(f"Loading original model from {self.checkpoint_path}")

        from fish_speech.models.text2semantic.llama import (
            DualARTransformer,
            NaiveTransformer,
        )

        # Find model checkpoint
        model_path = self.checkpoint_path / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        # Determine model class from checkpoint
        if "dual_ar" in str(model_path).lower() or True:  # Default to DualAR
            # Get config from checkpoint or use defaults
            from fish_speech.models.text2semantic.llama import BaseModelArgs

            model_args = BaseModelArgs()
            model = DualARTransformer(model_args)

        # Load weights
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        self._original_model = model
        logger.info("Model loaded successfully")

        return model

    def export_slow_prefill(self) -> str:
        """
        Export slow transformer prefill model to ONNX.

        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting slow transformer (prefill phase)...")

        model = self._load_original_model()
        wrapper = ONNXSlowTransformerPrefill(model, self.config)
        wrapper.eval()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "slow_prefill.onnx"

        # Create sample inputs
        B = 1
        C = self.config.num_codebooks + 1
        T = 64  # Sample sequence length

        sample_input = torch.randint(0, 1000, (B, C, T), dtype=torch.long)

        # Export
        logger.info(f"Exporting to {output_path}...")

        try:
            torch.onnx.export(
                wrapper,
                (sample_input,),
                str(output_path),
                input_names=["input_ids"],
                output_names=["logits", "hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch", 2: "seq_len"},
                    "logits": {0: "batch"},
                    "hidden_states": {0: "batch"},
                },
                opset_version=self.config.opset_version,
                do_constant_folding=True,
            )
            logger.info(f"Exported slow prefill to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    def export_all(self) -> Dict[str, str]:
        """
        Export all models to ONNX.

        Returns:
            Dictionary of model names to ONNX paths
        """
        logger.info("Starting ONNX export...")

        paths = {}

        # For now, just export what we can
        # Full implementation would include decode and fast models

        try:
            paths["slow_prefill"] = self.export_slow_prefill()
        except Exception as e:
            logger.error(f"Failed to export slow_prefill: {e}")

        logger.info(f"Export complete. Exported {len(paths)} models.")
        return paths


def check_onnx_model(model_path: str) -> dict:
    """
    Validate an exported ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        Dictionary with validation results
    """
    try:
        import onnx
        from onnx import checker

        model = onnx.load(model_path)
        checker.check_model(model)

        return {
            "valid": True,
            "inputs": [i.name for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
            "opset": model.opset_import[0].version,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


if __name__ == "__main__":
    # Test export
    exporter = ONNXExporter()
    paths = exporter.export_all()
    print(f"Exported models: {paths}")
