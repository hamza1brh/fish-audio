from dataclasses import dataclass

import loralib as lora


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0


def _replace_embedding_with_lora(model, attr_name, lora_config):
    """Replace an embedding layer with a LoRA embedding while preserving weights."""
    old_embedding = getattr(model, attr_name)
    new_embedding = lora.Embedding(
        num_embeddings=old_embedding.num_embeddings,
        embedding_dim=old_embedding.embedding_dim,
        padding_idx=old_embedding.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )
    # Copy the pretrained weights
    new_embedding.weight.data = old_embedding.weight.data.clone()
    setattr(model, attr_name, new_embedding)


def _replace_linear_with_lora(module, attr_name, lora_config):
    """Replace a linear layer with a LoRA linear while preserving weights."""
    old_linear = getattr(module, attr_name)
    new_linear = lora.Linear(
        in_features=old_linear.in_features,
        out_features=old_linear.out_features,
        bias=old_linear.bias is not None,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
    )
    # Copy the pretrained weights
    new_linear.weight.data = old_linear.weight.data.clone()
    if old_linear.bias is not None:
        new_linear.bias.data = old_linear.bias.data.clone()
    setattr(module, attr_name, new_linear)


def setup_lora(model, lora_config):
    # Replace the embedding layers with LoRA layers while preserving weights
    _replace_embedding_with_lora(model, "embeddings", lora_config)
    _replace_embedding_with_lora(model, "codebook_embeddings", lora_config)

    # Replace output layer with a LoRA layer
    linears = [(model, "output")]

    # Replace all linear layers with LoRA layers
    for layer in model.layers:
        linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        linears.extend(
            [
                (layer.feed_forward, "w1"),
                (layer.feed_forward, "w2"),
                (layer.feed_forward, "w3"),
            ]
        )

    if hasattr(model, "fast_layers"):
        _replace_embedding_with_lora(model, "fast_embeddings", lora_config)

        # Dual-AR model
        linears.append((model, "fast_output"))

        for layer in model.fast_layers:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

    for module, layer in linears:
        _replace_linear_with_lora(module, layer, lora_config)

    # Mark only the LoRA layers as trainable
    lora.mark_only_lora_as_trainable(model, bias="none")


def get_merged_state_dict(model):
    # This line will merge the state dict of the model and the LoRA parameters
    model.eval()

    # Then we need to remove the LoRA parameters from the state dict
    state_dict = model.state_dict()
    for name in list(state_dict.keys()):
        if "lora" in name:
            state_dict.pop(name)

    return state_dict
