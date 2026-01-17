from pathlib import Path

import click
import hydra
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda", quantize_int8=False):
    """
    Load DAC decoder model.

    Args:
        config_name: Hydra config name (e.g., "modded_dac_vq")
        checkpoint_path: Path to codec.pth file
        device: Target device ("cuda" or "cpu")
        quantize_int8: If True, apply INT8 weight-only quantization to reduce VRAM
                       (~1.87 GB -> ~0.97 GB, ~48% reduction)
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        str(checkpoint_path), map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")

    # Apply INT8 quantization if requested
    if quantize_int8 and device != "cpu":
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            logger.info("Applying INT8 quantization to DAC decoder...")
            vram_before = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            quantize_(model, Int8WeightOnlyConfig())
            vram_after = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            logger.info(f"DAC INT8 quantization complete. VRAM: {vram_before:.1f} MB -> {vram_after:.1f} MB (saved {vram_before - vram_after:.1f} MB)")
        except ImportError:
            logger.warning("TorchAO not available for DAC INT8 quantization")
        except Exception as e:
            logger.warning(f"Failed to apply INT8 quantization to DAC: {e}")

    return model


@torch.no_grad()
@click.command()
@click.option(
    "--input-path",
    "-i",
    default="test.wav",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/codec.pth",
)
@click.option(
    "--device",
    "-d",
    default="cuda",
)
def main(input_path, output_path, config_name, checkpoint_path, device):
    model = load_model(config_name, checkpoint_path, device=device)

    if input_path.suffix in AUDIO_EXTENSIONS:
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio with soundfile fallback for Windows compatibility
        try:
            audio, sr = torchaudio.load(str(input_path))
        except (ImportError, RuntimeError):
            # Fallback to soundfile for Windows (torchcodec not available)
            audio_np, sr = sf.read(str(input_path))
            audio = torch.from_numpy(audio_np).float()
            
            # Ensure correct shape [channels, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.T
        
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, model.sample_rate)

        audios = audio[None].to(device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices, indices_lens = model.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    fake_audios, audio_lengths = model.decode(indices, indices_lens)
    audio_time = fake_audios.shape[-1] / model.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio - resample to 24kHz for correct playback
    # The model outputs at 44.1kHz but should be played at 24kHz
    fake_audio = fake_audios[0, 0].float().cpu()
    target_sr = 24000
    if model.sample_rate != target_sr:
        logger.info(f"Resampling from {model.sample_rate}Hz to {target_sr}Hz")
        fake_audio = torchaudio.functional.resample(
            fake_audio.unsqueeze(0), 
            model.sample_rate, 
            target_sr
        ).squeeze(0)
    
    sf.write(output_path, fake_audio.numpy(), target_sr)
    logger.info(f"Saved audio to {output_path} at {target_sr}Hz")


if __name__ == "__main__":
    main()
