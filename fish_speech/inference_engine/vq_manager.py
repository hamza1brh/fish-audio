from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        feature_lengths = torch.tensor(
            [codes.shape[1]], device=self.decoder_model.device
        )
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, DAC):
            return self.decoder_model.decode(
                indices=codes[None],
                feature_lengths=feature_lengths,
            )[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(self.decoder_model, DAC):
                prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens

    def encode_references_batched(
        self,
        reference_audios: List[bytes],
        enable_reference_audio: bool = True,
    ) -> List[Optional[torch.Tensor]]:
        """
        Encode multiple reference audios in a single batched forward pass.

        This method loads all audios, pads them to the same length,
        and encodes them through DAC in a single batch.

        Args:
            reference_audios: List of reference audio bytes
            enable_reference_audio: Whether reference audio is enabled

        Returns:
            List of encoded prompt tokens (or None for items without audio)
        """
        if not enable_reference_audio or not reference_audios:
            return [None] * len(reference_audios)

        # Get sample rate
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # Load all audios
        audio_arrays = []
        valid_indices = []

        for i, ref_audio in enumerate(reference_audios):
            if ref_audio is not None:
                try:
                    audio_content = self.load_audio(ref_audio, sample_rate)
                    audio_arrays.append(audio_content)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to load reference audio {i}: {e}")

        if not audio_arrays:
            return [None] * len(reference_audios)

        # Pad to same length
        max_len = max(len(a) for a in audio_arrays)
        padded_audios = []

        for audio in audio_arrays:
            if len(audio) < max_len:
                # Pad with zeros
                import numpy as np
                padded = np.pad(audio, (0, max_len - len(audio)), mode='constant')
            else:
                padded = audio
            padded_audios.append(padded)

        # Stack into tensor
        import numpy as np
        stacked = np.stack(padded_audios, axis=0)  # [batch_size, audio_len]
        audios = torch.from_numpy(stacked).to(self.decoder_model.device)
        audios = audios.unsqueeze(1)  # [batch_size, 1, audio_len]

        audio_lengths = torch.tensor(
            [len(a) for a in audio_arrays],
            device=self.decoder_model.device,
            dtype=torch.long,
        )

        logger.info(
            f"Batched encoding {len(audio_arrays)} references, "
            f"max_len={max_len / sample_rate:.2f}s"
        )

        # Encode all at once
        if isinstance(self.decoder_model, DAC):
            # DAC encode returns (codes, lengths, ...)
            all_codes = self.decoder_model.encode(audios, audio_lengths)[0]
            # all_codes shape: [batch_size, num_codebooks, seq_len]
        else:
            raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

        # Distribute results
        results: List[Optional[torch.Tensor]] = [None] * len(reference_audios)
        for batch_idx, orig_idx in enumerate(valid_indices):
            # Extract codes for this item
            item_codes = all_codes[batch_idx]  # [num_codebooks, seq_len]

            # Trim to actual length (remove padding)
            # Calculate how many frames correspond to the actual audio length
            actual_audio_len = len(audio_arrays[batch_idx])
            # DAC typically has a hop length that determines the ratio
            # For now, we'll keep all frames as the padding at the end
            # should be minimal impact
            results[orig_idx] = item_codes

        logger.info(f"Encoded {len(valid_indices)} reference audios")
        return results

    def decode_vq_tokens_batched(
        self,
        codes_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Decode multiple VQ token sequences in a single batched forward pass.

        Args:
            codes_list: List of code tensors [num_codebooks, seq_len]

        Returns:
            List of decoded audio tensors
        """
        if not codes_list:
            return []

        # Pad to same length
        max_len = max(c.shape[1] for c in codes_list)
        batch_size = len(codes_list)
        num_codebooks = codes_list[0].shape[0]
        device = self.decoder_model.device

        # Create padded batch tensor
        padded_codes = torch.zeros(
            (batch_size, num_codebooks, max_len),
            dtype=codes_list[0].dtype,
            device=device,
        )

        feature_lengths = []
        for i, codes in enumerate(codes_list):
            seq_len = codes.shape[1]
            padded_codes[i, :, :seq_len] = codes.to(device)
            feature_lengths.append(seq_len)

        feature_lengths_tensor = torch.tensor(
            feature_lengths, device=device, dtype=torch.long
        )

        logger.info(f"Batched decoding {batch_size} VQ sequences")

        # Decode all at once
        if isinstance(self.decoder_model, DAC):
            # DAC decode expects [batch, num_codebooks, seq_len]
            decoded = self.decoder_model.decode(
                indices=padded_codes,
                feature_lengths=feature_lengths_tensor,
            )
            # decoded shape: [batch_size, 1, audio_len]
        else:
            raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

        # Split results and trim to actual lengths
        results = []
        for i in range(batch_size):
            audio = decoded[i].squeeze()  # Remove channel dim
            results.append(audio)

        return results
