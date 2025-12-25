"""
Custom dataset that uses the interleave format matching inference.
This fixes the format mismatch between training and inference.
"""

import random
from pathlib import Path
from random import Random
from typing import Optional

import torch
from loguru import logger as log
from torch.utils.data import IterableDataset

from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.tokenizer import FishTokenizer
from fish_speech.utils.braceexpand import braceexpand
from fish_speech.datasets.semantic import split_by_rank_worker

CODEBOOK_PAD_TOKEN_ID = 0


def clean_text(text: str) -> str:
    """Clean text by removing special characters."""
    import re
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class InterleaveFormatDataset(IterableDataset):
    """
    Dataset that produces training samples in the interleave format.
    
    This matches the format used during inference:
    <|interleave|><|speaker:0|> TEXT VQ_TOKENS <|im_end|>
    """

    def __init__(
        self,
        proto_files: list[str],
        tokenizer: FishTokenizer,
        seed: int = 42,
        max_length: int = 4096,
        num_codebooks: int = 10,
        samples_per_group: int = 1,  # How many sentences to combine per sample
    ):
        super().__init__()
        self.proto_files = proto_files
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length
        self.num_codebooks = num_codebooks
        self.samples_per_group = samples_per_group
        self.sentences = None

    def init_data(self):
        if self.sentences is not None:
            return

        expanded_proto_files = []
        for filename in self.proto_files:
            for i in braceexpand(filename):
                i = Path(i)
                if i.is_file():
                    expanded_proto_files.append(i)
                elif i.is_dir():
                    expanded_proto_files.extend(i.rglob("*.proto"))
                    expanded_proto_files.extend(i.rglob("*.protos"))

        expanded_proto_files = sorted(expanded_proto_files)
        Random(self.seed).shuffle(expanded_proto_files)

        if not expanded_proto_files:
            raise ValueError(f"No .proto or .protos files found in {self.proto_files}")

        self.sentences = []
        shard_proto_files = split_by_rank_worker(expanded_proto_files)
        log.info(f"Reading {len(shard_proto_files)} / {len(expanded_proto_files)} files")

        for filename in shard_proto_files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    for sentence in text_data.sentences:
                        self.sentences.append(sentence)

        log.info(f"Read total {len(self.sentences)} sentences")
        Random(self.seed).shuffle(self.sentences)

    def __iter__(self):
        while True:
            sample = self.get_sample()
            if sample is not None:
                yield sample

    def get_sample(self):
        if self.sentences is None:
            self.init_data()

        # Pick random sentence(s)
        idx = random.randint(0, len(self.sentences) - 1)
        sentence = self.sentences[idx]

        if not sentence.texts or not sentence.semantics:
            return None

        text = clean_text(random.choice(sentence.texts))
        vq_codes = torch.tensor(
            [list(sem.values) for sem in sentence.semantics],
            dtype=torch.int32
        )

        # Create sequence in interleave format
        seq = ContentSequence(modality='interleave')
        seq.append(
            [TextPart(text=text), VQPart(codes=vq_codes, cal_loss=True)],
            add_end=True,
            speaker=0,
        )

        # Encode
        encoded = seq.encode(self.tokenizer, add_shift=True)
        
        # Build tokens tensor
        tokens_raw = encoded.tokens
        tokens = torch.zeros((self.num_codebooks + 1, len(tokens_raw)), dtype=torch.long)
        tokens[0] = tokens_raw

        vq_parts = encoded.vq_parts
        if vq_parts and len(vq_parts) > 0:
            vq_parts = torch.cat(vq_parts, dim=1).long()
            tokens[1:, encoded.vq_mask_tokens] = vq_parts
            tokens[1:, ~encoded.vq_mask_tokens] = CODEBOOK_PAD_TOKEN_ID

        # Build labels tensor
        labels_raw = encoded.labels
        labels = torch.full((self.num_codebooks + 1, len(labels_raw)), -100, dtype=torch.long)
        labels[0] = labels_raw
        if vq_parts is not None and len(vq_parts) > 0:
            labels[1:, encoded.vq_mask_labels] = vq_parts
            labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        return {"tokens": tokens, "labels": labels}

