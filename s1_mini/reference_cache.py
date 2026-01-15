"""
S1-Mini Reference Audio Cache
=============================

This module provides optimized caching for reference audio used in voice cloning.
Reference audio needs to be encoded through the VQ encoder before it can be used
for inference. This cache stores pre-encoded references to avoid repeated encoding.

Key Features:
-------------
1. LRU cache with configurable size
2. SHA256-based deduplication
3. Memory-aware eviction
4. Pre-warming capability
5. Persistent storage option

Architecture:
-------------
    Reference Audio (bytes)
            │
            ▼
    ┌─────────────────────┐
    │   SHA256 Hash       │
    │   (Deduplication)   │
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────┐     Cache Miss
    │   LRU Cache         │─────────────────┐
    │   (Memory/Disk)     │                 │
    └─────────────────────┘                 │
            │ Cache Hit                     │
            ▼                               ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │  Pre-encoded VQ     │     │   VQ Encoder        │
    │  Tokens             │◄────│   (DAC)             │
    └─────────────────────┘     └─────────────────────┘

Usage:
------
    from s1_mini.reference_cache import ReferenceCache

    cache = ReferenceCache(max_entries=1000, max_memory_mb=2048)

    # Encode and cache reference
    tokens = cache.encode_reference(audio_bytes, decoder_model)

    # Or get from cache if exists
    tokens = cache.get_or_encode(audio_bytes, decoder_model)

    # Pre-warm cache from directory
    cache.warm_from_directory("references/")

Memory Management:
------------------
The cache monitors memory usage and evicts entries when:
1. Number of entries exceeds max_entries
2. Total memory usage exceeds max_memory_mb
3. Manual eviction is requested

Eviction follows LRU (Least Recently Used) policy.
"""

import hashlib
import io
import os
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
from loguru import logger


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheEntry:
    """
    A single cache entry for reference audio.

    Attributes:
        hash: SHA256 hash of the audio bytes
        tokens: Encoded VQ tokens (torch.Tensor)
        text: Associated text (optional)
        memory_bytes: Estimated memory usage
        access_count: Number of times accessed
    """

    hash: str
    tokens: torch.Tensor
    text: Optional[str]
    memory_bytes: int
    access_count: int = 0

    def update_access(self):
        """Update access count."""
        self.access_count += 1


@dataclass
class CacheStats:
    """
    Cache statistics for monitoring.

    Attributes:
        total_entries: Number of entries in cache
        total_memory_mb: Total memory used by cache
        hit_count: Number of cache hits
        miss_count: Number of cache misses
        hit_rate: Cache hit rate (0.0-1.0)
        eviction_count: Number of evictions
    """

    total_entries: int
    total_memory_mb: float
    hit_count: int
    miss_count: int
    hit_rate: float
    eviction_count: int


# =============================================================================
# Reference Cache
# =============================================================================


class ReferenceCache:
    """
    LRU cache for pre-encoded reference audio.

    This cache stores VQ tokens for reference audio, avoiding
    repeated encoding for frequently used references.

    Thread-safe for concurrent access.

    Attributes:
        max_entries: Maximum number of cached entries
        max_memory_mb: Maximum memory usage in MB
        persistent_path: Optional path for persistent storage

    Example:
        >>> cache = ReferenceCache(max_entries=100)
        >>> tokens = cache.get_or_encode(audio_bytes, encoder)
        >>> print(f"Cache stats: {cache.get_stats()}")
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: int = 2048,
        persistent_path: Optional[str] = None,
    ):
        """
        Initialize the reference cache.

        Args:
            max_entries: Maximum number of entries to cache
            max_memory_mb: Maximum memory usage in megabytes
            persistent_path: Optional path to save/load cache
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.persistent_path = Path(persistent_path) if persistent_path else None

        # LRU cache using OrderedDict (most recent at end)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        # Load persistent cache if exists
        if self.persistent_path and self.persistent_path.exists():
            self._load_persistent()

        logger.info(
            f"ReferenceCache initialized: max_entries={max_entries}, "
            f"max_memory_mb={max_memory_mb}"
        )

    # =========================================================================
    # Core Methods
    # =========================================================================

    def get_or_encode(
        self,
        audio_bytes: bytes,
        encoder: Any,
        text: Optional[str] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Get cached tokens or encode and cache new ones.

        This is the main method for getting reference tokens.
        It checks the cache first, and only encodes if not cached.

        Args:
            audio_bytes: Raw audio data
            encoder: VQ encoder model
            text: Associated text (optional)
            device: Target device for tokens

        Returns:
            VQ tokens as torch.Tensor
        """
        # Compute hash for lookup
        audio_hash = self._compute_hash(audio_bytes)

        # Try cache lookup
        with self._lock:
            if audio_hash in self._cache:
                self._hit_count += 1
                entry = self._cache[audio_hash]
                entry.update_access()

                # Move to end (most recently used)
                self._cache.move_to_end(audio_hash)

                logger.debug(f"Cache hit for reference {audio_hash[:8]}...")
                return entry.tokens.to(device)

        # Cache miss - need to encode
        self._miss_count += 1
        logger.debug(f"Cache miss for reference {audio_hash[:8]}..., encoding")

        # Encode reference audio
        tokens = self._encode_audio(audio_bytes, encoder, device)

        # Add to cache
        self._add_entry(audio_hash, tokens, text)

        return tokens

    def get(self, audio_hash: str) -> Optional[torch.Tensor]:
        """
        Get cached tokens by hash.

        Args:
            audio_hash: SHA256 hash of audio

        Returns:
            Cached tokens or None if not found
        """
        with self._lock:
            if audio_hash in self._cache:
                self._hit_count += 1
                entry = self._cache[audio_hash]
                entry.update_access()
                self._cache.move_to_end(audio_hash)
                return entry.tokens

        self._miss_count += 1
        return None

    def put(
        self,
        audio_bytes: bytes,
        tokens: torch.Tensor,
        text: Optional[str] = None,
    ) -> str:
        """
        Put tokens into cache.

        Args:
            audio_bytes: Original audio data (for hashing)
            tokens: Encoded VQ tokens
            text: Associated text

        Returns:
            Cache key (hash)
        """
        audio_hash = self._compute_hash(audio_bytes)
        self._add_entry(audio_hash, tokens, text)
        return audio_hash

    def contains(self, audio_bytes: bytes) -> bool:
        """Check if audio is in cache."""
        audio_hash = self._compute_hash(audio_bytes)
        with self._lock:
            return audio_hash in self._cache

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._eviction_count = 0
        logger.info("Reference cache cleared")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            total_memory = sum(e.memory_bytes for e in self._cache.values())
            total_accesses = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_accesses if total_accesses > 0 else 0

            return CacheStats(
                total_entries=total_entries,
                total_memory_mb=total_memory / (1024 * 1024),
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                hit_rate=hit_rate,
                eviction_count=self._eviction_count,
            )

    # =========================================================================
    # Pre-warming
    # =========================================================================

    def warm_from_directory(
        self,
        directory: str,
        encoder: Any,
        device: str = "cuda",
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac"),
    ) -> int:
        """
        Pre-warm cache from a directory of audio files.

        Args:
            directory: Path to directory with audio files
            encoder: VQ encoder model
            device: Target device
            extensions: Audio file extensions to process

        Returns:
            Number of files cached
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return 0

        cached_count = 0
        for ext in extensions:
            for audio_file in directory.glob(f"*{ext}"):
                try:
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()

                    # Check if already cached
                    if self.contains(audio_bytes):
                        continue

                    # Encode and cache
                    self.get_or_encode(
                        audio_bytes,
                        encoder,
                        text=audio_file.stem,
                        device=device,
                    )
                    cached_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cache {audio_file}: {e}")

        logger.info(f"Warmed cache with {cached_count} files from {directory}")
        return cached_count

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[str] = None) -> None:
        """Save cache to disk."""
        save_path = Path(path) if path else self.persistent_path
        if save_path is None:
            raise ValueError("No save path specified")

        with self._lock:
            # Convert tensors to CPU numpy for serialization
            cache_data = {}
            for hash_key, entry in self._cache.items():
                cache_data[hash_key] = {
                    "tokens": entry.tokens.cpu().numpy(),
                    "text": entry.text,
                    "access_count": entry.access_count,
                }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(cache_data, f)

        logger.info(f"Saved {len(cache_data)} cache entries to {save_path}")

    def _load_persistent(self) -> None:
        """Load cache from persistent storage."""
        if not self.persistent_path or not self.persistent_path.exists():
            return

        try:
            with open(self.persistent_path, "rb") as f:
                cache_data = pickle.load(f)

            for hash_key, data in cache_data.items():
                tokens = torch.from_numpy(data["tokens"])
                entry = CacheEntry(
                    hash=hash_key,
                    tokens=tokens,
                    text=data.get("text"),
                    memory_bytes=tokens.numel() * tokens.element_size(),
                    access_count=data.get("access_count", 0),
                )
                self._cache[hash_key] = entry

            logger.info(
                f"Loaded {len(cache_data)} cache entries from {self.persistent_path}"
            )

        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def _add_entry(
        self,
        audio_hash: str,
        tokens: torch.Tensor,
        text: Optional[str],
    ) -> None:
        """Add an entry to the cache, evicting if necessary."""
        memory_bytes = tokens.numel() * tokens.element_size()

        entry = CacheEntry(
            hash=audio_hash,
            tokens=tokens.cpu(),  # Store on CPU to save GPU memory
            text=text,
            memory_bytes=memory_bytes,
        )

        with self._lock:
            # Check if we need to evict
            while len(self._cache) >= self.max_entries:
                self._evict_lru()

            # Check memory limit
            total_memory = sum(e.memory_bytes for e in self._cache.values())
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            while total_memory + memory_bytes > max_memory_bytes and self._cache:
                self._evict_lru()
                total_memory = sum(e.memory_bytes for e in self._cache.values())

            # Add entry
            self._cache[audio_hash] = entry

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # First item in OrderedDict is least recently used
        oldest_key = next(iter(self._cache))
        del self._cache[oldest_key]
        self._eviction_count += 1
        logger.debug(f"Evicted cache entry {oldest_key[:8]}...")

    def _encode_audio(
        self,
        audio_bytes: bytes,
        encoder: Any,
        device: str,
    ) -> torch.Tensor:
        """
        Encode audio bytes to VQ tokens.

        This method handles the actual encoding using the DAC model.
        """
        import torchaudio
        import soundfile as sf

        # Load audio from bytes
        buffer = io.BytesIO(audio_bytes)

        try:
            # Try torchaudio first
            audio, sr = torchaudio.load(buffer)
        except (ImportError, RuntimeError):
            # Fallback to soundfile
            buffer.seek(0)
            audio_np, sr = sf.read(buffer)
            audio = torch.from_numpy(audio_np).float()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.T

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        # Get encoder sample rate
        if hasattr(encoder, "spec_transform"):
            target_sr = encoder.spec_transform.sample_rate
        else:
            target_sr = encoder.sample_rate

        # Resample if needed
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)

        # Encode
        audio = audio[None].to(device)
        audio_lengths = torch.tensor([audio.shape[-1]], device=device, dtype=torch.long)

        with torch.inference_mode():
            tokens, _ = encoder.encode(audio, audio_lengths)

        if tokens.ndim == 3:
            tokens = tokens[0]

        return tokens

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "ReferenceCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.persistent_path:
            self.save()

    def __len__(self) -> int:
        return len(self._cache)
