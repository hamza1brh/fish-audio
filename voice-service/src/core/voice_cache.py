"""Pre-encoded voice token cache."""

from pathlib import Path

import numpy as np
from loguru import logger


class VoiceCache:
    """Cache for pre-encoded voice tokens (.npy files).

    Loads voice tokens from disk at startup for fast inference.
    """

    def __init__(self, voices_dir: Path | str = "voices") -> None:
        self._dir = Path(voices_dir)
        self._cache: dict[str, tuple[np.ndarray, str]] = {}

    def load_all(self) -> None:
        """Load all voice tokens from directory."""
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created voices directory: {self._dir}")
            return

        count = 0
        for npy_file in self._dir.glob("*.npy"):
            voice_id = npy_file.stem
            try:
                tokens = np.load(npy_file)
                txt_file = npy_file.with_suffix(".txt")
                transcript = txt_file.read_text().strip() if txt_file.exists() else ""
                self._cache[voice_id] = (tokens, transcript)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load voice {voice_id}: {e}")

        logger.info(f"Loaded {count} voices from {self._dir}")

    def get(self, voice_id: str) -> tuple[np.ndarray, str] | None:
        """Get voice tokens and transcript by ID."""
        return self._cache.get(voice_id)

    def add(self, voice_id: str, tokens: np.ndarray, transcript: str) -> None:
        """Add voice to cache and persist to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)

        npy_path = self._dir / f"{voice_id}.npy"
        txt_path = self._dir / f"{voice_id}.txt"

        np.save(npy_path, tokens)
        txt_path.write_text(transcript)

        self._cache[voice_id] = (tokens, transcript)
        logger.info(f"Saved voice: {voice_id}")

    def remove(self, voice_id: str) -> bool:
        """Remove voice from cache and disk."""
        if voice_id not in self._cache:
            return False

        npy_path = self._dir / f"{voice_id}.npy"
        txt_path = self._dir / f"{voice_id}.txt"

        if npy_path.exists():
            npy_path.unlink()
        if txt_path.exists():
            txt_path.unlink()

        del self._cache[voice_id]
        return True

    def list_voices(self) -> list[str]:
        """List all cached voice IDs."""
        return list(self._cache.keys())

    def has_voice(self, voice_id: str) -> bool:
        """Check if voice exists."""
        return voice_id in self._cache



