"""HuggingFace model downloader for S1 Mini checkpoints."""

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger


def download_s1_mini_model(
    repo_id: str = "fishaudio/openaudio-s1-mini",
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> Path:
    """Download S1 Mini model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID.
        cache_dir: Directory to cache/download model. Defaults to checkpoints/.
        token: HuggingFace token for private repos.

    Returns:
        Path to downloaded model directory.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent.parent / "checkpoints" / "openaudio-s1-mini"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading S1 Mini model from {repo_id} to {cache_dir}")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise



