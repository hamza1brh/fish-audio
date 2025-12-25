"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Run GPU tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="Need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Set test environment before importing app modules
os.environ.setdefault("VOICE_TTS_PROVIDER", "mock")
os.environ.setdefault("VOICE_REFERENCE_STORAGE", "local")
os.environ.setdefault("VOICE_LOG_LEVEL", "warning")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def checkpoint_path():
    """Path to model checkpoint for GPU tests."""
    paths = [
        Path("checkpoints/openaudio-s1-mini"),
        Path("voice-service/checkpoints/openaudio-s1-mini"),
        Path(os.environ.get("VOICE_S1_CHECKPOINT_PATH", "")),
    ]
    for p in paths:
        if p.exists() and (p / "model.pth").exists():
            return p
    pytest.skip("Model checkpoint not found")


@pytest.fixture(scope="session")
def engine(checkpoint_path, gpu_available):
    """Initialize S1MiniEngine for GPU tests."""
    if not gpu_available:
        pytest.skip("GPU not available")

    os.environ["VOICE_S1_CHECKPOINT_PATH"] = str(checkpoint_path)
    os.environ["VOICE_TTS_PROVIDER"] = "s1_mini"

    # Fix pyrootutils issue: fish-speech installed as package needs project root
    # Create .project-root marker in parent directory if it doesn't exist
    parent_dir = Path.cwd().parent
    project_root_marker = parent_dir / ".project-root"
    if not project_root_marker.exists():
        project_root_marker.touch()
    
    # Set PYTHONPATH to include parent directory
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if str(parent_dir) not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{parent_dir}:{current_pythonpath}" if current_pythonpath else str(parent_dir)

    # Find fish_speech path
    fish_speech_paths = [
        Path.cwd().parent,
        Path.cwd(),
        Path(os.environ.get("VOICE_FISH_SPEECH_PATH", "")),
    ]
    for p in fish_speech_paths:
        if (p / "fish_speech").exists():
            os.environ["VOICE_FISH_SPEECH_PATH"] = str(p)
            break

    from src.inference.engine import S1MiniBackend

    eng = S1MiniBackend(
        checkpoint_path=checkpoint_path,
        codec_path=checkpoint_path / "codec.pth",
        device="cuda",
        compile=False,
    )
    eng.initialize()
    yield eng
    eng.shutdown()


