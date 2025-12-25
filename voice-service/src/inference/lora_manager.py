"""LoRA adapter hot-swap manager."""

from pathlib import Path
from typing import Any

from loguru import logger


class LoRAManager:
    """Manages LoRA adapter loading and hot-swapping.

    Provides functionality to load, unload, and switch between LoRA adapters
    without restarting the service.
    """

    def __init__(self) -> None:
        self._active_lora: Path | None = None
        self._model = None
        self._base_model_state = None

    @property
    def active_lora(self) -> Path | None:
        """Currently active LoRA adapter path."""
        return self._active_lora

    @property
    def is_loaded(self) -> bool:
        """Check if a LoRA is currently loaded."""
        return self._active_lora is not None

    def set_model(self, model: Any) -> None:
        """Set the model reference for LoRA operations.

        Args:
            model: The base transformer model.
        """
        self._model = model

    async def load_lora(self, lora_path: Path) -> bool:
        """Load a LoRA adapter.

        Args:
            lora_path: Path to the LoRA adapter directory.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not lora_path.exists():
            logger.error(f"LoRA path does not exist: {lora_path}")
            return False

        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required_files if not (lora_path / f).exists()]
        if missing:
            alt_file = lora_path / "adapter_model.bin"
            if "adapter_model.safetensors" in missing and alt_file.exists():
                missing.remove("adapter_model.safetensors")

        if missing:
            logger.error(f"LoRA missing files: {missing}")
            return False

        if self._model is None:
            logger.error("Model not set, cannot load LoRA")
            return False

        try:
            from peft import PeftModel

            if self._active_lora is not None:
                await self.unload_lora()

            self._model = PeftModel.from_pretrained(
                self._model,
                str(lora_path),
                is_trainable=False,
            )

            self._active_lora = lora_path
            logger.info(f"Loaded LoRA adapter: {lora_path}")
            return True

        except ImportError:
            logger.error("peft not installed, LoRA support unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False

    async def unload_lora(self) -> bool:
        """Unload the current LoRA adapter.

        Returns:
            True if unloaded successfully.
        """
        if self._active_lora is None:
            return True

        try:
            if hasattr(self._model, "unload"):
                self._model.unload()
            elif hasattr(self._model, "base_model"):
                self._model = self._model.base_model.model

            self._active_lora = None
            logger.info("Unloaded LoRA adapter")
            return True

        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return False

    async def swap_lora(self, new_lora_path: Path) -> bool:
        """Hot-swap to a different LoRA adapter.

        Args:
            new_lora_path: Path to the new LoRA adapter.

        Returns:
            True if swap successful.
        """
        if self._active_lora == new_lora_path:
            logger.info("LoRA already loaded, skipping swap")
            return True

        logger.info(f"Hot-swapping LoRA: {self._active_lora} -> {new_lora_path}")

        if not await self.unload_lora():
            return False

        return await self.load_lora(new_lora_path)

    def get_model(self) -> Any:
        """Get the current model (with or without LoRA).

        Returns:
            The model instance.
        """
        return self._model

















