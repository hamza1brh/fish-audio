"""Provider registry for dynamic provider selection."""

from typing import Type

from loguru import logger

from src.providers.base import TTSProvider

_providers: dict[str, Type[TTSProvider]] = {}
_instances: dict[str, TTSProvider] = {}


def register_provider(name: str, provider_class: Type[TTSProvider]) -> None:
    """Register a provider class.

    Args:
        name: Provider identifier (e.g., "s1_mini", "elevenlabs").
        provider_class: The provider class to register.
    """
    _providers[name] = provider_class
    logger.debug(f"Registered provider: {name}")


def get_provider_class(name: str) -> Type[TTSProvider] | None:
    """Get a registered provider class by name.

    Args:
        name: Provider identifier.

    Returns:
        Provider class or None if not found.
    """
    return _providers.get(name)


async def get_provider(name: str, **kwargs) -> TTSProvider:
    """Get or create a provider instance.

    Args:
        name: Provider identifier.
        **kwargs: Provider initialization arguments.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider is not registered.
    """
    if name in _instances:
        return _instances[name]

    provider_class = _providers.get(name)
    if not provider_class:
        available = list(_providers.keys())
        raise ValueError(f"Provider '{name}' not registered. Available: {available}")

    instance = provider_class(**kwargs)
    await instance.initialize()
    _instances[name] = instance
    logger.info(f"Initialized provider: {name}")
    return instance


async def shutdown_providers() -> None:
    """Shutdown all active provider instances."""
    for name, instance in _instances.items():
        try:
            await instance.shutdown()
            logger.info(f"Shutdown provider: {name}")
        except Exception as e:
            logger.error(f"Error shutting down provider {name}: {e}")
    _instances.clear()


def list_providers() -> list[str]:
    """List registered provider names."""
    return list(_providers.keys())

