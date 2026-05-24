"""meow-embed package."""

from typing import TYPE_CHECKING

from meow_embed._metadata import __description__, __version__

if TYPE_CHECKING:
    from meow_embed.cache import EmbedCache, EmbedCacheProgress
    from meow_embed.client import MeowEmbedClient

__all__ = [
    "EmbedCache",
    "EmbedCacheProgress",
    "MeowEmbedClient",
    "__description__",
    "__version__",
]


def __getattr__(name: str) -> object:
    if name == "MeowEmbedClient":
        from meow_embed.client import MeowEmbedClient

        return MeowEmbedClient
    if name == "EmbedCache":
        from meow_embed.cache import EmbedCache

        return EmbedCache
    if name == "EmbedCacheProgress":
        from meow_embed.cache import EmbedCacheProgress

        return EmbedCacheProgress
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
