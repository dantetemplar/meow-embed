"""Package version and description (no optional dependencies)."""

from importlib.metadata import metadata, version

__version__ = version("meow-embed")
__description__ = metadata("meow-embed")["Summary"]
