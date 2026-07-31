"""Leakage-safe League of Legends draft win prediction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lol-draft-predictor")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
