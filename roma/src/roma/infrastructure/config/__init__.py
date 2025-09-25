"""Infrastructure configuration layer."""

from .hydra_integration import cs, register_configs

__all__ = ["register_configs", "cs"]
