"""Base configuration schemas for ROMA-DSPy."""

from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import Optional


@dataclass
class LLMConfig:
    """Language model configuration."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is within valid range."""
        if not (0 < v <= 100000):
            raise ValueError(f"Max tokens must be between 1 and 100000, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError(f"Timeout must be positive, got {v}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


@dataclass
class RuntimeConfig:
    """Runtime system configuration."""

    max_concurrency: int = 5
    timeout: int = 30
    verbose: bool = False
    cache_dir: str = ".cache/dspy"

    @field_validator("max_concurrency")
    @classmethod
    def validate_max_concurrency(cls, v: int) -> int:
        """Validate max_concurrency is within valid range."""
        if not (1 <= v <= 50):
            raise ValueError(f"Max concurrency must be between 1 and 50, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is within valid range."""
        if not (1 <= v <= 300):
            raise ValueError(f"Timeout must be between 1 and 300 seconds, got {v}")
        return v

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Validate cache directory path."""
        if not v or not v.strip():
            raise ValueError("Cache directory cannot be empty")
        return v.strip()