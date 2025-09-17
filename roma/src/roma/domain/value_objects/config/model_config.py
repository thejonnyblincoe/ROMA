"""
Model Configuration Value Object.

Defines LLM model configuration as a domain value object.
Single source of truth for model configuration across the system.
"""

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ModelConfig:
    """LLM model configuration value object."""
    
    provider: str = "litellm"
    model_id: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 120
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    top_p: Optional[float] = None
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError(f"temperature must be 0-2, got: {v}")
        return v
    
    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1 or v > 100000:
            raise ValueError(f"max_tokens must be 1-100000, got: {v}")
        return v
    
    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > 600:  # 1 second to 10 minutes
            raise ValueError(f"timeout must be 1-600 seconds, got: {v}")
        return v
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        # Only litellm provider - handles all models through unified interface
        valid_providers = ["litellm"]
        if v.lower() not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got: {v}")
        return v.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "api_base": self.api_base,
            "api_key": self.api_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**data)