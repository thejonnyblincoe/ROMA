"""Configuration settings for DSPy LLM and system parameters."""

from pathlib import Path
from typing import Optional

import dspy
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """LLM configuration parameters."""

    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for generation")
    base_url: Optional[str] = Field(default=None, description="Base URL for custom LLM endpoints")


class SystemConfig(BaseModel):
    """System configuration parameters."""

    cache_dir: Path = Field(default=Path(".cache/dspy"), description="Cache directory for DSPy")
    max_concurrency: int = Field(default=5, description="Maximum concurrent LLM calls")
    timeout: int = Field(default=30, description="Timeout for LLM calls in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed calls")
    verbose: bool = Field(default=False, description="Enable verbose logging")


class Settings(BaseSettings):
    """Application settings combining all configurations."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    # Environment prefix for loading from env vars
    model_config = {"env_prefix": "ROMA_"}

    def initialize_dspy(self) -> None:
        """Initialize DSPy with current settings."""
        # Create cache directory if it doesn't exist
        self.system.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DSPy LM
        if self.llm.model.startswith("gpt"):
            from dspy import OpenAI
            lm = OpenAI(
                model=self.llm.model,
                api_key=self.llm.api_key,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
        else:
            # Add other LLM providers as needed
            raise ValueError(f"Unsupported model: {self.llm.model}")

        dspy.configure(lm=lm)


# Global settings instance
settings = Settings()