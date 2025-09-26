"""Configuration settings for DSPy LLM and system parameters."""

from pathlib import Path
from typing import Optional

import dspy
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.roma_dspy.types import RetryConfig, CircuitBreakerConfig, RetryStrategy
from src.roma_dspy.resilience import RetryPolicy, create_default_retry_policy


class LLMConfig(BaseModel):
    """LLM configuration parameters."""

    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens for generation")
    base_url: Optional[str] = Field(default=None, description="Base URL for custom LLM endpoints")


class ResilienceConfig(BaseModel):
    """Resilience configuration parameters."""

    # Retry configuration
    retry_strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL_BACKOFF, description="Default retry strategy")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")
    max_delay: float = Field(default=60.0, description="Maximum delay between retries")
    jitter_factor: float = Field(default=0.1, description="Jitter factor for retry delays")

    # Circuit breaker configuration
    circuit_failure_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_recovery_timeout: float = Field(default=60.0, description="Circuit breaker recovery timeout")
    circuit_success_threshold: int = Field(default=2, description="Circuit breaker success threshold")
    circuit_evaluation_window: float = Field(default=300.0, description="Circuit breaker evaluation window")

    def to_retry_config(self) -> RetryConfig:
        """Convert to RetryConfig object."""
        return RetryConfig(
            strategy=self.retry_strategy,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter_factor=self.jitter_factor,
            backoff_multiplier=2.0  # Standard exponential backoff
        )

    def to_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Convert to CircuitBreakerConfig object."""
        return CircuitBreakerConfig(
            failure_threshold=self.circuit_failure_threshold,
            recovery_timeout=self.circuit_recovery_timeout,
            success_threshold=self.circuit_success_threshold,
            evaluation_window=self.circuit_evaluation_window
        )


class SystemConfig(BaseModel):
    """System configuration parameters."""

    cache_dir: Path = Field(default=Path(".cache/dspy"), description="Cache directory for DSPy")
    max_concurrency: int = Field(default=5, description="Maximum concurrent LLM calls")
    timeout: int = Field(default=30, description="Timeout for LLM calls in seconds")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Resilience configuration
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig, description="Resilience settings")


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

    def create_retry_policy(self) -> RetryPolicy:
        """Create RetryPolicy from current settings."""
        retry_config = self.system.resilience.to_retry_config()
        return RetryPolicy(retry_config)

    def get_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Get CircuitBreakerConfig from current settings."""
        return self.system.resilience.to_circuit_breaker_config()

    def initialize_resilience(self) -> None:
        """Initialize global resilience policies."""
        # This could be used to configure module-level circuit breakers
        # with settings-based configurations if needed
        pass


# Global settings instance
settings = Settings()