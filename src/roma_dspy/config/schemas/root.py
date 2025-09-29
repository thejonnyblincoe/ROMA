"""Root configuration schema for ROMA-DSPy."""

from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator
import warnings
from typing import Optional

from .base import RuntimeConfig
from .agents import AgentsConfig
from .resilience import ResilienceConfig


@dataclass
class ROMAConfig:
    """Complete ROMA-DSPy configuration."""

    # Project metadata
    project: str = "roma-dspy"
    version: str = "0.1.0"
    environment: str = "development"

    # Core configurations
    agents: Optional[AgentsConfig] = None
    resilience: Optional[ResilienceConfig] = None
    runtime: Optional[RuntimeConfig] = None

    def __post_init__(self):
        """Initialize nested configs with defaults if not provided."""
        if self.agents is None:
            self.agents = AgentsConfig()
        if self.resilience is None:
            self.resilience = ResilienceConfig()
        if self.runtime is None:
            self.runtime = RuntimeConfig()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        allowed_environments = {"development", "testing", "production"}
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of {allowed_environments}, got: {v}")
        return v

    @model_validator(mode="after")
    def validate_global_consistency(self):
        """Validate global configuration consistency."""
        # Check model consistency across agents
        models = [
            self.agents.atomizer.llm.model,
            self.agents.planner.llm.model,
            self.agents.executor.llm.model,
            self.agents.aggregator.llm.model,
            self.agents.verifier.llm.model
        ]

        # Group models by provider
        openai_models = [m for m in models if "gpt" in m.lower()]
        anthropic_models = [m for m in models if "claude" in m.lower()]
        other_models = [m for m in models if "gpt" not in m.lower() and "claude" not in m.lower()]

        # Warn about mixed providers (not an error, just a warning)
        provider_count = sum([
            1 if openai_models else 0,
            1 if anthropic_models else 0,
            1 if other_models else 0
        ])

        if provider_count > 1:
            warnings.warn(
                "Mixed model providers detected. Ensure API keys are configured correctly. "
                f"OpenAI models: {openai_models}, Anthropic models: {anthropic_models}, "
                f"Other models: {other_models}",
                UserWarning
            )

        # Validate timeout consistency
        agent_timeouts = [
            self.agents.atomizer.llm.timeout,
            self.agents.planner.llm.timeout,
            self.agents.executor.llm.timeout,
            self.agents.aggregator.llm.timeout,
            self.agents.verifier.llm.timeout
        ]

        max_agent_timeout = max(agent_timeouts)
        if self.runtime.timeout < max_agent_timeout:
            raise ValueError(
                f"Runtime timeout ({self.runtime.timeout}s) is less than maximum "
                f"agent timeout ({max_agent_timeout}s). This may cause premature timeouts."
            )

        return self