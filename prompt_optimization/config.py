"""Configuration management for prompt optimization."""

from dataclasses import dataclass, field
from typing import Optional

from dspy import ChatAdapter, Adapter


@dataclass
class LMConfig:
    """Language model configuration."""
    model: str
    temperature: float = 0.6
    max_tokens: int = 120000
    cache: bool = False
    adapter: Optional[Adapter] = ChatAdapter()


@dataclass
class OptimizationConfig:
    """Complete configuration for optimization pipeline."""

    # LM configs
    executor_lm: LMConfig = field(default_factory=lambda: LMConfig("fireworks_ai/accounts/fireworks/models/gpt-oss-120b"))
    atomizer_lm: LMConfig = field(default_factory=lambda: LMConfig("gemini/gemini-2.5-flash"))
    planner_lm: LMConfig = field(default_factory=lambda: LMConfig("gemini/gemini-2.5-flash"))
    aggregator_lm: LMConfig = field(default_factory=lambda: LMConfig("fireworks_ai/accounts/fireworks/models/gpt-oss-120b"))
    judge_lm: LMConfig = field(default_factory=lambda: LMConfig("openrouter/anthropic/claude-sonnet-4.5", temperature=1.0, max_tokens=64000))
    reflection_lm: LMConfig = field(default_factory=lambda: LMConfig("openrouter/anthropic/claude-sonnet-4.5", temperature=1.0, max_tokens=64000))

    # Dataset configs
    train_size: int = 5
    val_size: int = 5
    test_size: int = 15
    dataset_seed: int = 0

    # Execution configs
    max_parallel: int = 12
    concurrency: int = 12

    # GEPA configs
    max_metric_calls: int = 10
    num_threads: int = 4
    reflection_minibatch_size: int = 8

    # Solver configs
    max_depth: int = 1
    enable_logging: bool = True

    # Output
    output_path: Optional[str] = None


def get_default_config() -> OptimizationConfig:
    """Returns default optimization configuration."""
    return OptimizationConfig()
