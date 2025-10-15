"""GEPA optimizer factory for prompt optimization."""

import dspy
from dspy import GEPA
from .config import OptimizationConfig
from .metrics import MetricWithFeedback
from .selectors import SELECTORS


def create_optimizer(
    config: OptimizationConfig,
    metric: MetricWithFeedback,
    component_selector: str = "planner_only"
) -> GEPA:
    """
    Create configured GEPA optimizer.

    Args:
        config: Optimization configuration
        metric: Metric function (typically MetricWithFeedback)
        component_selector: Name of selector function (see selectors.py)
            Options: "planner_only", "atomizer_only", "executor_only",
                     "aggregator_only", "round_robin"

    Returns:
        Configured GEPA optimizer

    Example:
        >>> config = get_default_config()
        >>> judge = ComponentJudge(config.judge_lm)
        >>> metric = MetricWithFeedback(judge)
        >>> optimizer = create_optimizer(config, metric, "planner_only")
    """

    # Initialize reflection LM
    reflection_lm = dspy.LM(
        model=config.reflection_lm.model,
        temperature=config.reflection_lm.temperature,
        max_tokens=config.reflection_lm.max_tokens,
        cache=config.reflection_lm.cache
    )

    # Get selector function
    selector_fn = SELECTORS.get(component_selector, SELECTORS["planner_only"])

    # Create GEPA optimizer
    return GEPA(
        metric=metric,
        component_selector=selector_fn,
        max_metric_calls=config.max_metric_calls,
        num_threads=config.num_threads,
        track_stats=True,
        reflection_minibatch_size=config.reflection_minibatch_size,
        reflection_lm=reflection_lm
    )
