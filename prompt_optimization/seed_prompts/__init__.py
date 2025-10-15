"""Few-shot seed prompts and demos for the Planner component.

This package provides `demos` for use with DSPy examples.
"""

from .aggregator_seed import AGGREGATOR_PROMPT
from .atomizer_seed import ATOMIZER_PROMPT, ATOMIZER_DEMOS
from .planner_seed import PLANNER_PROMPT, PLANNER_DEMOS

__all__ = [
    "AGGREGATOR_PROMPT",
    "ATOMIZER_PROMPT",
    "ATOMIZER_DEMOS",
    "PLANNER_PROMPT",
    "PLANNER_DEMOS",
]