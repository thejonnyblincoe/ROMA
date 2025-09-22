"""Planner module for execution planning."""

import dspy
from typing import Union, Any
from src.roma_dspy.signatures.signatures import PlannerSignature
from src.roma_dspy.types.prediction_strategy import PredictionStrategy


class Planner(dspy.Module):
    """Plans task execution strategy."""

    def __init__(self, prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT, **strategy_kwargs: Any):
        super().__init__()
        self.signature = PlannerSignature
        if isinstance(prediction_strategy, str):
            prediction_strategy = PredictionStrategy.from_string(prediction_strategy)
        self.planner = prediction_strategy.build(self.signature, **strategy_kwargs)

    def forward(self, input_task: str):
        return self.planner(input_task)