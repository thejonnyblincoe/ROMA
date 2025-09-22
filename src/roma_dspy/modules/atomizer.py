"""Atomizer module for task decomposition."""

import dspy
from typing import Union, Any
from src.roma_dspy.signatures.signatures import AtomizerSignature
from src.roma_dspy.types.prediction_strategy import PredictionStrategy


class Atomizer(dspy.Module):
    """Decomposes tasks into atomic units."""

    def __init__(self, prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT, **strategy_kwargs: Any):
        super().__init__()
        self.signature = AtomizerSignature
        if isinstance(prediction_strategy, str):
            prediction_strategy = PredictionStrategy.from_string(prediction_strategy)
        self.atomizer = prediction_strategy.build(self.signature, **strategy_kwargs)

    def forward(self, input_task: str):
        return self.atomizer(input_task)