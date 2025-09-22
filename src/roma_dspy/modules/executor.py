"""Executor module for task execution and tool routing."""

from __future__ import annotations

import dspy
from typing import Union, Any, Optional, Dict, Mapping, Sequence, Mapping as TMapping, List

from src.roma_dspy.signatures.signatures import ExecutorSignature
from src.roma_dspy.types.prediction_strategy import PredictionStrategy
from src.roma_dspy.modules.base_module import BaseModule


class Executor(BaseModule):
    """Executes atomic tasks and routes to tools."""

    def __init__(
        self,
        prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT,
        *,
        lm: Optional[dspy.LM] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        **strategy_kwargs: Any,
    ) -> None:
        super().__init__(
            signature=ExecutorSignature,
            prediction_strategy=prediction_strategy,
            lm=lm,
            model=model,
            model_config=model_config,
            tools=tools,
            **strategy_kwargs,
        )

    @classmethod
    def from_provider(
        cls,
        prediction_strategy: Union[PredictionStrategy, str] = PredictionStrategy.CHAIN_OF_THOUGHT,
        *,
        model: str,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        **model_config: Any,
    ) -> "Executor":
        return cls(
            prediction_strategy,
            model=model,
            model_config=model_config or None,
            tools=tools,
        )