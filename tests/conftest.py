"""Pytest configuration and helpers for ROMA DSPy tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

# Configure DSPy cache directory before importing DSPy modules
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".dspy_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

cache_path = str(_CACHE_DIR)
os.environ.setdefault("DSPY_CACHEDIR", cache_path)
os.environ.setdefault("DSPY_CACHE_DIR", cache_path)

import pytest

from roma_dspy import SubTask
from roma_dspy.types import NodeType, PredictionStrategy, TaskType


def _build_response(signature: Any, payload: Dict[str, Any]) -> Any:
    name = getattr(signature, "__name__", str(signature))

    if name == "AtomizerSignature":
        goal = payload.get("goal", "").lower()
        is_atomic = goal.startswith("atomic")
        node_type = NodeType.EXECUTE if is_atomic else NodeType.PLAN
        return SimpleNamespace(is_atomic=is_atomic, node_type=node_type)

    if name == "PlannerSignature":
        goal = payload.get("goal", "")
        subtasks = [
            SubTask(goal=f"{goal} -> step 1", task_type=TaskType.THINK, dependencies=[]),
            SubTask(goal=f"{goal} -> step 2", task_type=TaskType.WRITE, dependencies=["step 1"]),
        ]
        dependencies = {"step 2": ["step 1"]}
        return SimpleNamespace(subtasks=subtasks, dependencies_graph=dependencies)

    if name == "ExecutorSignature":
        goal = payload.get("goal", "")
        return SimpleNamespace(output=f"executed:{goal}", sources=["stub-source"])

    if name == "AggregatorResult":
        subtasks = payload.get("subtasks_results", [])
        combined = " | ".join(getattr(item, "goal", str(item)) for item in subtasks)
        return SimpleNamespace(synthesized_result=combined or "no subtasks provided")

    if name == "VerifierSignature":
        candidate = payload.get("candidate_output", "")
        verdict = "fail" not in candidate.lower()
        feedback = None if verdict else "Output flagged by verifier"
        return SimpleNamespace(verdict=verdict, feedback=feedback)

    raise ValueError(f"Unhandled signature '{name}' in test stub")


@pytest.fixture(autouse=True)
def stub_prediction_strategy(monkeypatch: pytest.MonkeyPatch):
    class DummyPredictor:
        def __init__(self, signature: Any, strategy: PredictionStrategy, **kwargs: Any) -> None:
            self.signature = signature
            self.strategy = strategy
            self.build_kwargs = kwargs
            self.calls = []

        def _respond(self, kwargs: Dict[str, Any]) -> Any:
            self.calls.append(kwargs)
            return _build_response(self.signature, kwargs)

        def forward(self, **kwargs: Any) -> Any:
            return self._respond(kwargs)

        async def acall(self, **kwargs: Any) -> Any:
            return self._respond(kwargs)

        def __call__(self, **kwargs: Any) -> Any:
            return self.forward(**kwargs)

    def build(self: PredictionStrategy, signature: Any, **kwargs: Any) -> DummyPredictor:
        return DummyPredictor(signature=signature, strategy=self, **kwargs)

    monkeypatch.setattr(PredictionStrategy, "build", build)
    yield
