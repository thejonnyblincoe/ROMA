"""Thin DSPy module wrapper around the RecursiveSolver orchestration engine."""

from __future__ import annotations

from typing import Optional, Callable

import dspy

from ..engine.solve import RecursiveSolver
from ..engine.dag import TaskDAG
from ..signatures import TaskNode
from ...visualizer import LLMTraceVisualizer


class RecursiveSolverModule(dspy.Module):
    """Expose RecursiveSolver through a DSPy module interface.

    This is a thin wrapper: it accepts a pre-configured `RecursiveSolver` and
    delegates execution to it. Attributes and methods not found on the module
    are proxied to the underlying solver for convenience.
    """

    def __init__(self, *, solver: RecursiveSolver) -> None:
        super().__init__()
        self._solver = solver

        # Expose commonly used components as direct attributes (shared references)
        self.atomizer = solver.atomizer
        self.planner = solver.planner
        self.executor = solver.executor
        self.aggregator = solver.aggregator
        self.verifier = solver.verifier
        self.runtime = solver.runtime
        self.max_depth = solver.max_depth

    def forward(
        self,
        goal: str,
        *,
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 1,
    ) -> dspy.Prediction:
        completed_task = self._solver.event_solve(
            task=goal,
            dag=dag,
            depth=depth,
            priority_fn=priority_fn,
            concurrency=concurrency,
        )

        viz = LLMTraceVisualizer(show_metrics=False, show_summary=False, verbose=True)
        trace = viz.visualize(self._solver)

        return dspy.Prediction(
            goal=goal,
            completed_task=completed_task,
            status=completed_task.status,
            result_text=str(completed_task.result) if completed_task.result is not None else None,
            output_trace=trace,
        )

    async def aforward(
        self,
        goal: str,
        *,
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 8,
    ) -> dspy.Prediction:
        completed_task = await self._solver.async_event_solve(
            task=goal,
            dag=dag,
            depth=depth,
            priority_fn=priority_fn,
            concurrency=concurrency,
        )

        viz = LLMTraceVisualizer(show_metrics=False, show_summary=False, verbose=True)
        trace = viz.visualize(self._solver)

        return dspy.Prediction(
            goal=goal,
            completed_task=completed_task,
            status=completed_task.status,
            result_text=str(completed_task.result) if completed_task.result is not None else None,
            output_trace=trace,
        )
