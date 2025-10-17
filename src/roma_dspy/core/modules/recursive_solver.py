"""Thin DSPy module wrapper around the RecursiveSolver orchestration engine."""

from __future__ import annotations

from typing import Optional, Callable, List, Tuple, Any
import copy

import dspy

from roma_dspy.core.engine.solve import RecursiveSolver
from roma_dspy.core.engine.dag import TaskDAG
from roma_dspy.core.signatures import TaskNode
from roma_dspy.visualizer import LLMTraceVisualizer
from roma_dspy.types import AgentType, TaskType
from loguru import logger


class RecursiveSolverModule(dspy.Module):
    """Expose RecursiveSolver through a DSPy module interface.

    This is a thin wrapper: it accepts a pre-configured `RecursiveSolver` and
    delegates execution to it. Attributes and methods not found on the module
    are proxied to the underlying solver for convenience.
    """

    def __init__(self, *, solver: RecursiveSolver) -> None:
        super().__init__()
        self._solver = solver
        self._last_solver: Optional[RecursiveSolver] = None  # Snapshot from most recent run

        # Expose solver's core attributes (these actually exist on RecursiveSolver)
        self.runtime = solver.runtime
        self.max_depth = solver.max_depth
        self.registry = solver.registry  # For accessing agents via registry.get_agent(AgentType, TaskType)

    def forward(
        self,
        goal: str,
        *,
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 1,
    ) -> dspy.Prediction:
        solver_instance = self._spawn_solver()

        completed_task = solver_instance.event_solve(
            task=goal,
            dag=dag,
            depth=depth,
            priority_fn=priority_fn,
            concurrency=concurrency,
        )

        viz = LLMTraceVisualizer(show_metrics=False, show_summary=False, verbose=True)
        trace = viz.visualize(solver_instance)
        self._last_solver = solver_instance

        return dspy.Prediction(
            goal=goal,
            completed_task=completed_task,
            status=completed_task.status,
            result_text=str(completed_task.result) if completed_task.result is not None else None,
            output_trace=trace,
        )

    def named_predictors(self) -> List[Tuple[str, Any]]:
        """
        Surface all predictors exposed by the recursive solver's agents.

        GEPA inspects DSPy modules via named_predictors(); exposing the agents'
        predictors here allows the optimizer to mutate individual instructions.
        """
        predictors = list(super().named_predictors())

        if not hasattr(self._solver, "registry"):
            return predictors

        seen_ids = {id(pred) for _, pred in predictors}

        for agent_type, task_type, module in self._solver.registry.iter_agents():
            if not hasattr(module, "named_predictors"):
                continue

            agent_predictors = module.named_predictors()
            if not agent_predictors:
                logger.debug(
                    "Agent %s (task=%s) has no named predictors (type=%s)",
                    getattr(agent_type, "value", agent_type),
                    getattr(task_type, "value", task_type) if task_type is not None else "default",
                    type(module).__name__
                )
                continue

            agent_label = agent_type.value.lower() if isinstance(agent_type, AgentType) else str(agent_type).lower()
            task_label = (
                task_type.value.lower()
                if isinstance(task_type, TaskType)
                else ("default" if task_type is None else str(task_type).lower())
            )

            for predictor_name, predictor in agent_predictors:
                if predictor is None:
                    logger.debug(
                        "Skipping None predictor from agent %s (task=%s)",
                        getattr(agent_type, "value", agent_type),
                        getattr(task_type, "value", task_type) if task_type is not None else "default",
                    )
                    continue

                predictor_id = id(predictor)
                if predictor_id in seen_ids:
                    continue

                composite_name = f"{agent_label}__{task_label}__{predictor_name}"
                predictors.append((composite_name, predictor))
                seen_ids.add(predictor_id)

        exported_names = [name for name, _ in predictors]
        logger.debug(
            f"RecursiveSolverModule.named_predictors exported {len(exported_names)} predictors: {exported_names}"
        )

        return predictors

    async def aforward(
        self,
        goal: str,
        *,
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 8,
    ) -> dspy.Prediction:
        solver_instance = self._spawn_solver()

        completed_task = await solver_instance.async_event_solve(
            task=goal,
            dag=dag,
            depth=depth,
            priority_fn=priority_fn,
            concurrency=concurrency,
        )

        viz = LLMTraceVisualizer(show_metrics=False, show_summary=False, verbose=True)
        trace = viz.visualize(solver_instance)
        self._last_solver = solver_instance

        return dspy.Prediction(
            goal=goal,
            completed_task=completed_task,
            status=completed_task.status,
            result_text=str(completed_task.result) if completed_task.result is not None else None,
            output_trace=trace,
        )

    def _spawn_solver(self) -> RecursiveSolver:
        """
        Create an isolated solver instance for a single call.

        DSPy optimizers (e.g., GEPA) run multiple rollouts concurrently. The
        underlying RecursiveSolver mutates execution state (overall objective,
        DAG, context), so sharing the same instance across concurrent calls can
        mix traces/goals between rollouts. Deep copying ensures each execution
        has isolated state while preserving any prompt/LM edits applied to the
        template solver.
        """
        return copy.deepcopy(self._solver)
