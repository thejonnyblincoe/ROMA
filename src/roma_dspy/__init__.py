"""ROMA-DSPy: modular hierarchical task decomposition framework."""

from typing import Optional, Sequence

from .core import (
    TaskDAG,
    RecursiveSolver,
    solve,
    async_solve,
    event_solve,
    async_event_solve,
    Atomizer,
    Planner,
    Executor,
    Aggregator,
    Verifier,
    AtomizerSignature,
    PlannerSignature,
    ExecutorSignature,
    AggregatorSignature,
    VerifierSignature,
    SubTask,
    TaskNode,
    RecursiveSolverModule,
)

__all__ = [
    "TaskDAG",
    "RecursiveSolver",
    "solve",
    "async_solve",
    "event_solve",
    "async_event_solve",
    "Atomizer",
    "Planner",
    "Executor",
    "Aggregator",
    "Verifier",
    "AtomizerSignature",
    "PlannerSignature",
    "ExecutorSignature",
    "AggregatorSignature",
    "VerifierSignature",
    "SubTask",
    "TaskNode",
    "RecursiveSolverModule",
    "main",
]