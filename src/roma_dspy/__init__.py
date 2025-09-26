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
    "main",
]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point referenced by ``pyproject.toml``."""

    import argparse
    from importlib import metadata

    parser = argparse.ArgumentParser(description="ROMA-DSPy command-line interface")
    parser.add_argument("--version", action="store_true", help="print package version")
    parser.add_argument(
        "--help-solvers",
        action="store_true",
        help="show solvers exposed by the library",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        try:
            version = metadata.version("roma-dspy")
        except metadata.PackageNotFoundError:
            version = "unknown (local checkout)"
        print(f"ROMA-DSPy {version}")
        return

    if args.help_solvers:
        print("Available solvers:\n - solve\n - async_solve\n - event_solve\n - async_event_solve")
        return

    parser.print_help()
