"""Solver factory and setup for prompt optimization."""

import dspy
from roma_dspy import (
    RecursiveSolverModule,
    RecursiveSolverFactory,
    Executor,
    Atomizer,
    Planner,
    Aggregator
)
from .config import OptimizationConfig
from .seed_prompts import ATOMIZER_PROMPT, PLANNER_PROMPT, AGGREGATOR_PROMPT
from dspy import ChatAdapter

def create_solver_module(config: OptimizationConfig) -> RecursiveSolverModule:
    """
    Create configured RecursiveSolverModule with all components.

    Args:
        config: Optimization configuration

    Returns:
        Configured RecursiveSolverModule
    """

    # Initialize LMs from config
    executor_lm = dspy.LM(
        model=config.executor_lm.model,
        temperature=config.executor_lm.temperature,
        max_tokens=config.executor_lm.max_tokens,
        cache=config.executor_lm.cache
    )
    atomizer_lm = dspy.LM(
        model=config.atomizer_lm.model,
        temperature=config.atomizer_lm.temperature,
        max_tokens=config.atomizer_lm.max_tokens,
        cache=config.atomizer_lm.cache
    )
    planner_lm = dspy.LM(
        model=config.planner_lm.model,
        temperature=config.planner_lm.temperature,
        max_tokens=config.planner_lm.max_tokens,
        cache=config.planner_lm.cache
    )
    aggregator_lm = dspy.LM(
        model=config.aggregator_lm.model,
        temperature=config.aggregator_lm.temperature,
        max_tokens=config.aggregator_lm.max_tokens,
        cache=config.aggregator_lm.cache
    )

    # Initialize modules
    atomizer = Atomizer(lm=atomizer_lm)
    planner = Planner(lm=planner_lm)
    executor = Executor(lm=executor_lm)
    aggregator = Aggregator(lm=aggregator_lm)

    # Set custom instructions
    atomizer.signature.instructions = ATOMIZER_PROMPT
    planner.signature.instructions = PLANNER_PROMPT
    aggregator.signature.instructions = AGGREGATOR_PROMPT
    # Create solver factory
    solver_factory = RecursiveSolverFactory(
        atomizer,
        planner,
        executor,
        aggregator,
        max_depth=config.max_depth,
        enable_logging=config.enable_logging
    )

    return RecursiveSolverModule(solver_factory=solver_factory)
