"""DSPy Signatures for hierarchical task decomposition."""

import dspy


class AtomizeSignature(dspy.Signature):
    """Signature for task atomization."""
    pass


class PlanSignature(dspy.Signature):
    """Signature for task planning."""
    pass


class ExecuteSignature(dspy.Signature):
    """Signature for task execution."""
    pass


class AggregateSignature(dspy.Signature):
    """Signature for result aggregation."""
    pass


class VerifySignature(dspy.Signature):
    """Signature for result verification."""
    pass