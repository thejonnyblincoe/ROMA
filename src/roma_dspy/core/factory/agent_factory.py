"""Factory for creating agent instances with signature fallback."""

from typing import Type, Optional, Dict
import dspy

from loguru import logger

from roma_dspy.core.modules import (
    Atomizer, Planner, Executor, Aggregator, Verifier, BaseModule
)
from roma_dspy.core.signatures import (
    AtomizerSignature, PlannerSignature, ExecutorSignature,
    AggregatorSignature, VerifierSignature
)
from roma_dspy.config.schemas.agents import AgentConfig
from roma_dspy.types import AgentType, TaskType


class AgentFactory:
    """
    Factory for creating agent instances with signature fallback.

    Key Features:
    1. Default signatures for each agent type
    2. Inline signature parsing from config
    3. Graceful fallback on parsing errors
    4. Validation and error reporting
    """

    # Default signature mapping (includes Verifier)
    DEFAULT_SIGNATURES: Dict[AgentType, Type[dspy.Signature]] = {
        AgentType.ATOMIZER: AtomizerSignature,
        AgentType.PLANNER: PlannerSignature,
        AgentType.EXECUTOR: ExecutorSignature,
        AgentType.AGGREGATOR: AggregatorSignature,
        AgentType.VERIFIER: VerifierSignature,
    }

    # Module class mapping (includes Verifier)
    MODULE_CLASSES: Dict[AgentType, Type[BaseModule]] = {
        AgentType.ATOMIZER: Atomizer,
        AgentType.PLANNER: Planner,
        AgentType.EXECUTOR: Executor,
        AgentType.AGGREGATOR: Aggregator,
        AgentType.VERIFIER: Verifier,
    }

    def create_agent(
        self,
        agent_type: AgentType,
        agent_config: AgentConfig,
        task_type: Optional[TaskType] = None
    ) -> BaseModule:
        """
        Create agent instance with signature resolution.

        Signature Resolution Order:
        1. Try agent_config.signature (inline override)
        2. Fall back to default signature for agent_type
        3. On parsing error, log warning and use default

        Args:
            agent_type: Type of agent (ATOMIZER, PLANNER, EXECUTOR, AGGREGATOR, VERIFIER)
            agent_config: Configuration with optional signature override
            task_type: Task type for logging/debugging

        Returns:
            Configured agent instance

        Raises:
            ValueError: If agent_type is invalid or config is malformed
        """
        if agent_type not in self.MODULE_CLASSES:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Resolve signature with fallback
        signature = self._resolve_signature(agent_type, agent_config, task_type)

        # Get module class
        module_class = self.MODULE_CLASSES[agent_type]

        # Create instance with resolved signature
        instance = module_class(
            signature=signature,
            config=agent_config
        )

        logger.info(
            f"Created {agent_type.value} agent "
            f"(task_type={task_type.value if task_type else 'default'}, "
            f"signature={'custom' if agent_config.signature else 'default'})"
        )

        return instance

    def _resolve_signature(
        self,
        agent_type: AgentType,
        agent_config: AgentConfig,
        task_type: Optional[TaskType]
    ) -> Type[dspy.Signature]:
        """
        Resolve signature with robust fallback mechanism.

        Returns:
            dspy.Signature class (either parsed or default)
        """
        default_signature = self.DEFAULT_SIGNATURES[agent_type]

        # No custom signature - use default
        if not agent_config.signature:
            logger.debug(f"Using default signature for {agent_type.value}")
            return default_signature

        # Try parsing custom signature
        try:
            custom_signature = self._parse_inline_signature(
                agent_config.signature,
                agent_config.signature_instructions
            )

            logger.info(
                f"Using custom signature for {agent_type.value}: "
                f"{agent_config.signature}"
            )

            return custom_signature

        except Exception as e:
            # Parsing failed - fall back to default with warning
            logger.warning(
                f"Failed to parse inline signature for {agent_type.value} "
                f"(task_type={task_type}): {agent_config.signature}. "
                f"Error: {e}. Falling back to default signature."
            )

            return default_signature

    @staticmethod
    def _parse_inline_signature(
        signature_str: str,
        instructions: Optional[str] = None
    ) -> Type[dspy.Signature]:
        """
        Parse DSPy inline signature string.

        Formats supported:
        - "input -> output"
        - "input: type -> output: type"
        - "field1, field2: type -> output1, output2: type"

        Args:
            signature_str: Inline signature string
            instructions: Optional instructions for the signature

        Returns:
            dspy.Signature class

        Raises:
            ValueError: If signature_str is malformed

        Examples:
            >>> _parse_inline_signature("goal -> is_atomic: bool")
            >>> _parse_inline_signature(
            ...     "goal: str -> output: str, sources: list[str]",
            ...     instructions="Execute task with tools"
            ... )
        """
        if not signature_str or "->" not in signature_str:
            raise ValueError(
                f"Invalid signature format: '{signature_str}'. "
                f"Must contain '->'"
            )

        # Clean signature string
        signature_str = signature_str.strip()

        # DSPy.Signature can be created from string directly
        if instructions:
            signature = dspy.Signature(signature_str, instructions=instructions)
        else:
            signature = dspy.Signature(signature_str)

        return signature

    @classmethod
    def get_default_signature(cls, agent_type: AgentType) -> Type[dspy.Signature]:
        """Get default signature for agent type (utility method)."""
        return cls.DEFAULT_SIGNATURES[agent_type]