"""Agent configuration schemas for ROMA-DSPy."""

from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator
from typing import List, Dict, Any, Optional

from .base import LLMConfig
from src.roma_dspy.types import PredictionStrategy


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""

    llm: Optional[LLMConfig] = None
    prediction_strategy: str = "chain_of_thought"
    tools: Optional[List[str]] = None
    enabled: bool = True

    # Separate agent-specific and strategy-specific configurations
    agent_config: Optional[Dict[str, Any]] = None      # Agent business logic parameters
    strategy_config: Optional[Dict[str, Any]] = None   # Prediction strategy algorithm parameters

    def __post_init__(self):
        """Initialize nested configs with defaults if not provided."""
        if self.llm is None:
            self.llm = LLMConfig()
        if self.tools is None:
            self.tools = []
        if self.agent_config is None:
            self.agent_config = {}
        if self.strategy_config is None:
            self.strategy_config = {}

    @field_validator("prediction_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate prediction strategy against available strategies."""
        try:
            PredictionStrategy.from_string(v)
            return v
        except ValueError:
            available = [strategy.value for strategy in PredictionStrategy]
            raise ValueError(f"Invalid prediction strategy '{v}'. Available: {available}")

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: Optional[List[str]]) -> List[str]:
        """Validate tools against available tools."""
        # Handle None case (can happen when merging configs)
        if v is None:
            return []

        # Based on actual tools in the codebase
        available_tools = ["calculator", "web_search"]

        for tool in v:
            if tool not in available_tools:
                raise ValueError(f"Unknown tool '{tool}'. Available tools: {available_tools}")
        return v


@dataclass
class AgentsConfig:
    """Configuration for all ROMA agents."""

    atomizer: Optional[AgentConfig] = None
    planner: Optional[AgentConfig] = None
    executor: Optional[AgentConfig] = None
    aggregator: Optional[AgentConfig] = None
    verifier: Optional[AgentConfig] = None

    def __post_init__(self):
        """Initialize agent configs with defaults if not provided."""
        if self.atomizer is None:
            self.atomizer = AgentConfig(
                llm=LLMConfig(temperature=0.1, max_tokens=1000),
                prediction_strategy="chain_of_thought",
                tools=[],
                agent_config={"confidence_threshold": 0.8},
                strategy_config={}
            )

        if self.planner is None:
            self.planner = AgentConfig(
                llm=LLMConfig(temperature=0.3, max_tokens=3000),
                prediction_strategy="chain_of_thought",
                tools=[],
                agent_config={"max_subtasks": 10},
                strategy_config={}
            )

        if self.executor is None:
            self.executor = AgentConfig(
                llm=LLMConfig(temperature=0.5),
                prediction_strategy="chain_of_thought",  # Use CoT instead of ReAct for now
                tools=[],
                agent_config={"max_executions": 5},
                strategy_config={}
            )

        if self.aggregator is None:
            self.aggregator = AgentConfig(
                llm=LLMConfig(temperature=0.2, max_tokens=4000),
                prediction_strategy="chain_of_thought",
                tools=[],
                agent_config={"synthesis_strategy": "hierarchical"},
                strategy_config={}
            )

        if self.verifier is None:
            self.verifier = AgentConfig(
                llm=LLMConfig(temperature=0.1),
                prediction_strategy="chain_of_thought",
                tools=[],
                agent_config={"verification_depth": "moderate"},
                strategy_config={}
            )

    @model_validator(mode="after")
    def validate_tool_strategy_compatibility(self):
        """Ensure tools are only used with compatible strategies."""
        tool_compatible_strategies = ["react", "code_act"]

        # Check executor (most likely to have tools)
        if self.executor.tools:
            if self.executor.prediction_strategy not in tool_compatible_strategies:
                raise ValueError(
                    f"Executor has tools {self.executor.tools} but strategy "
                    f"'{self.executor.prediction_strategy}' doesn't support tools. "
                    f"Use one of: {tool_compatible_strategies}"
                )

        # Check other agents that might have tools
        for agent_name, agent_config in [
            ("atomizer", self.atomizer),
            ("planner", self.planner),
            ("aggregator", self.aggregator),
            ("verifier", self.verifier)
        ]:
            if agent_config.tools:
                if agent_config.prediction_strategy not in tool_compatible_strategies:
                    raise ValueError(
                        f"{agent_name} has tools {agent_config.tools} but strategy "
                        f"'{agent_config.prediction_strategy}' doesn't support tools. "
                        f"Use one of: {tool_compatible_strategies}"
                    )

        return self