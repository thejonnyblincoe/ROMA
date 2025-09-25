"""
TaskType enumeration for ROMA v2.0

Implements the MECE (Mutually Exclusive, Collectively Exhaustive) framework
for task classification with RETRIEVE replacing SEARCH from v1.

ROMA v2 task types (five total):
- RETRIEVE: External data acquisition from multiple sources
- WRITE: Content generation and synthesis
- THINK: Analysis, reasoning, decision making
- CODE_INTERPRET: Code execution and data processing
- IMAGE_GENERATION: Visual content creation

Note: AGGREGATE is an agent type (Aggregator), not a task type.
"""

from enum import Enum
from typing import Any, Literal


class TaskType(str, Enum):
    """
    MECE task classification for universal task decomposition.
    """

    RETRIEVE = "RETRIEVE"  # Multi-source data acquisition
    WRITE = "WRITE"  # Content generation and synthesis
    THINK = "THINK"  # Analysis, reasoning, decision making
    CODE_INTERPRET = "CODE_INTERPRET"  # Code execution and data processing
    IMAGE_GENERATION = "IMAGE_GENERATION"  # Visual content creation

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """
        Convert string to TaskType.

        Args:
            value: String representation of task type

        Returns:
            TaskType enum value

        Raises:
            ValueError: If value is not a valid task type
        """
        try:
            return cls(value.upper())
        except ValueError as e:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid task type '{value}'. Valid types: {valid_types}") from e

    @property
    def is_retrieve(self) -> bool:
        """Check if this is a RETRIEVE task type."""
        return self == TaskType.RETRIEVE

    @property
    def is_write(self) -> bool:
        """Check if this is a WRITE task type."""
        return self == TaskType.WRITE

    @property
    def is_think(self) -> bool:
        """Check if this is a THINK task type."""
        return self == TaskType.THINK

    @property
    def is_code_interpret(self) -> bool:
        """Check if this is a CODE_INTERPRET task type."""
        return self == TaskType.CODE_INTERPRET

    @property
    def is_image_generation(self) -> bool:
        """Check if this is an IMAGE_GENERATION task type."""
        return self == TaskType.IMAGE_GENERATION

    def get_description(self) -> str:
        """Get detailed description for this task type."""
        descriptions = {
            TaskType.RETRIEVE: "External data acquisition from multiple sources including web search, APIs, databases, and knowledge stores. Focuses on gathering information efficiently with intelligent source routing.",
            TaskType.WRITE: "Content generation, synthesis, and document creation. Includes writing articles, reports, summaries, creative content, and structured documentation with proper formatting.",
            TaskType.THINK: "Analysis, reasoning, decision making, and computation. Covers logical reasoning, problem-solving, mathematical calculations, strategic planning, and complex cognitive tasks.",
            TaskType.CODE_INTERPRET: "Code execution, data processing, and computational tasks. Handles running code, data analysis, file processing, API interactions, and technical computations.",
            TaskType.IMAGE_GENERATION: "Visual content creation including generating images, diagrams, charts, and visual representations using AI image generation models and tools.",
        }
        return descriptions[self]

    def get_examples(self) -> list[str]:
        """Get concrete examples for this task type."""
        examples = {
            TaskType.RETRIEVE: [
                "Find Tesla's current stock price",
                "Research AI impact on healthcare industry",
                "Get latest unemployment statistics for US",
                "Look up Bitcoin price on January 1, 2024",
                "Search for recent advances in quantum computing",
            ],
            TaskType.WRITE: [
                "Write a summary of market trends",
                "Create a business proposal document",
                "Draft an email response to customer inquiry",
                "Generate a technical specification",
                "Compose a research report conclusion",
            ],
            TaskType.THINK: [
                "Analyze pros and cons of investment strategy",
                "Calculate optimal portfolio allocation",
                "Determine root cause of system failure",
                "Plan project timeline and milestones",
                "Evaluate competitive market positioning",
            ],
            TaskType.CODE_INTERPRET: [
                "Process CSV data and generate statistics",
                "Run Python script to analyze logs",
                "Execute SQL query on database",
                "Call REST API and parse response",
                "Validate JSON data structure",
            ],
            TaskType.IMAGE_GENERATION: [
                "Create a logo for new product",
                "Generate diagram of system architecture",
                "Design chart showing sales trends",
                "Create illustration for article",
                "Generate mockup of user interface",
            ],
        }
        return examples[self]

    def get_atomic_indicators(self) -> list[str]:
        """Get indicators that suggest this task is atomic (doesn't need decomposition)."""
        indicators = {
            TaskType.RETRIEVE: [
                "Single specific fact lookup",
                "Direct API call result",
                "One data point query",
                "Current status check",
                "Simple definition lookup",
            ],
            TaskType.WRITE: [
                "Short response or summary",
                "Single document section",
                "Specific format output",
                "Template-based content",
                "Direct transcription task",
            ],
            TaskType.THINK: [
                "Single calculation or formula",
                "Binary decision (yes/no)",
                "Direct comparison of two items",
                "Simple logical deduction",
                "Straightforward analysis",
            ],
            TaskType.CODE_INTERPRET: [
                "Single function execution",
                "One-step data transformation",
                "Direct API call",
                "Simple calculation",
                "Single file operation",
            ],
            TaskType.IMAGE_GENERATION: [
                "Single image creation",
                "One specific visual element",
                "Direct style transfer",
                "Simple diagram or chart",
                "Single icon or symbol",
            ],
        }
        return indicators[self]

    def get_composite_indicators(self) -> list[str]:
        """Get indicators that suggest this task needs decomposition."""
        indicators = {
            TaskType.RETRIEVE: [
                "Multiple related information pieces",
                "Research requiring synthesis",
                "Comparative data gathering",
                "Historical analysis",
                "Multi-source investigation",
            ],
            TaskType.WRITE: [
                "Comprehensive reports or documents",
                "Multi-section content",
                "Research-based writing",
                "Complex synthesis tasks",
                "Large-scale documentation",
            ],
            TaskType.THINK: [
                "Complex multi-step analysis",
                "Strategic planning with dependencies",
                "Multi-criteria decision making",
                "System-wide optimization",
                "Comprehensive evaluation",
            ],
            TaskType.CODE_INTERPRET: [
                "Multi-step data pipeline",
                "Complex data analysis workflow",
                "System integration tasks",
                "Batch processing operations",
                "Multi-file operations",
            ],
            TaskType.IMAGE_GENERATION: [
                "Multi-image series or collection",
                "Complex scenes with multiple elements",
                "Coordinated visual campaign",
                "Interactive visual content",
                "Multi-format visual outputs",
            ],
        }
        return indicators[self]

    @classmethod
    def get_all_task_info(cls) -> dict[str, dict[str, Any]]:
        """Get comprehensive information about all task types."""
        return {
            task_type.value: {
                "description": task_type.get_description(),
                "examples": task_type.get_examples(),
                "atomic_indicators": task_type.get_atomic_indicators(),
                "composite_indicators": task_type.get_composite_indicators(),
            }
            for task_type in cls
        }


# Type hints for use in other modules
TaskTypeLiteral = Literal["RETRIEVE", "WRITE", "THINK", "CODE_INTERPRET", "IMAGE_GENERATION"]
