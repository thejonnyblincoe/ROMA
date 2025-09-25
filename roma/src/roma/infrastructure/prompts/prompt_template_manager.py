"""
Prompt Template Manager for ROMA v2.

Centralized management of Jinja2 prompt templates with caching and validation.
Handles template loading, rendering, and validation for all agent types.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateError

if TYPE_CHECKING:
    from roma.application.services.context_builder_service import ContextBuilderService
    from roma.domain.context.task_context import TaskContext
    from roma.domain.entities.task_node import TaskNode

from datetime import UTC

from roma.domain.value_objects.task_type import TaskType

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages ALL template operations: loading, caching, rendering, validation.

    Single responsibility: Template management and rendering with variable export.
    Uses ContextBuilderService to get all available template variables.
    Templates can choose which variables to include based on their needs.
    """

    def __init__(
        self,
        templates_dir: str | None = None,
        context_builder: Optional["ContextBuilderService"] = None,
    ):
        """
        Initialize prompt template manager.

        Args:
            templates_dir: Base directory for prompt templates
            context_builder: Optional context builder for variable export
        """
        if templates_dir is None:
            self.templates_dir = self._discover_templates_directory()
        else:
            self.templates_dir = Path(templates_dir)

        self.context_builder = context_builder  # Injected dependency
        self._template_cache: dict[str, Template] = {}
        self._initialized = False

        # Create Jinja2 environment with custom filters
        if self.templates_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
            # Add custom filters for template use
            self.env.filters["truncate"] = self._truncate_filter
            self.env.filters["format_list"] = self._format_list_filter
            self.env.filters["safe_get"] = self._safe_get_filter
        else:
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.env = Environment()

        self._initialized = True
        logger.info(f"PromptTemplateManager initialized with templates dir: {self.templates_dir}")

    def _discover_templates_directory(self) -> Path:
        """
        Discover templates directory using multiple strategies.

        Resolution order:
        1. ROMA_PROMPTS_DIR environment variable
        2. Traverse up from current file to find src/prompts directory
        3. Fallback to relative path calculation

        Returns:
            Path to templates directory

        Raises:
            FileNotFoundError: If templates directory cannot be found
        """
        import os

        # 1. Check environment variable first
        env_dir = os.getenv("ROMA_PROMPTS_DIR")
        if env_dir:
            templates_dir = Path(env_dir)
            if templates_dir.exists() and templates_dir.is_dir():
                logger.info(f"Using templates directory from ROMA_PROMPTS_DIR: {templates_dir}")
                return templates_dir
            else:
                logger.warning(f"ROMA_PROMPTS_DIR points to non-existent directory: {env_dir}")

        # 2. Traverse upward to find src/prompts or roma/src/prompts
        current_path = Path(__file__).resolve()
        for parent in [current_path] + list(current_path.parents):
            # Check for direct src/prompts structure
            src_prompts = parent / "src" / "prompts"
            if src_prompts.exists() and src_prompts.is_dir():
                logger.info(f"Found templates directory by traversal: {src_prompts}")
                return src_prompts

            # Check for roma/src/prompts structure
            roma_src_prompts = parent / "roma" / "src" / "prompts"
            if roma_src_prompts.exists() and roma_src_prompts.is_dir():
                logger.info(f"Found templates directory by roma traversal: {roma_src_prompts}")
                return roma_src_prompts

        # 3. Fallback to relative path calculation
        # __file__ is at roma/src/roma/infrastructure/prompts/prompt_template_manager.py
        # We want roma/src/prompts, so go up 4 levels then into src/prompts
        current_file = Path(__file__).resolve()
        if len(current_file.parents) >= 4:
            fallback_path = current_file.parents[4] / "src" / "prompts"
            if fallback_path.exists():
                logger.info(f"Using fallback templates directory: {fallback_path}")
                return fallback_path

        # 4. All strategies failed
        fallback_desc = (
            "insufficient parent levels" if len(current_file.parents) < 4 else "path does not exist"
        )
        raise FileNotFoundError(
            f"Cannot find templates directory. Tried:\n"
            f"- Environment variable ROMA_PROMPTS_DIR: {env_dir or 'not set'}\n"
            f"- Traversal from {current_path}\n"
            f"- Fallback calculation: {fallback_desc}"
        )

    def load_template(self, template_path: str) -> Template:
        """
        Load a template from file with caching, with fallback for legacy filename patterns.

        Args:
            template_path: Relative path from templates directory (e.g., "atomizer/retrieve.jinja2")

        Returns:
            Jinja2 Template instance

        Raises:
            FileNotFoundError: If template file doesn't exist
            TemplateError: If template syntax is invalid
        """
        if template_path in self._template_cache:
            logger.debug(f"Template cache hit: {template_path}")
            return self._template_cache[template_path]

        # Try original path first
        full_path = self.templates_dir / template_path

        # If original path doesn't exist, try fallback patterns
        if not full_path.exists():
            fallback_path = self._get_fallback_template_path(template_path)
            if fallback_path:
                fallback_full_path = self.templates_dir / fallback_path
                if fallback_full_path.exists():
                    template_path = fallback_path  # Use fallback path
                    full_path = fallback_full_path
                    logger.debug(f"Using fallback template path: {fallback_path}")
                else:
                    raise FileNotFoundError(
                        f"Template not found: {full_path} (fallback {fallback_full_path} also not found)"
                    )
            else:
                raise FileNotFoundError(f"Template not found: {full_path}")

        try:
            # Use Jinja environment to load template directly - enables includes/extends
            template = self.env.get_template(template_path)

            # Cache the template
            self._template_cache[template_path] = template

            logger.debug(f"Loaded and cached template: {template_path}")
            return template

        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            raise TemplateError(f"Invalid template {template_path}: {e}") from e

    async def render_agent_prompt(
        self, agent_type: str, task_type: str, task: "TaskNode", task_context: "TaskContext"
    ) -> str:
        """
        Main method: Render prompt for agent with ALL available variables.

        Args:
            agent_type: Type of agent (atomizer, planner, etc.)
            task_type: Type of task (RETRIEVE, WRITE, etc.)
            task: TaskNode being processed
            task_context: Complete TaskContext from ContextBuilderService

        Returns:
            Rendered prompt string

        Raises:
            TemplateError: If template rendering fails
        """
        try:
            # Get template path and load template
            template_path = self.get_template_path(agent_type, task_type)
            template = self.load_template(template_path)

            # Export ALL available variables using ContextBuilderService
            if self.context_builder:
                template_vars = await self.context_builder.export_template_variables(
                    task, task_context
                )
            else:
                # Fallback to basic variables if no context builder
                template_vars = self._get_basic_template_variables(
                    agent_type, task_type, task, task_context
                )

            # Render template with all variables
            rendered = template.render(**template_vars)

            logger.debug(f"Rendered {agent_type}/{task_type} template successfully")
            return rendered.strip()

        except Exception as e:
            logger.error(f"Failed to render {agent_type}/{task_type} template: {e}")
            raise TemplateError(f"Template rendering failed for {agent_type}/{task_type}: {e}") from e

    def get_template_path(self, agent_type: str, task_type: str) -> str:
        """Get template path for agent and task type."""
        return f"{agent_type.lower()}/{task_type.lower()}.jinja2"

    def _get_basic_template_variables(
        self, agent_type: str, task_type: str, task: "TaskNode", task_context: "TaskContext"
    ) -> dict[str, Any]:
        """
        Get basic template variables when ContextBuilderService is not available.

        Provides minimal required variables for templates to function.
        """
        from datetime import datetime

        now = datetime.now(UTC)

        return {
            # Essential core variables (always available)
            "agent_type": agent_type,
            "task_type": task_type,
            "task": task,
            "goal": task.goal,
            "task_status": task.status.value if task.status else "PENDING",
            "overall_objective": task_context.overall_objective,
            # Minimal temporal context (required for LLM grounding)
            "current_date": now.strftime("%Y-%m-%d"),
            "current_year": now.year,
            # Basic context
            "constraints": getattr(task_context, "constraints", []),
            "user_preferences": getattr(task_context, "user_preferences", {}),
            "execution_metadata": getattr(task_context, "execution_metadata", {}),
            # Helper flags
            "has_constraints": bool(getattr(task_context, "constraints", [])),
            "has_prior_work": False,  # Default when no context builder
            "has_toolkits": False,  # Default when no context builder
            "has_artifacts": False,  # Default when no context builder
            # Task type information from domain layer
            "task_types_info": TaskType.get_all_task_info(),
            "current_task_type_info": {
                "description": TaskType.from_string(task_type).get_description(),
                "examples": TaskType.from_string(task_type).get_examples(),
                "atomic_indicators": TaskType.from_string(task_type).get_atomic_indicators(),
                "composite_indicators": TaskType.from_string(task_type).get_composite_indicators(),
            },
        }

    # Custom Jinja2 filters for templates
    def _truncate_filter(self, text: str, length: int = 200) -> str:
        """Truncate text to specified length."""
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= length:
            return text
        return text[:length] + "..."

    def _format_list_filter(self, items: list, prefix: str = "- ") -> str:
        """Format list as string with prefix."""
        if not items:
            return ""
        return "\n".join([f"{prefix}{item}" for item in items])

    def _safe_get_filter(self, obj: dict, key: str, default: str = "") -> str:
        """Safely get value from dict with default."""
        if not isinstance(obj, dict):
            return default
        return str(obj.get(key, default))

    def render_template(self, template_path: str, context: dict[str, Any]) -> str:
        """
        Load and render a template with provided context.

        Args:
            template_path: Relative path from templates directory
            context: Template variables for rendering

        Returns:
            Rendered template string

        Raises:
            FileNotFoundError: If template file doesn't exist
            TemplateError: If template rendering fails
        """
        try:
            template = self.load_template(template_path)
            rendered = template.render(**context)
            logger.debug(f"Rendered template: {template_path}")
            return rendered.strip()

        except Exception as e:
            logger.error(f"Template rendering failed for {template_path}: {e}")
            raise TemplateError(f"Template rendering failed: {e}") from e

    def validate_template(self, template_path: str) -> bool:
        """
        Validate template syntax without rendering.

        Args:
            template_path: Relative path from templates directory

        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.load_template(template_path)
            return True
        except (FileNotFoundError, TemplateError):
            return False

    def template_exists(self, template_path: str) -> bool:
        """
        Check if a template file exists.

        Args:
            template_path: Relative path from templates directory

        Returns:
            True if template exists, False otherwise
        """
        full_path = self.templates_dir / template_path
        return full_path.exists()

    def get_default_template_path(self, agent_type: str, task_type: str) -> str:
        """
        Generate default template path for agent and task type.

        Args:
            agent_type: Agent type (atomizer, planner, executor, etc.)
            task_type: Task type (RETRIEVE, WRITE, THINK, etc.)

        Returns:
            Default template path string
        """
        return f"{agent_type.lower()}/{task_type.lower()}.jinja2"

    def list_templates(self, agent_type: str | None = None) -> list[str]:
        """
        List available templates, optionally filtered by agent type.

        Args:
            agent_type: Optional agent type to filter by

        Returns:
            List of template paths
        """
        if not self.templates_dir.exists():
            return []

        pattern = "**/*.jinja2"
        if agent_type:
            pattern = f"{agent_type.lower()}/**/*.jinja2"

        templates = []
        for template_file in self.templates_dir.glob(pattern):
            relative_path = template_file.relative_to(self.templates_dir)
            templates.append(str(relative_path))

        return sorted(templates)

    def _get_fallback_template_path(self, template_path: str) -> str | None:
        """
        Generate fallback template path for legacy filename patterns.

        For paths like "executor/retrieve.jinja2", tries "executor/retrieve_executor.jinja2"
        For paths like "planner/think.jinja2", tries "planner/think_planner.jinja2"

        Args:
            template_path: Original template path

        Returns:
            Fallback template path or None if no fallback pattern applies
        """
        try:
            # Split path into directory and filename
            path_parts = template_path.split("/")
            if len(path_parts) != 2:
                return None

            agent_type, filename = path_parts
            if not filename.endswith(".jinja2"):
                return None

            # Extract task type from filename (remove .jinja2 extension)
            task_type = filename[:-7]  # Remove .jinja2

            # Generate fallback pattern: {task_type}_{agent_type}.jinja2
            fallback_filename = f"{task_type}_{agent_type}.jinja2"
            fallback_path = f"{agent_type}/{fallback_filename}"

            logger.debug(f"Generated fallback template path: {template_path} -> {fallback_path}")
            return fallback_path

        except Exception as e:
            logger.debug(f"Could not generate fallback for {template_path}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
        logger.info("Template cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the template cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_templates": len(self._template_cache),
            "template_paths": list(self._template_cache.keys()),
            "templates_dir": str(self.templates_dir),
            "initialized": self._initialized,
        }

    def reload_template(self, template_path: str) -> Template:
        """
        Force reload a template, bypassing cache.

        Args:
            template_path: Relative path from templates directory

        Returns:
            Freshly loaded Template instance
        """
        # Remove from cache if present
        self._template_cache.pop(template_path, None)

        # Load fresh template
        return self.load_template(template_path)
