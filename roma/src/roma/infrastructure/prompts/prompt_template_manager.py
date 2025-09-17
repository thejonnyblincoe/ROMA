"""
Prompt Template Manager for ROMA v2.

Centralized management of Jinja2 prompt templates with caching and validation.
Handles template loading, rendering, and validation for all agent types.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
from jinja2 import Template, TemplateError, Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages prompt templates for ROMA agents.

    Provides centralized template loading, caching, and rendering capabilities
    for all agent types using Jinja2 templates.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize prompt template manager.

        Args:
            templates_dir: Base directory for prompt templates
        """
        if templates_dir is None:
            # Make path relative to this module to avoid working directory issues
            # __file__ is at roma/src/roma/infrastructure/prompts/prompt_template_manager.py
            # We want roma/src/prompts, so go up 4 levels then into src/prompts
            base = Path(__file__).resolve().parents[4] / "src" / "prompts"
            self.templates_dir = base
        else:
            self.templates_dir = Path(templates_dir)
        self._template_cache: Dict[str, Template] = {}
        self._initialized = False

        # Create Jinja2 environment for better template management
        if self.templates_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
        else:
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.env = Environment()

        self._initialized = True
        logger.info(f"PromptTemplateManager initialized with templates dir: {self.templates_dir}")

    def load_template(self, template_path: str) -> Template:
        """
        Load a template from file with caching.

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

        full_path = self.templates_dir / template_path
        if not full_path.exists():
            raise FileNotFoundError(f"Template not found: {full_path}")

        try:
            # Load template content
            template_content = full_path.read_text(encoding="utf-8")

            # Create Jinja2 template using environment
            template = self.env.from_string(template_content)

            # Cache the template
            self._template_cache[template_path] = template

            logger.debug(f"Loaded and cached template: {template_path}")
            return template

        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            raise TemplateError(f"Invalid template {template_path}: {e}")

    def render_template(
        self,
        template_path: str,
        context: Dict[str, Any]
    ) -> str:
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
            raise TemplateError(f"Template rendering failed: {e}")

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

    def list_templates(self, agent_type: Optional[str] = None) -> list[str]:
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

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
        logger.info("Template cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the template cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_templates": len(self._template_cache),
            "template_paths": list(self._template_cache.keys()),
            "templates_dir": str(self.templates_dir),
            "initialized": self._initialized
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