"""
Base Agno Toolkit integration for ROMA v2.

Clean implementation following SOLID principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Extensible through composition and inheritance
- Liskov Substitution: All implementations are substitutable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions

Handles both:
1. Default Agno toolkits - imported directly from Agno
2. Custom toolkits - inherit from Agno's Toolkit + specialized base classes

Configuration via Hydra YAML with lazy initialization.
"""

# No longer need ABC since we inherit from AgnoToolkit
import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC
from pathlib import Path
from typing import Any

from agno.tools import Toolkit as AgnoToolkit

logger = logging.getLogger(__name__)


class BaseAgnoToolkit(AgnoToolkit):
    """
    Base class for ROMA toolkit integration following SOLID principles.

    Single Responsibility: Manages toolkit configuration and lifecycle
    Open/Closed: Extensible through inheritance and composition
    Liskov Substitution: All subclasses can be used interchangeably
    """

    def __init__(self, config: dict[str, Any], **_kwargs):
        """Initialize toolkit with immutable configuration."""
        self._validate_config(config)
        toolkit_name = config.get("name", "unnamed_toolkit")

        # Initialize AgnoToolkit parent with just the name
        super().__init__(name=toolkit_name)

        # ROMA-specific configuration
        self._toolkit_type = config.get("type", "generic")
        self._config = config.copy()  # Defensive copy
        self._enabled = config.get("enabled", True)
        self._created = False

    @property
    def toolkit_type(self) -> str:
        """Toolkit type (immutable)."""
        return self._toolkit_type

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration at initialization (fail fast)."""
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")
        if not config.get("name", "").strip():
            raise ValueError("Toolkit name cannot be empty")

    def is_enabled(self) -> bool:
        """Check if toolkit is enabled."""
        return self._enabled

    async def create(self) -> None:
        """Initialize/create the toolkit (idempotent)."""
        if self._created:
            logger.debug(f"Toolkit {self.name} already created")
            return

        await self._perform_creation()
        self._created = True
        logger.info(f"Created toolkit: {self.name} (type: {self.toolkit_type})")

    async def _perform_creation(self) -> None:
        """Template method for subclass-specific creation logic."""

    def is_created(self) -> bool:
        """Check if toolkit is created."""
        return self._created

    async def update_config(self, new_config: dict[str, Any]) -> None:
        """Update mutable configuration parts only."""
        # Only allow updates to mutable configuration
        mutable_keys = {"enabled", "config"}
        updates = {k: v for k, v in new_config.items() if k in mutable_keys}

        if updates:
            self._config.update(updates)
            self._enabled = self._config.get("enabled", self._enabled)
            logger.info(f"Updated config for toolkit: {self.name}")

    def get_config(self) -> dict[str, Any]:
        """Get current toolkit configuration (defensive copy)."""
        return self._config.copy()

    async def delete(self) -> None:
        """Delete/cleanup the toolkit (idempotent)."""
        if not self._created:
            logger.debug(f"Toolkit {self.name} already deleted")
            return

        await self._perform_deletion()
        self._created = False
        logger.info(f"Deleted toolkit: {self.name}")

    async def _perform_deletion(self) -> None:
        """Template method for subclass-specific deletion logic."""

    def get_available_tools(self) -> list[str]:
        """Get list of available tool methods in this toolkit."""
        tools = []
        for attr_name in dir(self):
            if (not attr_name.startswith("_") and callable(getattr(self, attr_name))
                and attr_name not in [
                    "create",
                    "delete",
                    "update_config",
                    "get_config",
                    "is_enabled",
                    "is_created",
                    "is_default_agno_toolkit",
                    "is_custom_toolkit",
                ]):
                tools.append(attr_name)
        return tools

    def is_default_agno_toolkit(self) -> bool:
        """Check if this is a default Agno toolkit."""
        return "implementation" not in self._config

    def is_custom_toolkit(self) -> bool:
        """Check if this is a custom toolkit implementation."""
        return "implementation" in self._config


class DefaultAgnoToolkitWrapper:
    """
    Wrapper for default Agno toolkits following Single Responsibility Principle.

    Responsibility: Import and manage default Agno toolkits with include/exclude tool filtering
    Dependencies: Inverted - depends on AgnoToolkit abstraction
    """

    def __init__(self, toolkit_name: str, config: dict[str, Any]):
        """Initialize with fail-fast validation."""
        if not toolkit_name or not toolkit_name.strip():
            raise ValueError("Toolkit name cannot be empty")
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")

        self._name = config.get("name", toolkit_name)
        self._toolkit_type = config.get("type", "generic")
        self._config = config.copy()
        self._agno_toolkit = None
        self._toolkit_name = toolkit_name
        self._created = False

        # Store include/exclude tools for Agno toolkit creation
        self._include_tools = config.get("include_tools", [])
        self._exclude_tools = config.get("exclude_tools", [])

    @property
    def name(self) -> str:
        """Toolkit name (immutable)."""
        return self._name

    @property
    def toolkit_type(self) -> str:
        """Toolkit type (immutable)."""
        return self._toolkit_type

    async def create(self) -> None:
        """Create by importing from Agno (idempotent)."""
        if self._created:
            logger.debug(f"Default toolkit {self.name} already created")
            return

        try:
            self._agno_toolkit = await self._create_agno_toolkit()
            self._created = True
            logger.info(f"Imported default Agno toolkit: {self.name}")

        except Exception as e:
            logger.error(f"Failed to import Agno toolkit {self._toolkit_name}: {e}")
            raise

    async def _create_agno_toolkit(self) -> Any:
        """Factory method for creating Agno toolkit with include/exclude filtering."""
        try:
            # Import the appropriate Agno toolkit class based on type
            toolkit_class = self._get_agno_toolkit_class()
            if not toolkit_class:
                # Fallback to generic toolkit
                logger.warning(f"Unknown toolkit type {self._toolkit_type}, using generic")
                return AgnoToolkit(name=self.name)

            # Create toolkit with base configuration
            toolkit_kwargs = dict(self._config.get("config", {}))

            # Add include/exclude tools if specified
            if self._include_tools:
                toolkit_kwargs["include_tools"] = self._include_tools
            if self._exclude_tools:
                toolkit_kwargs["exclude_tools"] = self._exclude_tools

            # Create the Agno toolkit instance
            agno_toolkit = toolkit_class(**toolkit_kwargs)
            logger.info(
                f"Created {self._toolkit_type} toolkit with include_tools={self._include_tools} exclude_tools={self._exclude_tools}"
            )
            return agno_toolkit

        except Exception as e:
            logger.error(f"Failed to create Agno toolkit {self._toolkit_type}: {e}")
            # Fallback to basic toolkit
            return AgnoToolkit(name=self.name)

    def _get_agno_toolkit_class(self) -> type | None:
        """Get the appropriate Agno toolkit class based on type."""
        import importlib

        # Mapping of toolkit types to Agno toolkit classes
        toolkit_mapping = {
            "web_search": {
                "google_search": "agno.tools.googlesearch.GoogleSearchTools",
                "exa_search": "agno.tools.exa.ExaTools",
                "openai_web_search": "agno.tools.openai.OpenAITools",
                "arxiv_search": "agno.tools.arxiv.ArxivTools",
                "wikipedia_search": "agno.tools.wikipedia.WikipediaTools",
            },
            "code_execution": {
                "e2b_sandbox": "agno.tools.e2b.E2BTools",
                "python_tools": "agno.tools.python.PythonTools",
                "jupyter_kernel": "agno.tools.jupyter.JupyterTools",
            },
            "reasoning": {
                "logic_analyzer": "agno.tools.reasoning.ReasoningTools",
                "hypothesis_tester": "agno.tools.reasoning.ReasoningTools",
            },
            "image_generation": {
                "dalle_3": "agno.tools.dalle.DalleTools",
                "stable_diffusion": "agno.tools.stability.StabilityTools",
            },
            "visualization": {
                "mermaid_diagram": "agno.tools.mermaid.MermaidTools",
            },
        }

        # Get toolkit path based on type and name
        type_mapping = toolkit_mapping.get(self._toolkit_type, {})
        toolkit_path = type_mapping.get(self._name)

        if not toolkit_path:
            # Try generic mapping by type
            generic_mapping = {
                "web_search": "agno.tools.googlesearch.GoogleSearchTools",
                "code_execution": "agno.tools.python.PythonTools",
                "reasoning": "agno.tools.reasoning.ReasoningTools",
                "image_generation": "agno.tools.dalle.DalleTools",
                "visualization": "agno.tools.mermaid.MermaidTools",
            }
            toolkit_path = generic_mapping.get(self._toolkit_type)

        if not toolkit_path:
            return None

        try:
            module_path, class_name = toolkit_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import Agno toolkit {toolkit_path}: {e}")
            return None

    def get_agno_toolkit(self) -> Any:
        """Get the underlying Agno toolkit instance."""
        if not self._created or self._agno_toolkit is None:
            raise RuntimeError(f"Toolkit {self.name} not created yet")
        return self._agno_toolkit

    def is_default_agno_toolkit(self) -> bool:
        """Type check - always true for default wrappers."""
        return True

    def is_custom_toolkit(self) -> bool:
        """Type check - always false for default wrappers."""
        return False

    async def delete(self) -> None:
        """Cleanup resources (idempotent)."""
        if not self._created:
            return

        self._agno_toolkit = None
        self._created = False
        logger.info(f"Deleted default toolkit: {self.name}")

    def get_available_tools(self) -> list[str]:
        """Get list of available tool methods from underlying Agno toolkit."""
        if self._agno_toolkit and hasattr(self._agno_toolkit, "tools"):
            return (
                list(self._agno_toolkit.tools.keys())
                if isinstance(self._agno_toolkit.tools, dict)
                else []
            )
        return []

    async def update_config(self, new_config: dict[str, Any]) -> None:
        """Update toolkit configuration."""
        self._config.update(new_config)
        logger.info(f"Updated config for default toolkit: {self.name}")

    def get_config(self) -> dict[str, Any]:
        """Get current toolkit configuration."""
        return self._config.copy()


# CustomAgnoToolkit removed to avoid diamond inheritance
# Custom toolkits should inherit directly from BaseAgnoToolkit


# Base classes following exact v1 pattern
class BaseDataToolkit:
    """
    Base class for data-oriented toolkits (exact v1 pattern).

    Provides common data management functionality including:
    - Parquet file storage for large datasets
    - Configurable data thresholds and storage paths
    - Data validation and conversion utilities
    - Standardized data directory management

    This follows the exact v1 pattern with _init_data_helpers().
    """

    def _init_data_helpers(
        self,
        data_dir: str | Path,
        parquet_threshold: int = 1000,
        file_prefix: str = "",
        toolkit_name: str = "",
    ) -> None:
        """Initialize data management helpers (exact v1 pattern)."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_threshold = parquet_threshold
        self._file_prefix = file_prefix
        self._toolkit_name = toolkit_name

        logger.info(f"Data helpers initialized - Toolkit: {toolkit_name}, Dir: {self.data_dir}")

    def _should_store_as_parquet(self, data: Any) -> bool:
        """Check if data should be stored as parquet based on size threshold (v1 pattern)."""
        try:
            json_str = json.dumps(data, default=str, ensure_ascii=False)
            size_bytes = len(json_str.encode("utf-8"))
            size_kb = size_bytes / 1024
            return size_kb > self._parquet_threshold
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            if isinstance(data, list):
                return len(data) > 100
            return False

    def _store_parquet(self, data: Any, prefix: str) -> str:
        """Store data as parquet file (exact v1 pattern)."""
        import time

        filename = f"{self._file_prefix}{prefix}_{int(time.time())}.parquet"
        file_path = self.data_dir / filename

        # Mock parquet storage - real implementation would use pandas
        file_path.write_text(f"PARQUET_DATA: {json.dumps(data, default=str)}")

        logger.info(f"Stored parquet data: {file_path}")
        return str(file_path)


class BaseAPIToolkit:
    """
    Base class for API-oriented toolkits (exact v1 pattern).

    Provides common API functionality including:
    - API parameter validation and cleaning
    - Authentication management
    - Identifier resolution (symbols, coin IDs, etc.)
    - Standardized response formatting

    This follows the exact v1 pattern with _init_standard_configuration().
    """

    def _init_standard_configuration(
        self,
        http_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize standard API configuration (exact v1 pattern)."""
        self._http_timeout = http_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache_ttl = cache_ttl_seconds

        logger.info("Standard API configuration initialized")

    def _validate_configuration_mapping(
        self, value: str, valid_configs: dict[str, Any], param_name: str
    ) -> None:
        """Validate configuration against mapping (exact v1 pattern)."""
        if value not in valid_configs:
            raise ValueError(
                f"Invalid {param_name}: {value}. Valid options: {list(valid_configs.keys())}"
            )

    def _validate_api_parameters(
        self, params: dict[str, Any], required_params: list[str] | None = None
    ) -> dict[str, Any]:
        """Validate and clean API parameters (v1 pattern)."""
        if required_params:
            missing = [p for p in required_params if p not in params or params[p] is None]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")

        # Remove None values and return clean params
        return {k: v for k, v in params.items() if v is not None}

    def _resolve_identifier(
        self,
        identifier: str,
        identifier_type: str = "symbol",
        fallback_value: str | None = None,
    ) -> str:
        """Resolve and validate identifiers (v1 pattern)."""
        if not identifier or not identifier.strip():
            if fallback_value:
                return fallback_value.strip().upper()
            raise ValueError(f"Invalid {identifier_type}: empty or None")

        return identifier.strip().upper()


class ToolkitValidationMixin(ABC):
    """
    Optional validation mixin for toolkits.

    Provides standardized health check and validation capabilities
    that toolkits can optionally implement for monitoring.
    """

    @abstractmethod
    async def validate_health(self) -> tuple[bool, str | None, str | None]:
        """
        Validate toolkit health status.

        Returns:
            Tuple of (is_healthy, warning_message, error_message)
        """

    async def validate_configuration(self) -> tuple[bool, str | None, str | None]:
        """
        Validate toolkit configuration.

        Default implementation - can be overridden.

        Returns:
            Tuple of (is_valid, warning_message, error_message)
        """
        try:
            # Basic configuration checks
            if hasattr(self, "_config"):
                config = self._config
                if not isinstance(config, dict):
                    return False, None, "Configuration is not a dictionary"

                if not config.get("name", "").strip():
                    return False, None, "Toolkit name is empty"

            return True, None, None

        except Exception as e:
            return False, None, f"Configuration validation failed: {str(e)}"

    async def validate_tools(self) -> tuple[bool, str | None, str | None]:
        """
        Validate toolkit tools availability.

        Default implementation - can be overridden.

        Returns:
            Tuple of (tools_valid, warning_message, error_message)
        """
        try:
            if hasattr(self, "get_available_tools"):
                tools = self.get_available_tools()

                if not isinstance(tools, list):
                    return False, None, "get_available_tools() does not return list"

                if len(tools) == 0:
                    return True, "No tools available", None

                # Check if tool methods exist
                invalid_tools = []
                for tool in tools:
                    if isinstance(tool, str) and (not hasattr(self, tool) or not callable(getattr(self, tool))):
                            invalid_tools.append(tool)

                if invalid_tools:
                    return False, None, f"Invalid tools: {invalid_tools}"

            return True, None, None

        except Exception as e:
            return False, None, f"Tools validation failed: {str(e)}"

    async def run_full_validation(self) -> dict[str, Any]:
        """
        Run complete validation suite.

        Returns:
            Dictionary with validation results
        """
        results = {
            "toolkit_name": getattr(self, "name", "unknown"),
            "validation_timestamp": None,
            "overall_healthy": True,
            "checks": {},
        }

        # Run all validation checks
        checks = {
            "health": self.validate_health(),
            "configuration": self.validate_configuration(),
            "tools": self.validate_tools(),
        }

        for check_name, check_coro in checks.items():
            try:
                is_valid, warning, error = await check_coro
                results["checks"][check_name] = {
                    "passed": is_valid,
                    "warning": warning,
                    "error": error,
                }

                if not is_valid:
                    results["overall_healthy"] = False

            except Exception as e:
                results["checks"][check_name] = {
                    "passed": False,
                    "warning": None,
                    "error": f"Validation check failed: {str(e)}",
                }
                results["overall_healthy"] = False

        from datetime import datetime

        results["validation_timestamp"] = datetime.now(UTC).isoformat()

        return results
