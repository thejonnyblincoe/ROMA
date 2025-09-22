"""Telemetry and logging configuration for tracking and observability."""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TelemetryProvider(Enum):
    """Supported telemetry providers."""

    NONE = "none"
    MLFLOW = "mlflow"
    LANGFUSE = "langfuse"
    CONSOLE = "console"


class TelemetryConfig(BaseModel):
    """Telemetry configuration settings."""

    provider: TelemetryProvider = Field(
        default=TelemetryProvider.CONSOLE,
        description="Telemetry provider to use"
    )
    enabled: bool = Field(default=True, description="Enable telemetry")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")

    # MLflow specific
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: Optional[str] = Field(
        default="roma-dspy-experiments",
        description="MLflow experiment name"
    )

    # Langfuse specific
    langfuse_secret_key: Optional[str] = Field(
        default=None,
        description="Langfuse secret key"
    )
    langfuse_public_key: Optional[str] = Field(
        default=None,
        description="Langfuse public key"
    )
    langfuse_host: Optional[str] = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host URL"
    )


class TelemetryManager:
    """Manages telemetry and logging for the application."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        """Initialize telemetry manager."""
        self.config = config or TelemetryConfig()
        self.logger = self._setup_logging()
        self.tracker = self._setup_tracking()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("roma_dspy")
        logger.setLevel(getattr(logging, self.config.log_level))

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(console_handler)

        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(file_handler)

        return logger

    def _setup_tracking(self) -> Optional[Any]:
        """Set up experiment tracking based on provider."""
        if not self.config.enabled:
            return None

        if self.config.provider == TelemetryProvider.MLFLOW:
            try:
                import mlflow

                if self.config.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

                mlflow.set_experiment(self.config.mlflow_experiment_name)
                return mlflow
            except ImportError:
                self.logger.warning("MLflow not installed, falling back to console logging")
                return None

        elif self.config.provider == TelemetryProvider.LANGFUSE:
            try:
                from langfuse import Langfuse

                return Langfuse(
                    secret_key=self.config.langfuse_secret_key,
                    public_key=self.config.langfuse_public_key,
                    host=self.config.langfuse_host,
                )
            except ImportError:
                self.logger.warning("Langfuse not installed, falling back to console logging")
                return None

        return None

    def log_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Log task execution details."""
        self.logger.info(f"Task {task_id}: {task_data}")

        if self.tracker and self.config.provider == TelemetryProvider.MLFLOW:
            self.tracker.log_params({f"task_{task_id}": task_data})
        elif self.tracker and self.config.provider == TelemetryProvider.LANGFUSE:
            self.tracker.trace(
                name=f"task_{task_id}",
                metadata=task_data,
            )

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics."""
        self.logger.info(f"Metrics: {metrics}")

        if self.tracker and self.config.provider == TelemetryProvider.MLFLOW:
            self.tracker.log_metrics(metrics)
        elif self.tracker and self.config.provider == TelemetryProvider.LANGFUSE:
            for key, value in metrics.items():
                self.tracker.score(name=key, value=value)

    def start_run(self, run_name: Optional[str] = None) -> Optional[str]:
        """Start a new tracking run."""
        if self.tracker and self.config.provider == TelemetryProvider.MLFLOW:
            run = self.tracker.start_run(run_name=run_name)
            return run.info.run_id
        return None

    def end_run(self) -> None:
        """End the current tracking run."""
        if self.tracker and self.config.provider == TelemetryProvider.MLFLOW:
            self.tracker.end_run()


# Global telemetry manager instance
telemetry = TelemetryManager()