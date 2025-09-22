"""
Tests for ExecutionConfig - Configuration Limits and Enforcement.

Tests the ExecutionConfig value object and its integration with orchestration
components for enforcing execution limits and timeouts.
"""

import pytest
from roma.domain.value_objects.config.execution_config import ExecutionConfig


class TestExecutionConfig:
    """Test cases for ExecutionConfig."""

    def test_default_configuration(self):
        """Test default ExecutionConfig values."""
        config = ExecutionConfig()

        # Verify default values
        assert config.max_concurrent_tasks == 10
        assert config.max_iterations == 1000
        assert config.max_subtasks_per_node == 20
        assert config.max_tasks_per_level == 50
        assert config.total_timeout == 3600  # 1 hour
        assert config.task_timeout == 300    # 5 minutes
        assert config.retry_on_failure is True
        assert config.enable_partial_aggregation is True

    def test_custom_configuration(self):
        """Test ExecutionConfig with custom values."""
        config = ExecutionConfig(
            max_concurrent_tasks=5,
            max_iterations=50,
            max_subtasks_per_node=5,
            max_tasks_per_level=10,
            total_timeout=1800,  # 30 minutes
            task_timeout=120,    # 2 minutes
            retry_on_failure=False,
            enable_partial_aggregation=False
        )

        assert config.max_concurrent_tasks == 5
        assert config.max_iterations == 50
        assert config.max_subtasks_per_node == 5
        assert config.max_tasks_per_level == 10
        assert config.total_timeout == 1800
        assert config.task_timeout == 120
        assert config.retry_on_failure is False
        assert config.enable_partial_aggregation is False

    def test_immutability(self):
        """Test that ExecutionConfig is immutable."""
        config = ExecutionConfig(max_concurrent_tasks=5)

        # Should not be able to modify frozen fields
        with pytest.raises(TypeError, match="cannot assign to field"):
            config.max_concurrent_tasks = 10

    def test_validation_max_concurrent_tasks(self):
        """Test validation of max_concurrent_tasks."""
        # Valid values
        ExecutionConfig(max_concurrent_tasks=1)
        ExecutionConfig(max_concurrent_tasks=100)

        # Invalid values
        with pytest.raises(ValueError, match="max_concurrent_tasks must be positive"):
            ExecutionConfig(max_concurrent_tasks=0)

        with pytest.raises(ValueError, match="max_concurrent_tasks must be positive"):
            ExecutionConfig(max_concurrent_tasks=-1)

    def test_validation_max_iterations(self):
        """Test validation of max_iterations."""
        # Valid values
        ExecutionConfig(max_iterations=1)
        ExecutionConfig(max_iterations=1000)

        # Invalid values
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            ExecutionConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            ExecutionConfig(max_iterations=-1)

    def test_validation_timeouts(self):
        """Test validation of timeout values."""
        # Valid values
        ExecutionConfig(total_timeout=60, task_timeout=30)

        # Invalid values
        with pytest.raises(ValueError, match="total_timeout must be positive"):
            ExecutionConfig(total_timeout=0)

        with pytest.raises(ValueError, match="task_timeout must be positive"):
            ExecutionConfig(task_timeout=0)

        with pytest.raises(ValueError, match="task_timeout must be less than total_timeout"):
            ExecutionConfig(total_timeout=60, task_timeout=120)

    def test_validation_subtask_limits(self):
        """Test validation of subtask limit values."""
        # Valid values
        ExecutionConfig(max_subtasks_per_node=1, max_tasks_per_level=1)

        # Invalid values
        with pytest.raises(ValueError, match="max_subtasks_per_node must be positive"):
            ExecutionConfig(max_subtasks_per_node=0)

        with pytest.raises(ValueError, match="max_tasks_per_level must be positive"):
            ExecutionConfig(max_tasks_per_level=0)

    def test_serialization(self):
        """Test ExecutionConfig serialization."""
        config = ExecutionConfig(
            max_concurrent_tasks=5,
            max_iterations=50,
            total_timeout=1800,
            enable_recovery=False
        )

        # Test model_dump
        data = config.model_dump()
        expected_fields = {
            "max_concurrent_tasks", "max_iterations", "max_subtasks_per_node",
            "max_tasks_per_level", "total_timeout", "task_timeout",
            "enable_recovery", "enable_aggregation"
        }

        assert set(data.keys()) == expected_fields
        assert data["max_concurrent_tasks"] == 5
        assert data["max_iterations"] == 50
        assert data["total_timeout"] == 1800
        assert data["enable_recovery"] is False

    def test_deserialization(self):
        """Test ExecutionConfig deserialization."""
        data = {
            "max_concurrent_tasks": 3,
            "max_iterations": 25,
            "max_subtasks_per_node": 7,
            "max_tasks_per_level": 15,
            "total_timeout": 900,
            "task_timeout": 180,
            "enable_recovery": True,
            "enable_aggregation": False
        }

        config = ExecutionConfig.model_validate(data)

        assert config.max_concurrent_tasks == 3
        assert config.max_iterations == 25
        assert config.max_subtasks_per_node == 7
        assert config.max_tasks_per_level == 15
        assert config.total_timeout == 900
        assert config.task_timeout == 180
        assert config.enable_recovery is True
        assert config.enable_aggregation is False

    def test_partial_configuration(self):
        """Test ExecutionConfig with partial data (defaults for missing fields)."""
        # Only provide some fields, others should use defaults
        config = ExecutionConfig(
            max_concurrent_tasks=3,
            enable_recovery=False
        )

        assert config.max_concurrent_tasks == 3
        assert config.enable_recovery is False
        # Defaults for unspecified fields
        assert config.max_iterations == 100
        assert config.total_timeout == 3600
        assert config.enable_aggregation is True

    def test_equality(self):
        """Test ExecutionConfig equality comparison."""
        config1 = ExecutionConfig(max_concurrent_tasks=5, max_iterations=50)
        config2 = ExecutionConfig(max_concurrent_tasks=5, max_iterations=50)
        config3 = ExecutionConfig(max_concurrent_tasks=3, max_iterations=50)

        assert config1 == config2
        assert config1 != config3

    def test_hash(self):
        """Test ExecutionConfig hashing."""
        config1 = ExecutionConfig(max_concurrent_tasks=5)
        config2 = ExecutionConfig(max_concurrent_tasks=5)
        config3 = ExecutionConfig(max_concurrent_tasks=3)

        # Same configs should have same hash
        assert hash(config1) == hash(config2)
        # Different configs should have different hash
        assert hash(config1) != hash(config3)

        # Should be usable as dict keys
        config_dict = {config1: "test"}
        assert config_dict[config2] == "test"

    def test_repr(self):
        """Test ExecutionConfig string representation."""
        config = ExecutionConfig(max_concurrent_tasks=5, max_iterations=50)
        repr_str = repr(config)

        assert "ExecutionConfig" in repr_str
        assert "max_concurrent_tasks=5" in repr_str
        assert "max_iterations=50" in repr_str

    def test_performance_settings(self):
        """Test ExecutionConfig with performance-focused settings."""
        # High-performance config
        perf_config = ExecutionConfig(
            max_concurrent_tasks=20,
            max_iterations=1000,
            max_subtasks_per_node=20,
            max_tasks_per_level=50,
            total_timeout=7200,  # 2 hours
            task_timeout=600,    # 10 minutes
            enable_recovery=True,
            enable_aggregation=True
        )

        assert perf_config.max_concurrent_tasks == 20
        assert perf_config.max_iterations == 1000
        assert perf_config.total_timeout == 7200

    def test_conservative_settings(self):
        """Test ExecutionConfig with conservative settings."""
        # Conservative config for resource-constrained environments
        conservative_config = ExecutionConfig(
            max_concurrent_tasks=2,
            max_iterations=20,
            max_subtasks_per_node=3,
            max_tasks_per_level=5,
            total_timeout=600,   # 10 minutes
            task_timeout=60,     # 1 minute
            enable_recovery=True,
            enable_aggregation=False  # Disable to save resources
        )

        assert conservative_config.max_concurrent_tasks == 2
        assert conservative_config.max_iterations == 20
        assert conservative_config.enable_aggregation is False

    def test_timeout_relationships(self):
        """Test timeout relationship validations."""
        # Task timeout must be less than total timeout
        ExecutionConfig(total_timeout=600, task_timeout=300)  # Valid

        # This should raise an error
        with pytest.raises(ValueError):
            ExecutionConfig(total_timeout=300, task_timeout=600)  # Invalid

        # Edge case: equal timeouts should be invalid
        with pytest.raises(ValueError):
            ExecutionConfig(total_timeout=300, task_timeout=300)  # Invalid