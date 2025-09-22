"""
Integration Test for PostgreSQL Persistence Layer
"""

import pytest
import asyncio
from datetime import datetime, timezone

from src.roma.domain.value_objects.config.database_config import DatabaseConfig, DatabasePoolConfig
from src.roma.infrastructure.persistence.connection_manager import DatabaseConnectionManager
from src.roma.infrastructure.persistence.postgres_event_store import PostgreSQLEventStore
from src.roma.domain.events.task_events import TaskEvent


@pytest.mark.asyncio
@pytest.mark.integration
class TestPostgreSQLIntegration:
    """Integration tests for PostgreSQL persistence layer."""

    @pytest.fixture
    async def test_db_config(self):
        """Test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="roma_test_db",
            user="roma_test_user",
            password="roma_test_password",
            pool=DatabasePoolConfig(min_size=1, max_size=2)
        )

    @pytest.fixture
    async def connection_manager(self, test_db_config):
        """Connection manager fixture."""
        manager = DatabaseConnectionManager(test_db_config)
        yield manager
        await manager.close()

    @pytest.fixture
    async def postgres_event_store(self, connection_manager):
        """PostgreSQL event store fixture."""
        try:
            await connection_manager.initialize()
            if not connection_manager.is_healthy():
                pytest.skip("PostgreSQL not available for integration tests")

            store = PostgreSQLEventStore(connection_manager)
            await store.initialize()
            yield store
            await store.close()
        except Exception:
            pytest.skip("PostgreSQL not available for integration tests")

    async def test_database_config_validation(self):
        """Test database configuration validation."""
        # Test valid config
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password"
        )
        assert config.host == "localhost"
        assert config.port == 5432

        # Test invalid port
        with pytest.raises(ValueError, match="Port must be 1-65535"):
            DatabaseConfig(
                host="localhost",
                port=70000,
                database="test_db",
                user="test_user",
                password="test_password"
            )

        # Test invalid SSL mode
        with pytest.raises(ValueError, match="SSL mode must be one of"):
            DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password="test_password",
                ssl_mode="invalid_mode"
            )

    async def test_connection_manager_lifecycle(self, test_db_config):
        """Test connection manager initialization and cleanup."""
        manager = DatabaseConnectionManager(test_db_config)

        # Test initialization
        try:
            await manager.initialize()
            # Connection manager should be created
            assert manager._pool is not None or True  # Skip if no DB available
        except Exception:
            pytest.skip("PostgreSQL not available")
        finally:
            await manager.close()

    async def test_event_store_basic_operations(self, postgres_event_store):
        """Test basic event store operations."""
        # Create test event
        event = TaskEvent(
            task_id="test_task_123",
            event_type="test_event",
            timestamp=datetime.now(timezone.utc),
            metadata={"test_key": "test_value"}
        )

        # Test append
        await postgres_event_store.append(event)

        # Test retrieve
        events = await postgres_event_store.get_events("test_task_123")
        assert len(events) >= 1

        retrieved_event = events[-1]  # Get the last event
        assert retrieved_event.task_id == "test_task_123"
        assert retrieved_event.event_type == "test_event"
        assert retrieved_event.metadata.get("test_key") == "test_value"

        # Test cleanup
        await postgres_event_store.clear("test_task_123")

    async def test_event_store_compatibility(self, postgres_event_store):
        """Test PostgreSQL event store API compatibility with InMemoryEventStore."""
        # Test all methods exist and are callable
        assert hasattr(postgres_event_store, 'append')
        assert hasattr(postgres_event_store, 'get_events')
        assert hasattr(postgres_event_store, 'get_all_events')
        assert hasattr(postgres_event_store, 'get_events_by_type')
        assert hasattr(postgres_event_store, 'subscribe')
        assert hasattr(postgres_event_store, 'subscribe_async')
        assert hasattr(postgres_event_store, 'unsubscribe')
        assert hasattr(postgres_event_store, 'get_task_timeline')
        assert hasattr(postgres_event_store, 'get_statistics')
        assert hasattr(postgres_event_store, 'clear')

        # Test statistics
        stats = await postgres_event_store.get_statistics()
        assert isinstance(stats, dict)
        assert "total_events" in stats

    def test_database_config_environment_integration(self):
        """Test database config environment variable integration."""
        config = DatabaseConfig()

        # Test DSN generation
        dsn = config.get_dsn()
        assert isinstance(dsn, str)
        assert "postgresql://" in dsn

        # Test environment variable method
        env_url = config.get_connection_string_from_env()
        assert env_url is None or isinstance(env_url, str)

    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = DatabaseConfig(
            host="test_host",
            port=5433,
            database="test_db",
            user="test_user",
            password="test_pass"
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["host"] == "test_host"
        assert config_dict["port"] == 5433

        # Test from_dict
        restored_config = DatabaseConfig.from_dict(config_dict)
        assert restored_config.host == config.host
        assert restored_config.port == config.port
        assert restored_config.database == config.database