"""
Database Connection Manager for PostgreSQL Persistence Layer
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncpg
from asyncpg import Pool, Connection

from roma.domain.value_objects.config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Manages PostgreSQL connections and connection pooling.

    Features:
    - Connection pooling with asyncpg
    - Health checks and monitoring
    - Automatic reconnection
    - Transaction management
    - Connection lifecycle tracking
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[Pool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False
        self._last_health_check: Optional[datetime] = None
        self._connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "query_count": 0,
            "transaction_count": 0,
        }
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            logger.warning("Connection pool already initialized")
            return

        try:
            logger.info("Initializing PostgreSQL connection pool")

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                **self.config.get_asyncpg_kwargs(),
                min_size=self.config.pool.min_size,
                max_size=self.config.pool.max_size,
                max_queries=50000,  # asyncpg default
                max_inactive_connection_lifetime=self.config.max_connection_age,
                setup=self._setup_connection,
                init=self._init_connection,
            )

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            # Initial health check
            await self._perform_health_check()

            logger.info(
                f"PostgreSQL connection pool initialized: "
                f"{self.config.pool.min_size}-{self.config.pool.max_size} connections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool and cleanup resources."""
        logger.info("Closing PostgreSQL connection pool")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close connection pool
        if self._pool:
            await self._pool.close()
            self._pool = None

        self._is_healthy = False
        logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """
        Get a connection from the pool.

        Returns:
            Async context manager yielding a database connection
        """
        if not self._pool:
            raise RuntimeError("Connection pool not initialized")

        connection = None
        try:
            async with self._lock:
                self._connection_stats["total_connections"] += 1
                self._connection_stats["active_connections"] += 1

            connection = await self._pool.acquire()
            yield connection

        except Exception as e:
            async with self._lock:
                self._connection_stats["failed_connections"] += 1
            logger.error(f"Database connection error: {e}")
            raise

        finally:
            if connection:
                await self._pool.release(connection)
                async with self._lock:
                    self._connection_stats["active_connections"] -= 1

    @asynccontextmanager
    async def get_transaction(self):
        """
        Get a database transaction.

        Returns:
            Async context manager yielding a database transaction
        """
        async with self.get_connection() as conn:
            async with conn.transaction():
                async with self._lock:
                    self._connection_stats["transaction_count"] += 1
                yield conn

    async def execute_query(
        self,
        query: str,
        *args: Any,
        connection: Optional[Connection] = None
    ) -> Any:
        """
        Execute a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            connection: Optional existing connection to use

        Returns:
            Query result
        """
        async with self._lock:
            self._connection_stats["query_count"] += 1

        if connection:
            return await connection.fetch(query, *args)
        else:
            async with self.get_connection() as conn:
                return await conn.fetch(query, *args)

    async def execute_many(
        self,
        query: str,
        args_list: list,
        connection: Optional[Connection] = None
    ) -> None:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query to execute
            args_list: List of parameter tuples
            connection: Optional existing connection to use
        """
        async with self._lock:
            self._connection_stats["query_count"] += len(args_list)

        if connection:
            await connection.executemany(query, args_list)
        else:
            async with self.get_connection() as conn:
                await conn.executemany(query, args_list)

    async def _setup_connection(self, connection: Connection) -> None:
        """Setup callback for new connections."""
        # Set timezone to UTC
        await connection.execute("SET timezone = 'UTC'")

        # Set statement timeout
        timeout_ms = int(self.config.pool.query_timeout * 1000)
        await connection.execute(f"SET statement_timeout = {timeout_ms}")

    async def _init_connection(self, connection: Connection) -> None:
        """Init callback for connections."""
        # Register custom types if needed
        pass

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self._is_healthy = False

    async def _perform_health_check(self) -> None:
        """Perform database health check."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    self._is_healthy = True
                    self._last_health_check = datetime.now(timezone.utc)
                else:
                    self._is_healthy = False
                    logger.warning("Database health check returned unexpected result")

        except Exception as e:
            self._is_healthy = False
            logger.error(f"Database health check failed: {e}")

    def is_healthy(self) -> bool:
        """Check if the database connection is healthy."""
        return self._is_healthy

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        pool_stats = {}
        if self._pool:
            pool_stats = {
                "pool_size": self._pool.get_size(),
                "pool_min_size": self._pool.get_min_size(),
                "pool_max_size": self._pool.get_max_size(),
                "pool_idle_size": self._pool.get_idle_size(),
            }

        return {
            **self._connection_stats,
            **pool_stats,
            "is_healthy": self._is_healthy,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()