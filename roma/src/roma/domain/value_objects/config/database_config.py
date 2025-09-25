"""
Database Configuration Value Objects for ROMA v2.0 PostgreSQL Persistence
"""

import os
from typing import Any

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class DatabasePoolConfig:
    """PostgreSQL connection pool configuration."""

    min_size: int = 10
    max_size: int = 50
    command_timeout: float = 60.0
    query_timeout: float = 30.0
    connection_timeout: float = 10.0
    max_cached_statement_lifetime: int = 300
    max_cacheable_statement_size: int = 1024


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database configuration."""

    # Connection parameters
    host: str = "localhost"
    port: int = 5432
    database: str = "roma_db"
    user: str = "roma_user"
    password: str = "roma_password"

    # SSL and security
    ssl_mode: str = "prefer"
    ssl_cert: str | None = None
    ssl_key: str | None = None
    ssl_ca: str | None = None

    # Connection pool settings
    pool: DatabasePoolConfig = Field(default_factory=DatabasePoolConfig)

    # Performance settings
    max_connections: int = 100
    statement_cache_size: int = 1024

    # Monitoring and health checks
    health_check_interval: float = 30.0
    max_connection_age: int = 3600

    # Event store specific settings
    event_retention_days: int = 90
    checkpoint_retention_days: int = 30
    batch_size: int = 1000

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Database host cannot be empty")
        return v.strip()

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1 or v > 65535:
            raise ValueError(f"Port must be 1-65535, got: {v}")
        return v

    @field_validator("database", "user", "password")
    @classmethod
    def validate_required_fields(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Database connection field cannot be empty")
        return v.strip()

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v: str) -> str:
        valid_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in valid_modes:
            raise ValueError(f"SSL mode must be one of {valid_modes}, got: {v}")
        return v

    def get_connection_string_from_env(self) -> str | None:
        """Get connection string from environment, following SecurityConfig pattern."""
        return os.getenv("ROMA_DATABASE_URL")

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN connection string."""
        dsn_parts = [
            f"postgresql://{self.user}:{self.password}",
            f"@{self.host}:{self.port}/{self.database}",
        ]

        params = []
        if self.ssl_mode != "disable":
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert:
            params.append(f"sslcert={self.ssl_cert}")
        if self.ssl_key:
            params.append(f"sslkey={self.ssl_key}")
        if self.ssl_ca:
            params.append(f"sslrootcert={self.ssl_ca}")

        if params:
            dsn_parts.append("?" + "&".join(params))

        return "".join(dsn_parts)

    def get_asyncpg_kwargs(self) -> dict[str, Any]:
        """Get kwargs for asyncpg connection."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "ssl": self.ssl_mode if self.ssl_mode != "disable" else False,
            "command_timeout": self.pool.command_timeout,
            "query_timeout": self.pool.query_timeout,
            "timeout": self.pool.connection_timeout,
            "statement_cache_size": self.statement_cache_size,
            "max_cached_statement_lifetime": self.pool.max_cached_statement_lifetime,
            "max_cacheable_statement_size": self.pool.max_cacheable_statement_size,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "ssl_mode": self.ssl_mode,
            "ssl_cert": self.ssl_cert,
            "ssl_key": self.ssl_key,
            "ssl_ca": self.ssl_ca,
            "pool": {
                "min_size": self.pool.min_size,
                "max_size": self.pool.max_size,
                "command_timeout": self.pool.command_timeout,
                "query_timeout": self.pool.query_timeout,
                "connection_timeout": self.pool.connection_timeout,
                "max_cached_statement_lifetime": self.pool.max_cached_statement_lifetime,
                "max_cacheable_statement_size": self.pool.max_cacheable_statement_size,
            },
            "max_connections": self.max_connections,
            "statement_cache_size": self.statement_cache_size,
            "health_check_interval": self.health_check_interval,
            "max_connection_age": self.max_connection_age,
            "event_retention_days": self.event_retention_days,
            "checkpoint_retention_days": self.checkpoint_retention_days,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        pool_data = data.get("pool", {})
        pool_config = DatabasePoolConfig(
            min_size=pool_data.get("min_size", 10),
            max_size=pool_data.get("max_size", 50),
            command_timeout=pool_data.get("command_timeout", 60.0),
            query_timeout=pool_data.get("query_timeout", 30.0),
            connection_timeout=pool_data.get("connection_timeout", 10.0),
            max_cached_statement_lifetime=pool_data.get("max_cached_statement_lifetime", 300),
            max_cacheable_statement_size=pool_data.get("max_cacheable_statement_size", 1024),
        )

        return cls(
            host=data.get("host", "localhost"),
            port=data.get("port", 5432),
            database=data.get("database", "roma_db"),
            user=data.get("user", "roma_user"),
            password=data.get("password", "roma_password"),
            ssl_mode=data.get("ssl_mode", "prefer"),
            ssl_cert=data.get("ssl_cert"),
            ssl_key=data.get("ssl_key"),
            ssl_ca=data.get("ssl_ca"),
            pool=pool_config,
            max_connections=data.get("max_connections", 100),
            statement_cache_size=data.get("statement_cache_size", 1024),
            health_check_interval=data.get("health_check_interval", 30.0),
            max_connection_age=data.get("max_connection_age", 3600),
            event_retention_days=data.get("event_retention_days", 90),
            checkpoint_retention_days=data.get("checkpoint_retention_days", 30),
            batch_size=data.get("batch_size", 1000),
        )
