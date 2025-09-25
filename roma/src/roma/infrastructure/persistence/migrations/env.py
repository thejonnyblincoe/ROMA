"""
Alembic Environment Configuration for ROMA v2.0
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Import all models to ensure they are registered with Base.metadata
from roma.infrastructure.persistence.models.base import Base

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """
    Get database URL from environment variables or config.

    Priority:
    1. ROMA_DATABASE_URL environment variable
    2. Individual environment variables (ROMA_DB_*)
    3. alembic.ini configuration
    """
    # Check for full database URL
    database_url = os.getenv("ROMA_DATABASE_URL")
    if database_url:
        return database_url

    # Check for individual components
    host = os.getenv("ROMA_DB_HOST", "localhost")
    port = os.getenv("ROMA_DB_PORT", "5432")
    database = os.getenv("ROMA_DB_DATABASE", "roma_db")
    user = os.getenv("ROMA_DB_USER", "roma_user")
    password = os.getenv("ROMA_DB_PASSWORD", "roma_password")

    if any(
        [
            host != "localhost",
            port != "5432",
            database != "roma_db",
            user != "roma_user",
            password != "roma_password",
        ]
    ):
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    # Fall back to config file
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Include object names in migration comparisons
        include_object=lambda object, name, type_, reflected, compare_to: True,
        # Render item for autogenerate
        render_item=lambda type_, obj, autogen_context: False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using async engine."""
    database_url = get_database_url()

    # Convert to async URL if needed
    if not database_url.startswith("postgresql+asyncpg://"):
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        else:
            raise ValueError(f"Unsupported database URL format: {database_url}")

    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
        future=True,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
