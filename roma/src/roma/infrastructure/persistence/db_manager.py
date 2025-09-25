"""
Database Management CLI for ROMA v2.0
"""

import asyncio
import os
import sys
from pathlib import Path

import click
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from roma.domain.value_objects.config.database_config import DatabaseConfig
from roma.infrastructure.persistence.connection_manager import DatabaseConnectionManager
from roma.infrastructure.persistence.models.base import Base


class DatabaseManager:
    """Database management operations."""

    def __init__(self, database_config: DatabaseConfig):
        """Initialize database manager."""
        self.config = database_config
        self.connection_manager = DatabaseConnectionManager(database_config)

    def get_alembic_config(self) -> Config:
        """Get Alembic configuration."""
        # Get the migrations directory path
        migrations_dir = Path(__file__).parent / "migrations"
        alembic_ini = migrations_dir / "alembic.ini"

        if not alembic_ini.exists():
            raise FileNotFoundError(f"Alembic config not found: {alembic_ini}")

        config = Config(str(alembic_ini))

        # Set database URL from config
        database_url = self.config.get_dsn()
        config.set_main_option("sqlalchemy.url", database_url)

        return config

    async def create_database(self) -> None:
        """Create the database if it doesn't exist."""
        # Connect to default postgres database to create our database
        admin_config = self.config.model_copy(update={"database": "postgres"})
        admin_url = admin_config.get_dsn().replace("postgresql://", "postgresql+asyncpg://")

        engine = create_async_engine(admin_url)

        async with engine.connect() as conn:
            # Use autocommit mode for database creation
            await conn.execution_options(isolation_level="AUTOCOMMIT")

            # Check if database exists
            result = await conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (self.config.database,)
            )

            if not result.fetchone():
                await conn.execute(f'CREATE DATABASE "{self.config.database}"')
                print(f"Created database: {self.config.database}")
            else:
                print(f"Database already exists: {self.config.database}")

        await engine.dispose()

    async def create_tables(self) -> None:
        """Create all tables."""
        await self.connection_manager.initialize()

        engine_url = self.config.get_dsn().replace("postgresql://", "postgresql+asyncpg://")
        engine = create_async_engine(engine_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        await engine.dispose()
        await self.connection_manager.close()

        print("Created all tables")

    async def drop_tables(self) -> None:
        """Drop all tables."""
        await self.connection_manager.initialize()

        engine_url = self.config.get_dsn().replace("postgresql://", "postgresql+asyncpg://")
        engine = create_async_engine(engine_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        await engine.dispose()
        await self.connection_manager.close()

        print("Dropped all tables")

    def init_migrations(self) -> None:
        """Initialize Alembic migrations."""
        try:
            config = self.get_alembic_config()
            command.init(config, str(Path(__file__).parent / "migrations"))
            print("Initialized Alembic migrations")
        except Exception as e:
            if "already exists" in str(e):
                print("Migrations already initialized")
            else:
                raise

    def create_migration(self, message: str, autogenerate: bool = True) -> None:
        """Create a new migration."""
        config = self.get_alembic_config()
        command.revision(config, message=message, autogenerate=autogenerate)
        print(f"Created migration: {message}")

    def upgrade_database(self, revision: str = "head") -> None:
        """Upgrade database to latest or specific revision."""
        config = self.get_alembic_config()
        command.upgrade(config, revision)
        print(f"Upgraded database to revision: {revision}")

    def downgrade_database(self, revision: str) -> None:
        """Downgrade database to specific revision."""
        config = self.get_alembic_config()
        command.downgrade(config, revision)
        print(f"Downgraded database to revision: {revision}")

    def show_current_revision(self) -> None:
        """Show current database revision."""
        config = self.get_alembic_config()
        command.current(config)

    def show_migration_history(self) -> None:
        """Show migration history."""
        config = self.get_alembic_config()
        command.history(config)

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            await self.connection_manager.initialize()
            is_healthy = self.connection_manager.is_healthy()
            await self.connection_manager.close()

            if is_healthy:
                print("✅ Database connection successful")
                return True
            else:
                print("❌ Database connection failed")
                return False

        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False


# CLI Commands
@click.group()
@click.option("--db-host", default="localhost", help="Database host")
@click.option("--db-port", default=5432, help="Database port")
@click.option("--db-name", default="roma_db", help="Database name")
@click.option("--db-user", default="roma_user", help="Database user")
@click.option("--db-password", help="Database password")
@click.pass_context
def db_cli(ctx, db_host, db_port, db_name, db_user, db_password):
    """Database management commands for ROMA v2.0."""
    # Get password from environment if not provided
    if not db_password:
        db_password = os.getenv("ROMA_DB_PASSWORD", "roma_password")

    config = DatabaseConfig(
        host=db_host, port=db_port, database=db_name, user=db_user, password=db_password
    )

    ctx.ensure_object(dict)
    ctx.obj["db_manager"] = DatabaseManager(config)


@db_cli.command()
@click.pass_context
def test_connection(ctx):
    """Test database connection."""
    db_manager = ctx.obj["db_manager"]
    asyncio.run(db_manager.test_connection())


@db_cli.command()
@click.pass_context
def create_db(ctx):
    """Create the database."""
    db_manager = ctx.obj["db_manager"]
    asyncio.run(db_manager.create_database())


@db_cli.command()
@click.pass_context
def create_tables(ctx):
    """Create all database tables."""
    db_manager = ctx.obj["db_manager"]
    asyncio.run(db_manager.create_tables())


@db_cli.command()
@click.pass_context
def drop_tables(ctx):
    """Drop all database tables."""
    db_manager = ctx.obj["db_manager"]
    if click.confirm("Are you sure you want to drop all tables?"):
        asyncio.run(db_manager.drop_tables())


@db_cli.command()
@click.pass_context
def init_migrations(ctx):
    """Initialize Alembic migrations."""
    db_manager = ctx.obj["db_manager"]
    db_manager.init_migrations()


@db_cli.command()
@click.argument("message")
@click.option("--auto/--no-auto", default=True, help="Use autogenerate")
@click.pass_context
def create_migration(ctx, message, auto):
    """Create a new migration."""
    db_manager = ctx.obj["db_manager"]
    db_manager.create_migration(message, autogenerate=auto)


@db_cli.command()
@click.option("--revision", default="head", help="Revision to upgrade to")
@click.pass_context
def upgrade(ctx, revision):
    """Upgrade database schema."""
    db_manager = ctx.obj["db_manager"]
    db_manager.upgrade_database(revision)


@db_cli.command()
@click.argument("revision")
@click.pass_context
def downgrade(ctx, revision):
    """Downgrade database schema."""
    db_manager = ctx.obj["db_manager"]
    if click.confirm(f"Are you sure you want to downgrade to {revision}?"):
        db_manager.downgrade_database(revision)


@db_cli.command()
@click.pass_context
def current(ctx):
    """Show current revision."""
    db_manager = ctx.obj["db_manager"]
    db_manager.show_current_revision()


@db_cli.command()
@click.pass_context
def history(ctx):
    """Show migration history."""
    db_manager = ctx.obj["db_manager"]
    db_manager.show_migration_history()


@db_cli.command()
@click.pass_context
def setup(ctx):
    """Complete database setup (create db, tables, and initial migration)."""
    db_manager = ctx.obj["db_manager"]

    click.echo("Setting up ROMA database...")

    # Step 1: Create database
    click.echo("1. Creating database...")
    asyncio.run(db_manager.create_database())

    # Step 2: Test connection
    click.echo("2. Testing connection...")
    if not asyncio.run(db_manager.test_connection()):
        click.echo("❌ Setup failed: Could not connect to database")
        sys.exit(1)

    # Step 3: Create initial migration
    click.echo("3. Creating initial migration...")
    try:
        db_manager.create_migration("Initial database schema", autogenerate=True)
    except Exception as e:
        click.echo(f"Migration creation failed: {e}")

    # Step 4: Run migration
    click.echo("4. Running migration...")
    try:
        db_manager.upgrade_database()
    except Exception as e:
        click.echo(f"Migration failed: {e}")
        sys.exit(1)

    click.echo("✅ Database setup complete!")


if __name__ == "__main__":
    db_cli()
