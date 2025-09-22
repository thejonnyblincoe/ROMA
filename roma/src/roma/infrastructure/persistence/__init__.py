"""
PostgreSQL Persistence Layer for ROMA v2.0

This module provides database persistence capabilities including:
- Event storage and retrieval
- Execution history tracking
- Checkpoint and recovery management
- Database migrations and schema management
"""

from .connection_manager import DatabaseConnectionManager

__all__ = [
    "DatabaseConnectionManager",
]