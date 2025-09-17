"""
ROMA v2 Package Entry Point.

Allows running ROMA as a module: python -m roma
"""

from src.roma.presentation.cli.main import cli

if __name__ == "__main__":
    cli()