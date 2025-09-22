"""CLI interface for ROMA-DSPy."""

import typer

app = typer.Typer()


@app.command()
def solve(task: str):
    """Solve a task using hierarchical decomposition."""
    # TODO: Implement CLI solve
    typer.echo(f"Solving: {task}")


@app.command()
def config():
    """Display configuration."""
    # TODO: Show config
    pass


if __name__ == "__main__":
    app()