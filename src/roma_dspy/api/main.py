"""FastAPI server for ROMA-DSPy (optional dependency)."""

from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[misc]


async def _root() -> dict[str, str]:
    return {"message": "ROMA-DSPy API"}


async def _solve_task(task: str) -> dict[str, str]:
    return {"task": task, "status": "pending"}


def create_app() -> "FastAPI":
    """Instantiate the FastAPI application.

    Raises:
        ImportError: if FastAPI is not installed.
    """

    if FastAPI is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "FastAPI is not installed. Install `roma-dspy[api]` to use the REST server."
        )

    app = FastAPI(title="ROMA-DSPy API")
    app.add_api_route("/", _root, methods=["GET"])
    app.add_api_route("/solve", _solve_task, methods=["POST"])
    return app


app: Optional["FastAPI"] = create_app() if FastAPI is not None else None
