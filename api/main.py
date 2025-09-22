"""FastAPI server for ROMA-DSPy."""

from fastapi import FastAPI

app = FastAPI(title="ROMA-DSPy API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ROMA-DSPy API"}


@app.post("/solve")
async def solve_task(task: str):
    """Solve a task asynchronously."""
    # TODO: Implement async solving
    return {"task": task, "status": "pending"}