"""Async client for ROMA-DSPy visualization APIs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class VizApiClient:
    """Lightweight helper to fetch visualization data from the ROMA API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _get(self, path: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()


    async def fetch_metrics(self, execution_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/v1/executions/{execution_id}/metrics")

    async def fetch_lm_traces(self, execution_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        resp = await self._get(
            f"/api/v1/executions/{execution_id}/lm-traces?limit={limit}"
        )
        return resp

    async def fetch_toolkit_metrics(self, execution_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/v1/executions/{execution_id}/toolkit-metrics")

    async def fetch_execution_data(self, execution_id: str) -> Dict[str, Any]:
        """
        Fetch consolidated execution data from the new /data endpoint.

        This endpoint provides real-time trace data with agent executions,
        suitable for live visualization. Includes task hierarchy.

        Returns:
            Dict with keys: execution_id, experiment, tasks, summary, traces, fallback_spans
        """
        return await self._get(f"/api/v1/executions/{execution_id}/data")
