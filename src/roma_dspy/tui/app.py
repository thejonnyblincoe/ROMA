"""Interactive Textual-based visualization for ROMA-DSPy executions."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import re
from datetime import datetime
from textwrap import shorten

from textual import events
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static, TabPane, TabbedContent, Tree

try:  # pragma: no cover - textual compatibility shim
    from textual.widgets import ScrollView  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from textual.widgets.scroll_view import ScrollView  # type: ignore
    except ImportError:
        try:
            from textual.widgets._scrollview import ScrollView  # type: ignore
        except ImportError:
            class ScrollView(Static):  # type: ignore
                """Fallback ScrollView that behaves like Static when unavailable."""

                def scroll_home(self, animate: bool = False) -> None:
                    return
from textual.screen import ModalScreen
from textual.message import Message

from .client import VizApiClient


class DataLoaded(Message):
    """Message emitted when remote data finished loading."""

    def __init__(self, success: bool, error: Optional[str] = None) -> None:
        self.success = success
        self.error = error
        super().__init__()


class RomaVizApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #task-tree {
        width: 40%;
        min-width: 30;
        border: tall $accent;
    }

    #detail-tabs {
        border: tall $accent-darken-1;
        width: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    #trace-container {
        layout: vertical;
        height: 1fr;
    }

    #trace-heading {
        padding: 0 1;
        height: auto;
    }

    #trace-detail {
        border-top: wide $surface-lighten-2;
        padding: 1 1;
        min-height: 10;
        height: 1fr;
        overflow-y: auto;
    }

    #timeline-table {
        height: 60%;
    }

    #timeline-graph {
        border-top: wide $surface-lighten-2;
        padding: 1 0 0 0;
        height: 1fr;
        overflow: hidden;
    }

    #task-info {
        overflow-y: auto;
        padding: 1 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload"),
        ("t", "toggle_io", "Toggle I/O"),
        ("enter", "open_span_modal", "Span Detail"),
    ]

    show_io = reactive(False)

    def __init__(
        self,
        execution_id: str,
        profile: Optional[str] = None,
        base_url: str = "http://localhost:8000",
    ) -> None:
        super().__init__()
        self.execution_id = execution_id
        self.profile = profile
        self.client = VizApiClient(base_url=base_url, profile=profile)

        self.mlflow_data: Dict[str, any] = {}
        self.snapshot: Dict[str, any] = {}
        self.metrics: Dict[str, any] = {}
        self.lm_traces: List[Dict[str, any]] = []

        self.task_lookup: Dict[str, Dict[str, any]] = {}
        self.children_map: Dict[str, List[Dict[str, any]]] = {}
        self.selected_task: Optional[Dict[str, any]] = None
        self.current_spans: List[Dict[str, any]] = []
        self.selected_span_index: Optional[int] = None
        self.available_sources: Dict[str, bool] = {
            "mlflow": False,
            "checkpoint": False,
            "lm_traces": False,
        }
        self.mlflow_warning: Optional[str] = None
        self.execution_summary_entry: Dict[str, Any] = {}
        self.fallback_spans: List[Dict[str, Any]] = []
        self.fallback_by_task: Dict[str, List[Dict[str, Any]]] = {}
        self.span_children_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            yield Tree("Loading…", id="task-tree")
            with TabbedContent(id="detail-tabs"):
                with TabPane("Trace", id="tab-trace"):
                    with Container(id="trace-container"):
                        yield Static("", id="trace-heading")
                        trace_table = DataTable(id="trace-table")
                        trace_table.cursor_type = "row"
                        trace_table.show_cursor = True
                        trace_table.add_columns("Span", "Duration", "Tokens", "Cost", "Model", "Tools")
                        yield trace_table
                        detail = ScrollView(id="trace-detail")
                        detail.can_focus = True
                        detail.update("[dim]Select a span to see details.[/dim]")
                        yield detail
                with TabPane("Timeline", id="tab-timeline"):
                    timeline = DataTable(id="timeline-table")
                    timeline.add_columns("Span", "Start", "Duration", "Tokens", "Cost", "Model")
                    yield timeline
                    graph_view = ScrollView(id="timeline-graph")
                    graph_view.can_focus = True
                    graph_view.update("[dim](no timeline data)[/dim]")
                    yield graph_view
                with TabPane("Task Info", id="tab-info"):
                    yield Static("Select a task to view details", id="task-info")
                with TabPane("Run Summary", id="tab-summary"):
                    yield Static("Loading run summary…", id="summary-info")
                with TabPane("LM Calls", id="tab-lm"):
                    lm_table = DataTable(id="lm-table")
                    lm_table.add_columns("Module", "Model", "Tokens", "Cost", "Latency (ms)", "Preview")
                    yield lm_table
        yield Footer()

    async def on_mount(self) -> None:
        await self._load_data()

    async def action_reload(self) -> None:
        await self._load_data()

    def action_toggle_io(self) -> None:
        self.show_io = not self.show_io
        selected = self.get_selected_task()
        if selected:
            self._render_task_views(selected)
        # Update context panels that mention data sources or toggles
        if getattr(self, "is_mounted", False):
            try:
                self._render_summary_tab()
            except Exception:  # pragma: no cover - UI refresh guard
                self.log("Failed to refresh summary tab after toggle.")

    def _data_sources_summary(self) -> str:
        labels: List[str] = []
        if self.available_sources.get("mlflow"):
            labels.append("MLflow spans")
        elif self.mlflow_warning:
            labels.append("MLflow spans unavailable (fallback active)")
        if self.available_sources.get("checkpoint"):
            label = "Checkpoint snapshot"
            if not self.available_sources.get("mlflow"):
                label += " (primary)"
            labels.append(label)
        if self.available_sources.get("lm_traces"):
            labels.append("MLflow LM traces (tracking server)")
        return ", ".join(labels) if labels else "none"

    async def _load_data(self) -> None:
        tree = self.query_one("#task-tree", Tree)
        tree.root.label = f"Execution {self.execution_id} (loading…)"
        try:
            self.current_spans = []
            summary: Dict[str, Any] = {}
            results = await asyncio.gather(
                self.client.fetch_mlflow_tree(self.execution_id),
                self.client.fetch_snapshot(self.execution_id),
                self.client.fetch_metrics(self.execution_id),
                self.client.fetch_lm_traces(self.execution_id),
                return_exceptions=True,
            )
            mlflow_data, snapshot, metrics, lm_traces = results

            mlflow_error = None
            snapshot_error = None
            metrics_error = None
            traces_error = None

            if isinstance(mlflow_data, Exception):
                mlflow_error = str(mlflow_data)
                self.log(f"MLflow trace fetch failed: {mlflow_error}")
                mlflow_data = {}
            if isinstance(snapshot, Exception):
                snapshot_error = str(snapshot)
                self.log(f"Checkpoint snapshot fetch failed: {snapshot_error}")
                snapshot = {}
            if isinstance(metrics, Exception):
                metrics_error = str(metrics)
                self.log(f"Metrics fetch failed: {metrics_error}")
                metrics = {}
            if isinstance(lm_traces, Exception):
                traces_error = str(lm_traces)
                self.log(f"LM trace fetch failed: {traces_error}")
                lm_traces = []

            if isinstance(mlflow_data, str):
                try:
                    mlflow_data = json.loads(mlflow_data)
                except json.JSONDecodeError:
                    mlflow_data = {}

            if isinstance(snapshot, str):
                try:
                    snapshot = json.loads(snapshot)
                except json.JSONDecodeError:
                    snapshot = {}

            fallback_snapshot = {}
            if isinstance(mlflow_data, dict):
                fallback_snapshot = mlflow_data.get("snapshot") if isinstance(mlflow_data.get("snapshot"), dict) else {}

            if not snapshot and fallback_snapshot:
                snapshot = fallback_snapshot
            elif fallback_snapshot:
                # Merge snapshot fields without overriding existing ones
                snapshot.setdefault("tasks", snapshot.get("tasks") or {})
                snapshot.setdefault("subgraphs", snapshot.get("subgraphs") or {})
                snapshot.setdefault("statistics", snapshot.get("statistics") or {})
                fb_tasks = fallback_snapshot.get("tasks")
                if isinstance(snapshot.get("tasks"), dict) and isinstance(fb_tasks, dict):
                    for tid, task in fb_tasks.items():
                        snapshot["tasks"].setdefault(tid, task)
                fb_subgraphs = fallback_snapshot.get("subgraphs")
                if isinstance(snapshot.get("subgraphs"), dict) and isinstance(fb_subgraphs, dict):
                    for sid, subgraph in fb_subgraphs.items():
                        snapshot["subgraphs"].setdefault(sid, subgraph)

            self.mlflow_data = mlflow_data if isinstance(mlflow_data, dict) else {}
            self.snapshot = snapshot if isinstance(snapshot, dict) else {}
            self.metrics = metrics if isinstance(metrics, dict) else {}
            self.lm_traces = lm_traces if isinstance(lm_traces, list) else []
            self.fallback_spans = []
            self.span_children_map = defaultdict(list)
            self.fallback_by_task = {}
            summary = self.mlflow_data.get("summary") if isinstance(self.mlflow_data, dict) else {}
            if not isinstance(summary, dict):
                summary = {}
            if isinstance(self.mlflow_data, dict):
                raw_fallback = self.mlflow_data.get("fallback_spans")
                if isinstance(raw_fallback, list):
                    self.fallback_spans = [span for span in raw_fallback if isinstance(span, dict)]

            warning = self.mlflow_data.get("warning")
            self.mlflow_warning = warning or mlflow_error
            if traces_error and not self.mlflow_warning:
                self.mlflow_warning = traces_error
            self.available_sources = {
                "mlflow": bool(self.mlflow_data.get("tasks")) or bool(self.fallback_spans),
                "checkpoint": bool(self.snapshot.get("tasks")),
                "lm_traces": bool(self.lm_traces),
            }
            if self.available_sources["mlflow"] and mlflow_error:
                # MLflow returned data but earlier error was captured; treat as warning.
                self.mlflow_warning = mlflow_error

            self._build_task_maps()
            self._populate_tree()
            self._render_summary_tab()
            self.post_message(DataLoaded(True))
        except Exception as exc:  # pragma: no cover - CLI diagnostic
            tree.root.label = f"Execution {self.execution_id} (load failed)"
            tree.root.set_label(f"⚠️ {exc}")
            self.post_message(DataLoaded(False, str(exc)))

    def _build_task_maps(self) -> None:
        tasks = self.mlflow_data.get("tasks") or []
        if isinstance(tasks, dict):
            tasks = list(tasks.values())
        elif not isinstance(tasks, list):
            tasks = []

        self.task_lookup = {}

        def normalize_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
            metrics = metrics or {}
            result = {
                "duration": 0.0,
                "tokens": 0,
                "cost": 0.0,
            }
            for key in ("duration", "tokens", "cost"):
                value = metrics.get(key)
                if value in (None, ""):
                    continue
                try:
                    if key == "tokens":
                        result[key] = int(value)
                    else:
                        result[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return result

        def normalize_spans(spans: Any) -> List[Dict[str, Any]]:
            if not isinstance(spans, list):
                return []
            normalized: List[Dict[str, Any]] = []
            for span in spans:
                if isinstance(span, dict):
                    normalized.append(dict(span))
            return normalized

        # Load MLflow tasks first
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = task.get("task_id")
            if not tid:
                continue
            entry = dict(task)
            entry["metrics"] = normalize_metrics(task.get("metrics"))
            entry["spans"] = normalize_spans(task.get("spans"))
            self.task_lookup[tid] = entry

        snapshot_tasks = {}
        snapshot_subgraphs = {}
        if isinstance(self.snapshot, dict):
            raw_tasks = self.snapshot.get("tasks")
            if isinstance(raw_tasks, dict):
                snapshot_tasks = raw_tasks
            raw_subgraphs = self.snapshot.get("subgraphs")
            if isinstance(raw_subgraphs, dict):
                snapshot_subgraphs = raw_subgraphs

        visited_snapshot: Set[str] = set()

        def merge_snapshot_task(task_data: Dict[str, Any], parent_id: Optional[str]) -> None:
            tid = task_data.get("task_id")
            if not tid or tid in visited_snapshot:
                return
            visited_snapshot.add(tid)

            entry = self.task_lookup.get(tid)
            if entry is None:
                entry = {
                    "task_id": tid,
                    "goal": task_data.get("goal"),
                    "module": task_data.get("module"),
                    "task_type": task_data.get("task_type"),
                    "node_type": task_data.get("node_type"),
                    "status": task_data.get("status"),
                    "result": task_data.get("result"),
                    "parent_task_id": parent_id,
                    "metrics": normalize_metrics(task_data.get("metrics")),
                    "spans": normalize_spans(task_data.get("spans")),
                    "depth": task_data.get("depth"),
                }
                self.task_lookup[tid] = entry
            else:
                # Merge snapshot info into existing MLflow entry
                for key in ("goal", "module", "task_type", "node_type", "status"):
                    if not entry.get(key) and task_data.get(key):
                        entry[key] = task_data[key]
                if task_data.get("result") and not entry.get("result"):
                    entry["result"] = task_data["result"]
                if parent_id and not entry.get("parent_task_id"):
                    entry["parent_task_id"] = parent_id
                entry.setdefault("metrics", normalize_metrics(None))
                snap_metrics = normalize_metrics(task_data.get("metrics"))
                for metric_key in ("duration", "tokens", "cost"):
                    metric_val = snap_metrics.get(metric_key)
                    if metric_val and not entry["metrics"].get(metric_key):
                        entry["metrics"][metric_key] = metric_val
                if not entry.get("spans"):
                    entry["spans"] = normalize_spans(task_data.get("spans"))
                if entry.get("depth") is None and task_data.get("depth") is not None:
                    entry["depth"] = task_data.get("depth")

            subgraph_id = task_data.get("subgraph_id")
            if subgraph_id and subgraph_id in snapshot_subgraphs:
                subgraph = snapshot_subgraphs[subgraph_id] or {}
                subgraph_tasks = subgraph.get("tasks")
                if isinstance(subgraph_tasks, dict):
                    for child in subgraph_tasks.values():
                        if isinstance(child, dict):
                            merge_snapshot_task(child, tid)

        for task in snapshot_tasks.values():
            if isinstance(task, dict):
                merge_snapshot_task(task, task.get("parent_task_id"))

        # Merge LM trace records as pseudo spans / metrics
        if self.lm_traces:
            for call in self.lm_traces:
                if not isinstance(call, dict):
                    continue
                tid = call.get("task_id")
                if not tid:
                    continue
                entry = self.task_lookup.get(tid)
                if entry is None:
                    entry = {
                        "task_id": tid,
                        "goal": None,
                        "module": call.get("module_name"),
                        "task_type": None,
                        "node_type": None,
                        "status": None,
                        "result": None,
                        "parent_task_id": None,
                        "metrics": {"duration": 0.0, "tokens": 0, "cost": 0.0},
                        "spans": [],
                    }
                    self.task_lookup[tid] = entry
                entry.setdefault("metrics", {"duration": 0.0, "tokens": 0, "cost": 0.0})
                entry.setdefault("spans", [])

                try:
                    duration = float(call.get("latency_ms") or 0.0) / 1000.0
                except (TypeError, ValueError):
                    duration = 0.0
                tokens = call.get("total_tokens")
                try:
                    tokens = int(tokens) if tokens not in (None, "") else 0
                except (TypeError, ValueError):
                    tokens = 0
                try:
                    cost = float(call.get("cost_usd") or 0.0)
                except (TypeError, ValueError):
                    cost = 0.0

                entry["metrics"]["duration"] = (entry["metrics"].get("duration") or 0.0) + duration
                entry["metrics"]["tokens"] = (entry["metrics"].get("tokens") or 0) + tokens
                entry["metrics"]["cost"] = (entry["metrics"].get("cost") or 0.0) + cost

                metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
                tool_calls: List[Dict[str, Any]] = []
                seen_calls: Set[str] = set()
                self._collect_tool_calls_from_obj(metadata, tool_calls, seen_calls)
                self._collect_tool_calls_from_obj(call.get("response"), tool_calls, seen_calls)
                self._collect_tool_calls_from_obj(call, tool_calls, seen_calls)

                entry["spans"].append(
                    {
                        "span_id": f"lmtrace-{call.get('trace_id')}",
                        "name": call.get("module_name") or "LM Call",
                        "duration": duration,
                        "tokens": tokens,
                        "cost": cost,
                        "model": call.get("model"),
                        "tool_calls": tool_calls,
                        "inputs": call.get("prompt"),
                        "outputs": call.get("response"),
                        "reasoning": metadata.get("reasoning") if isinstance(metadata, dict) else None,
                        "start_time": call.get("created_at"),
                        "start_ts": call.get("start_ts") or call.get("created_at_ts"),
                    }
                )

        summary = self.mlflow_data.get("summary") if isinstance(self.mlflow_data, dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        span_owner: Dict[str, str] = {}
        for tid, task in self.task_lookup.items():
            for span in task.get("spans") or []:
                sid = span.get("span_id")
                if sid:
                    span_owner[sid] = tid

        goal_to_task: Dict[str, str] = {}
        for tid, task in self.task_lookup.items():
            goal = task.get("goal")
            if isinstance(goal, str) and goal.strip():
                key = goal.strip().lower()
                goal_to_task.setdefault(key, tid)
                if "current number of" in key:
                    goal_to_task.setdefault(
                        key.replace("current number of", "find the current number of").strip(),
                        tid,
                    )

        self.span_children_map = defaultdict(list)
        fallback_records: Dict[str, Dict[str, Any]] = {}
        for idx, raw_span in enumerate(self.fallback_spans):
            if not isinstance(raw_span, dict):
                continue
            span = dict(raw_span)
            span_id = span.get("span_id") or f"__fallback_span_{idx}"
            span.setdefault("span_id", span_id)

            parent_span_id = span.get("parent_span_id") or span.get("parent_id")
            if parent_span_id:
                self.span_children_map[parent_span_id].append(span)

            candidate_task_id = span.get("task_id")
            if candidate_task_id == "__execution__":
                candidate_task_id = None
            if candidate_task_id not in self.task_lookup:
                candidate_task_id = None

            if not candidate_task_id and parent_span_id:
                parent_record = fallback_records.get(parent_span_id)
                if parent_record:
                    candidate_task_id = parent_record["entry"]["task_id"]
                else:
                    owner_tid = span_owner.get(parent_span_id)
                    if owner_tid and owner_tid in self.task_lookup:
                        candidate_task_id = owner_tid

            if not candidate_task_id:
                candidates: List[str] = []
                inputs = span.get("inputs")
                if isinstance(inputs, dict):
                    for key in ("goal", "original_goal", "task_goal"):
                        value = inputs.get(key)
                        if isinstance(value, str) and value.strip():
                            candidates.append(value.strip())
                    sub_results = inputs.get("subtasks_results")
                    if isinstance(sub_results, list):
                        for item in sub_results:
                            if isinstance(item, dict):
                                value = item.get("goal")
                                if isinstance(value, str) and value.strip():
                                    candidates.append(value.strip())
                outputs = span.get("outputs")
                if isinstance(outputs, dict):
                    value = outputs.get("goal")
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())

                for goal_text in candidates:
                    match_tid = goal_to_task.get(goal_text.lower())
                    if match_tid:
                        candidate_task_id = match_tid
                        break

            newly_created = False
            if candidate_task_id and candidate_task_id in self.task_lookup:
                entry = self.task_lookup[candidate_task_id]
            else:
                assigned_id = candidate_task_id or f"__fallback__{idx}"
                entry = self.task_lookup.get(assigned_id)
                if entry is None:
                    newly_created = True
                    entry = {
                        "task_id": assigned_id,
                        "goal": span.get("name") or f"Span {span_id[:8]}",
                        "module": span.get("module"),
                        "task_type": "span",
                        "node_type": "span",
                        "status": span.get("status") or "unknown",
                        "result": None,
                        "parent_task_id": None,
                        "metrics": {"duration": 0.0, "tokens": 0, "cost": 0.0},
                        "spans": [],
                        "is_fallback": True,
                    }
                    self.task_lookup[assigned_id] = entry
                entry.setdefault("is_fallback", True)
                entry.setdefault("goal", entry.get("goal") or span.get("name") or f"Span {span_id[:8]}")
                if not entry.get("module") and span.get("module"):
                    entry["module"] = span.get("module")
                candidate_task_id = entry["task_id"]

            span.setdefault("task_id", candidate_task_id)
            if entry.get("goal") and not span.get("task_goal"):
                span["task_goal"] = entry["goal"]

            existing_ids = {
                sp.get("span_id")
                for sp in (entry.get("spans") or [])
                if isinstance(sp, dict) and sp.get("span_id")
            }
            if span_id not in existing_ids:
                entry.setdefault("spans", []).append(span)
                metrics_entry = entry.setdefault("metrics", {"duration": 0.0, "tokens": 0, "cost": 0.0})
                duration_val = span.get("duration")
                if duration_val not in (None, ""):
                    try:
                        metrics_entry["duration"] += float(duration_val)
                    except (TypeError, ValueError):
                        pass
                token_val = span.get("tokens")
                if token_val not in (None, ""):
                    try:
                        metrics_entry["tokens"] += int(token_val)
                    except (TypeError, ValueError):
                        pass
                cost_val = span.get("cost")
                if cost_val not in (None, ""):
                    try:
                        metrics_entry["cost"] += float(cost_val)
                    except (TypeError, ValueError):
                        pass

            self.fallback_by_task.setdefault(candidate_task_id, []).append(span)
            span_owner.setdefault(span_id, candidate_task_id)
            fallback_records[span_id] = {
                "entry": entry,
                "span": span,
                "is_new": newly_created,
            }

        for record in fallback_records.values():
            if not record.get("is_new"):
                continue
            span = record["span"]
            entry = record["entry"]
            parent_span_id = span.get("parent_span_id") or span.get("parent_id")
            parent_task_id = None
            if parent_span_id:
                parent_record = fallback_records.get(parent_span_id)
                if parent_record:
                    parent_task_id = parent_record["entry"]["task_id"]
                else:
                    parent_task_id = span_owner.get(parent_span_id)
            if parent_task_id == entry["task_id"]:
                parent_task_id = None
            entry["parent_task_id"] = parent_task_id or "__execution__"

        for entry in self.task_lookup.values():
            spans_list = entry.get("spans") or []
            try:
                spans_list.sort(key=lambda sp: self._get_span_start(sp) or 0.0)
            except Exception:
                pass

        aggregated_spans: List[Dict[str, Any]] = []
        for tid, task in self.task_lookup.items():
            for span in task.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                span_copy = dict(span)
                span_copy.setdefault("task_id", tid)
                span_copy.setdefault("task_goal", task.get("goal"))
                aggregated_spans.append(span_copy)

        self.children_map = defaultdict(list)
        for task in self.task_lookup.values():
            parent = task.get("parent_task_id")
            if parent:
                self.children_map[parent].append(task)

        status = None
        if isinstance(self.snapshot, dict):
            status = self.snapshot.get("status")
        if not status and isinstance(self.mlflow_data, dict):
            status = self.mlflow_data.get("status")

        self.execution_summary_entry = {
            "task_id": "__execution__",
            "goal": summary.get("root_goal") or f"Execution {self.execution_id}",
            "module": None,
            "task_type": "execution",
            "node_type": "root",
            "status": status or "unknown",
            "result": None,
            "parent_task_id": None,
            "metrics": {
                "duration": summary.get("total_duration", 0.0) or 0.0,
                "tokens": summary.get("total_tokens", 0) or 0,
                "cost": summary.get("total_cost", 0.0) or 0.0,
            },
            "spans": aggregated_spans,
            "is_execution_summary": True,
        }

        root_candidates: List[Tuple[int, str, str]] = []
        for tid, task in self.task_lookup.items():
            parent = task.get("parent_task_id")
            if parent == "__execution__":
                continue
            depth = task.get("depth", 0) or 0
            goal = task.get("goal") or ""
            if not parent or parent not in self.task_lookup:
                root_candidates.append((depth, goal, tid))
        root_candidates.sort()
        if root_candidates:
            self.root_task_ids = [tid for _, _, tid in root_candidates]
        else:
            self.root_task_ids = list(self.task_lookup.keys())

        # Update summary with latest metrics (including fallback tasks)
        summary = summary or {}
        summary["total_tasks"] = len(self.task_lookup)
        summary["total_spans"] = sum(len(task.get("spans") or []) for task in self.task_lookup.values())
        summary["total_duration"] = sum(
            task.get("metrics", {}).get("duration", 0.0) or 0.0 for task in self.task_lookup.values()
        )
        summary["total_tokens"] = sum(
            task.get("metrics", {}).get("tokens", 0) or 0 for task in self.task_lookup.values()
        )
        summary["total_cost"] = sum(
            task.get("metrics", {}).get("cost", 0.0) or 0.0 for task in self.task_lookup.values()
        )
        self.mlflow_data["summary"] = summary

    def _populate_tree(self) -> None:
        tree = self.query_one("#task-tree", Tree)
        tree.clear()
        tree.root.label = f"Execution {self.execution_id}"
        tree.root.data = self.execution_summary_entry or None
        if not self.task_lookup:
            tree.root.add("No task hierarchy available (using MLflow)")
            tree.root.expand()
            if tree.root.data:
                self._render_task_views(tree.root.data)
            return

        for root_id in sorted(self.root_task_ids):
            root_task = self.task_lookup[root_id]
            node = tree.root.add(self._task_label(root_task), data=root_task)
            self._populate_children(node, root_task)
        for child in sorted(self.children_map.get("__execution__", []), key=lambda t: t.get("goal", "")):
            node = tree.root.add(self._task_label(child), data=child)
            self._populate_children(node, child)
        tree.root.expand()
        if tree.root.data:
            tree.focus()
            tree.select_node(tree.root)
            self._render_task_views(tree.root.data)
        elif tree.root.children:
            first = tree.root.children[0]
            first.expand()
            tree.focus()
            tree.select_node(first)
            if first.data:
                self._render_task_views(first.data)

    def _populate_children(self, node, task: Dict[str, any]) -> None:
        children = self.children_map.get(task["task_id"], [])
        for child in sorted(children, key=lambda t: (t.get("depth", 0), t.get("goal", ""))):
            child_node = node.add(self._task_label(child), data=child)
            self._populate_children(child_node, child)

    def _task_label(self, task: Dict[str, any]) -> str:
        goal = task.get("goal") or f"Task {task['task_id'][:8]}"
        status_bits = []
        metrics = task.get("metrics", {})
        duration = metrics.get("duration")
        if duration:
            status_bits.append(f"{duration:.2f}s")
        tokens = metrics.get("tokens")
        if tokens:
            status_bits.append(f"{tokens} tok")
        module = task.get("module")
        tag = f"[{module}] " if module else ""
        info = f" ({', '.join(status_bits)})" if status_bits else ""
        return f"{tag}{goal[:60]}{info}"

    def get_selected_task(self) -> Optional[Dict[str, any]]:
        tree = self.query_one("#task-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            return tree.cursor_node.data
        return None

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        task = event.node.data
        if task:
            self._render_task_views(task)

    def _render_task_views(self, task: Dict[str, any]) -> None:
        self.selected_task = task
        self._render_trace_tab(task)
        self._render_timeline(task)
        self._render_task_info(task)
        self._render_lm_table(task)

    def _resolve_event_index(self, row_key: Any, event: Any, default: int = 0) -> int:
        if row_key is not None:
            if isinstance(row_key, int):
                return row_key
            if hasattr(row_key, "value"):
                try:
                    return int(row_key.value)
                except (TypeError, ValueError):
                    value = getattr(row_key, "value", None)
                    if isinstance(value, str) and value.isdigit():
                        return int(value)
            if isinstance(row_key, str):
                try:
                    return int(row_key)
                except (TypeError, ValueError):
                    pass
        candidate = getattr(event, "row_index", None)
        if candidate is None:
            candidate = getattr(event, "cursor_row", None)
        if candidate is None:
            candidate = getattr(event, "row", None)
        if candidate is None:
            return default
        try:
            return int(candidate)
        except (TypeError, ValueError):
            try:
                return int(str(candidate))
            except (TypeError, ValueError):
                return default

    def action_open_span_modal(self) -> None:
        if not self.current_spans:
            return
        index = self.selected_span_index or 0
        if index < 0 or index >= len(self.current_spans):
            index = 0
        self._show_span_detail(self.current_spans[index])

    def _get_task_spans(self, task: Dict[str, any]) -> List[Dict[str, any]]:
        spans = task.get("spans") or []
        if spans:
            return self._attach_fallback_children(spans)

        task_id = task.get("task_id")
        if not task_id:
            return []

        if task_id in self.fallback_by_task:
            return self._attach_fallback_children(self._format_fallback_spans(self.fallback_by_task[task_id]))

        fallback: List[Dict[str, any]] = []
        for call in self.lm_traces:
            if call.get("task_id") != task_id:
                continue

            metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
            tool_calls: List[Dict[str, any]] = []
            seen_calls: Set[str] = set()
            self._collect_tool_calls_from_obj(metadata, tool_calls, seen_calls)
            self._collect_tool_calls_from_obj(call.get("response"), tool_calls, seen_calls)
            self._collect_tool_calls_from_obj(call, tool_calls, seen_calls)

            latency_ms = call.get("latency_ms") or 0
            try:
                duration = float(latency_ms) / 1000.0
            except (TypeError, ValueError):
                duration = 0.0

            raw_cost = call.get("cost_usd")
            try:
                cost = float(raw_cost) if raw_cost not in (None, "") else None
            except (TypeError, ValueError):
                cost = None

            raw_start_ts = call.get("start_ts") or call.get("created_at_ts")
            try:
                start_ts = float(raw_start_ts) if raw_start_ts not in (None, "") else None
            except (TypeError, ValueError):
                start_ts = None

            fallback.append(
                {
                    "span_id": f"lmtrace-{call.get('trace_id')}",
                    "name": call.get("module_name") or "LM Call",
                    "duration": duration,
                    "tokens": call.get("total_tokens"),
                    "cost": cost,
                    "model": call.get("model"),
                    "tool_calls": tool_calls,
                    "inputs": call.get("prompt"),
                    "outputs": call.get("response"),
                    "reasoning": metadata.get("reasoning") if isinstance(metadata, dict) else None,
                    "start_time": call.get("created_at"),
                    "start_ts": start_ts,
                }
            )
        return self._attach_fallback_children(fallback)

    def _format_fallback_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for idx, span in enumerate(spans):
            if not isinstance(span, dict):
                continue
            entry = dict(span)
            entry.setdefault("span_id", f"fallback-{idx}")
            formatted.append(entry)
        return formatted

    def _render_trace_tab(self, task: Dict[str, any]) -> None:
        heading = self.query_one("#trace-heading", Static)
        table = self.query_one("#trace-table", DataTable)
        detail = self.query_one("#trace-detail", ScrollView)
        if task.get("is_execution_summary"):
            goal_text = task.get("goal") or f"Execution {self.execution_id}"
            heading_text = f"Execution: {goal_text}"
        else:
            goal_text = task.get("goal") or task.get("task_id") or "(unknown task)"
            heading_text = f"Task: {goal_text}"
        heading.update(self._escape_text(heading_text))
        table.clear()
        spans = self._get_task_spans(task)
        ordered_spans = self._ordered_span_tree(spans)
        self.current_spans = []
        self.selected_span_index = None

        if not ordered_spans:
            table.add_row("(no spans)", "", "", "", "", "", key="0")
            detail.update("[dim]No spans available for this task.[/dim]")
            try:
                detail.scroll_home(animate=False)
            except Exception:
                pass
            return

        rows: List[Dict[str, any]] = []
        for idx, (span, depth) in enumerate(ordered_spans):
            tool_count = len(span.get("tool_calls") or [])
            duration = span.get("duration")
            duration_display = f"{duration:.2f}s" if isinstance(duration, (int, float)) and duration else ""
            cost = span.get("cost")
            cost_display = ""
            if cost not in (None, ""):
                try:
                    cost_display = f"${float(cost):.4f}"
                except (TypeError, ValueError):
                    cost_display = str(cost)

            rows.append(span)
            base_label = self._span_label(task, span, width=max(12, 40 - depth * 2))
            indent = "  " * depth
            label = f"{indent}{base_label}"
            table.add_row(
                label,
                duration_display,
                str(span.get("tokens") or "") if span.get("tokens") not in (None, "") else "",
                cost_display,
                span.get("model") or "",
                str(tool_count) if tool_count else "",
                key=str(idx),
            )

        self.current_spans = rows
        self.selected_span_index = 0
        self._update_trace_detail(rows[0])

    def _escape_text(self, value: Any) -> str:
        if value is None:
            return ""
        return escape(str(value))

    def _stringify_raw(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                text = json.dumps(value, indent=2)
            except (TypeError, ValueError):
                text = str(value)
        else:
            text = str(value)
        return text

    def _stringify_block(self, value: Any) -> str:
        return self._escape_text(self._stringify_raw(value))

    def _short_snippet(self, value: Any, width: int = 80) -> str:
        text = self._stringify_raw(value)
        text = " ".join(text.split())
        if len(text) <= width:
            return self._escape_text(text)
        truncated = text[: max(1, width - 1)] + "…"
        return self._escape_text(truncated)

    def _collect_tool_calls_from_obj(self, obj: Any, sink: List[Dict[str, Any]], seen: Set[str]) -> None:
        if isinstance(obj, dict):
            if isinstance(obj.get("tool_calls"), list):
                for call in obj["tool_calls"]:
                    if isinstance(call, dict):
                        marker = json.dumps(call, sort_keys=True, default=str)
                        if marker not in seen:
                            seen.add(marker)
                            sink.append(call)
            if isinstance(obj.get("tool_call"), dict):
                call = obj["tool_call"]
                marker = json.dumps(call, sort_keys=True, default=str)
                if marker not in seen:
                    seen.add(marker)
                    sink.append(call)
            func_call = obj.get("function_call")
            if isinstance(func_call, dict):
                call = {
                    "type": "function",
                    "function": func_call,
                }
                marker = json.dumps(call, sort_keys=True, default=str)
                if marker not in seen:
                    seen.add(marker)
                    sink.append(call)
            for value in obj.values():
                self._collect_tool_calls_from_obj(value, sink, seen)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_tool_calls_from_obj(item, sink, seen)

    def _attach_fallback_children(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = list(spans)
        seen: Set[str] = set()

        for span in spans:
            sid = span.get("span_id")
            if sid:
                seen.add(sid)

        def include_children(span_id: Optional[str]) -> None:
            if not span_id:
                return
            for child in self.span_children_map.get(span_id, []):
                child_id = child.get("span_id") or f"child-{id(child)}"
                if child_id in seen:
                    continue
                seen.add(child_id)
                combined.append(child)
                include_children(child.get("span_id"))

        for span in spans:
            include_children(span.get("span_id"))
        return combined

    def _is_wrapper_name(self, span: Dict[str, Any]) -> bool:
        if not isinstance(span, dict):
            return False
        span_type = str(span.get("span_type") or "").lower()
        if span_type == "module_wrapper":
            return True
        name = str(span.get("name") or "").lower()
        return name in {"executor", "agent executor"}

    def _ordered_span_tree(self, spans: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
        if not spans:
            return []
        by_id: Dict[str, Dict[str, Any]] = {}
        for span in spans:
            sid = span.get("span_id")
            if sid:
                by_id[sid] = span
        children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        roots: List[Dict[str, Any]] = []

        def sort_key(span: Dict[str, Any]) -> Tuple[float, str]:
            return (self._get_span_start(span) or 0.0, span.get("name", ""))

        for span in spans:
            parent = span.get("parent_span_id") or span.get("parent_id")
            if parent and parent in by_id:
                children[parent].append(span)
            else:
                roots.append(span)

        for child_list in children.values():
            child_list.sort(key=sort_key)
        roots.sort(key=sort_key)

        ordered: List[Tuple[Dict[str, Any], int]] = []

        def visit(span: Dict[str, Any], depth: int) -> None:
            sid = span.get("span_id")
            has_children = bool(sid and children.get(sid))
            is_wrapper = depth == 0 and has_children and self._is_wrapper_name(span)
            next_depth = depth if is_wrapper else depth + 1
            if not is_wrapper:
                ordered.append((span, depth))
            sid = span.get("span_id")
            for child in children.get(sid, []):
                visit(child, next_depth)

        for root in roots:
            visit(root, 0)
        return ordered

    def _span_label(self, task: Dict[str, Any], span: Dict[str, Any], width: int = 36) -> str:
        label = self._escape_text(span.get("name", "span"))
        if task.get("is_execution_summary") and span.get("task_goal"):
            goal = self._short_snippet(span["task_goal"], width=width)
            label = f"{label} — {goal}"
        return label

    def _summarize_tool_call(self, call: Any, width: int = 80) -> str:
        if not isinstance(call, dict):
            return self._short_snippet(call, width=width)

        toolkit = call.get("toolkit") or call.get("toolkit_class")
        name = call.get("tool") or call.get("tool_name") or call.get("name") or call.get("type")

        func = call.get("function")
        if isinstance(func, dict):
            func_name = func.get("name")
            if func_name:
                name = func_name if not name else f"{name}.{func_name}"
            arguments = func.get("arguments")
        else:
            arguments = call.get("arguments") or call.get("args") or call.get("input")

        if toolkit and name:
            name = f"{toolkit}.{name}"

        output = call.get("output") or call.get("result") or call.get("return")

        summary = self._escape_text(name or "tool")
        if arguments not in (None, ""):
            summary += f"({self._short_snippet(arguments, width=width // 2)})"
        if output not in (None, ""):
            summary += f" → {self._short_snippet(output, width=width // 2)}"
        return summary

    def _format_tool_call(self, call: Dict[str, any]) -> str:
        if not isinstance(call, dict):
            return f"    {self._short_snippet(call)}"

        lines: List[str] = []
        summary = self._summarize_tool_call(call, width=120)
        lines.append(f"    {summary}")

        def append_block(label: str, value: Any) -> None:
            if value in (None, ""):
                return
            block = self._stringify_block(value)
            block_lines = block.splitlines() or [block]
            lines.append(f"      {label}:")
            for line in block_lines:
                lines.append(f"        {line}")

        # Include detailed sections if available
        if isinstance(call.get("function"), dict):
            func = call["function"]
            append_block("Arguments", func.get("arguments"))
        else:
            append_block("Arguments", call.get("arguments") or call.get("args") or call.get("input"))
        append_block("Output", call.get("output") or call.get("result") or call.get("return"))
        metadata = {
            k: v
            for k, v in call.items()
            if k not in ("function", "arguments", "args", "input", "output", "result", "return", "tool_calls")
        }
        if metadata:
            append_block("Metadata", metadata)

        return "\n".join(lines)

    def _format_span_detail(self, span: Dict[str, any]) -> str:
        lines: List[str] = []
        lines.append(f"[bold]Span:[/bold] {self._escape_text(span.get('name', 'span'))}")
        if span.get("task_goal"):
            lines.append(f"[bold]Task Goal:[/bold] {self._short_snippet(span['task_goal'], width=160)}")
        if span.get("task_id") and (self.selected_task or {}).get("is_execution_summary"):
            lines.append(f"[bold]Task ID:[/bold] {self._escape_text(span['task_id'])}")
        if span.get("start_time"):
            lines.append(f"[bold]Start:[/bold] {self._escape_text(span['start_time'])}")
        duration = span.get("duration")
        if duration not in (None, ""):
            try:
                lines.append(f"[bold]Duration:[/bold] {float(duration):.4f}s")
            except (TypeError, ValueError):
                lines.append(f"[bold]Duration:[/bold] {self._escape_text(duration)}")
        if span.get("tokens") not in (None, ""):
            lines.append(f"[bold]Tokens:[/bold] {self._escape_text(span['tokens'])}")
        cost = span.get("cost")
        if cost not in (None, ""):
            try:
                lines.append(f"[bold]Cost:[/bold] ${float(cost):.6f}")
            except (TypeError, ValueError):
                lines.append(f"[bold]Cost:[/bold] {self._escape_text(cost)}")
        if span.get("model"):
            lines.append(f"[bold]Model:[/bold] {self._escape_text(span['model'])}")

        tool_calls = span.get("tool_calls") or []
        if tool_calls:
            lines.append("")
            lines.append("[bold]Tool Calls:[/bold]")
            for call in tool_calls:
                formatted = self._format_tool_call(call)
                if formatted:
                    lines.append(f"[dim]{formatted}[/dim]")

        if self.show_io:
            inputs = span.get("inputs")
            if inputs not in (None, ""):
                lines.append("")
                lines.append("[bold]Input:[/bold]")
                block = self._stringify_block(inputs)
                lines.extend(block.splitlines())

            outputs = span.get("outputs")
            if outputs not in (None, ""):
                lines.append("")
                lines.append("[bold]Output:[/bold]")
                block = self._stringify_block(outputs)
                lines.extend(block.splitlines())

            reasoning = span.get("reasoning")
            if reasoning not in (None, ""):
                lines.append("")
                lines.append("[bold]Reasoning:[/bold]")
                block = self._stringify_block(reasoning)
                lines.extend(block.splitlines())

        return "\n".join(lines) if lines else "[dim]No details for this span.[/dim]"

    def _update_trace_detail(self, span: Optional[Dict[str, any]]) -> None:
        detail = self.query_one("#trace-detail", ScrollView)
        if not span:
            detail.update("[dim]Select a span to see its inputs, outputs, and tool calls.[/dim]")
        else:
            formatted = self._format_span_detail(span)
            try:
                detail.update(formatted)
            except Exception as exc:
                self.log(f"Trace detail markup error: {exc}")
                plain = re.sub(r"\[(\/?)((?:bold|dim))\]", "", formatted)
                detail.update(Text(plain))
        try:
            detail.scroll_home(animate=False)
        except Exception:
            pass

    def _parse_timestamp(self, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
                    return datetime.fromisoformat(cleaned).timestamp()
                except (ValueError, TypeError):
                    return None
        return None

    def _get_span_start(self, span: Dict[str, any]) -> Optional[float]:
        start_ts = self._parse_timestamp(span.get("start_ts"))
        if start_ts is not None:
            return start_ts
        return self._parse_timestamp(span.get("start_time"))

    def _get_span_duration(self, span: Dict[str, any]) -> float:
        duration = span.get("duration")
        if duration in (None, ""):
            return 0.0
        try:
            return float(duration)
        except (TypeError, ValueError):
            return 0.0

    def _render_timeline(self, task: Dict[str, any]) -> None:
        table = self.query_one("#timeline-table", DataTable)
        graph = self.query_one("#timeline-graph", ScrollView)
        table.clear()
        raw_spans = self._get_task_spans(task)
        child_map: Dict[str, int] = defaultdict(int)
        for span in raw_spans:
            parent_sid = span.get("parent_span_id") or span.get("parent_id")
            if parent_sid:
                child_map[parent_sid] += 1
        spans = []
        for span in raw_spans:
            sid = span.get("span_id")
            has_children = bool(sid and child_map.get(sid))
            if has_children and self._is_wrapper_name(span):
                continue
            spans.append(span)
        spans.sort(key=lambda sp: self._get_span_start(sp) or 0)
        lines: List[str] = []
        if not spans:
            table.add_row("(no spans)", "", "", "", "", "")
            graph.update("[dim](no timeline data)[/dim]")
            try:
                graph.scroll_home(animate=False)
            except Exception:
                pass
            return

        max_duration = max((self._get_span_duration(span) or 0) for span in spans) or 1.0
        for span in spans:
            name = self._span_label(task, span, width=40)
            cost = span.get("cost")
            if cost not in (None, ""):
                try:
                    cost_display = f"${float(cost):.4f}"
                except (TypeError, ValueError):
                    cost_display = str(cost)
            else:
                cost_display = ""
            table.add_row(
                name,
                span.get("start_time") or "",
                f"{self._get_span_duration(span):.2f}s" if self._get_span_duration(span) else "",
                str(span.get('tokens') or ""),
                cost_display,
                span.get('model') or "",
            )
        has_start = any(self._get_span_start(span) is not None for span in spans)
        label_width = 22
        graph_width = 48
        if has_start:
            starts = [self._get_span_start(span) for span in spans if self._get_span_start(span) is not None]
            earliest = min(starts) if starts else 0.0
            ends: List[float] = []
            for span in spans:
                start = self._get_span_start(span) or earliest
                duration = self._get_span_duration(span)
                ends.append(start + duration)
            max_end = max(ends) if ends else earliest + max_duration
            total_span = max(max_end - earliest, max_duration)
            if total_span <= 0:
                total_span = max_duration or 1.0
            total_label = f"{total_span:.2f}s"
            spacer = max(0, graph_width - len(total_label) - 2)
            lines.append(" " * (label_width + 1) + f"[dim]0s{' ' * spacer}{total_label}[/dim]")
            missing_start = False
            for span in spans:
                name = self._span_label(task, span, width=label_width - 1)
                start = self._get_span_start(span)
                if start is None:
                    missing_start = True
                offset_cols = 0
                if start is not None:
                    offset_cols = int(((start - earliest) / total_span) * graph_width)
                offset_cols = max(0, min(offset_cols, graph_width - 1))
                duration = self._get_span_duration(span)
                width_cols = max(1, int((duration / total_span) * graph_width)) if duration else 1
                if offset_cols + width_cols > graph_width:
                    width_cols = max(1, graph_width - offset_cols)
                bar = (" " * offset_cols + "█" * width_cols).ljust(graph_width)
                lines.append(f"{name:<{label_width}} {bar} {duration:.2f}s")
            if missing_start:
                lines.append("[dim]* spans without start time are aligned to 0s.[/dim]")
        else:
            for span in spans:
                duration = self._get_span_duration(span)
                width_cols = max(1, int((duration / max_duration) * graph_width)) if duration else 1
                bar = ("█" * width_cols).ljust(graph_width)
                name = self._span_label(task, span, width=label_width - 1)
                lines.append(f"{name:<{label_width}} {bar} {duration:.2f}s")
            lines.append("[dim]Start timestamps unavailable; bars scaled by duration only.[/dim]")
        graph.update("\n".join(lines) if lines else "[dim](no timeline data)[/dim]")
        try:
            graph.scroll_home(animate=False)
        except Exception:
            pass

    def _render_task_info(self, task: Dict[str, any]) -> None:
        info = self.query_one("#task-info", Static)
        metrics = task.get("metrics", {})
        spans = self._get_task_spans(task)
        modules = set()
        if task.get("module"):
            modules.add(task.get("module"))
        for span in spans:
            name = span.get("name")
            if name:
                modules.add(name.split('.')[0])
        if task.get("is_execution_summary"):
            id_line = f"[bold]Execution ID:[/bold] {self._escape_text(self.execution_id)}"
            parent_text = "—"
        else:
            id_line = f"[bold]Task ID:[/bold] {self._escape_text(task['task_id'])}"
            parent_text = task.get("parent_task_id") or "ROOT"
        parent_render = self._escape_text(parent_text)
        module_render = self._escape_text(task.get("module") or "-")
        goal_render = self._escape_text(task.get("goal") or "(unknown)")

        lines = [
            id_line,
            f"[bold]Parent:[/bold] {parent_render}",
            f"[bold]Data Sources:[/bold] {self._escape_text(self._data_sources_summary())}",
            f"[bold]Module:[/bold] {module_render}",
            f"[bold]Task Type / Node Type:[/bold] {task.get('task_type') or '-'} / {task.get('node_type') or '-'}",
            f"[bold]Status:[/bold] {task.get('status') or 'unknown'}",
            f"[bold]Modules Seen:[/bold] {self._escape_text(', '.join(sorted(modules))) if modules else '-'}",
            f"[bold]Goal:[/bold] {goal_render}",
        ]

        if task.get("is_execution_summary"):
            lines.append(f"[bold]Root Tasks:[/bold] {len(self.root_task_ids)}")
            lines.append(f"[bold]Tasks Loaded:[/bold] {len(self.task_lookup)}")

        lines.extend([
            "",
            f"[bold]Duration:[/bold] {metrics.get('duration', 0.0):.2f}s",
            f"[bold]Tokens:[/bold] {metrics.get('tokens', 0)}",
            f"[bold]Cost:[/bold] ${metrics.get('cost', 0.0):.4f}",
        ])

        result_text = task.get("result")
        if result_text:
            lines.append("")
            lines.append("[bold]Result:[/bold]")
            lines.append(self._short_snippet(result_text, width=200))

        tool_summaries: List[str] = []
        for span in spans:
            for call in span.get("tool_calls") or []:
                tool_summaries.append(self._summarize_tool_call(call, width=80))

        if tool_summaries:
            lines.append("")
            lines.append("[bold]Tool Calls:[/bold]")
            for summary in tool_summaries[:5]:
                lines.append(f"• {summary}")
            if len(tool_summaries) > 5:
                lines.append(f"… ({len(tool_summaries) - 5} more)")

        lines.append("")
        lines.append(
            f"[dim]Press 't' to toggle detailed span I/O (currently {'ON' if self.show_io else 'OFF'}). "
            "Highlight spans in the Trace tab to inspect them below the table, or press Enter for a pop-out view.[/dim]"
        )
        if self.mlflow_warning:
            lines.append(f"[dim]MLflow note: {self._escape_text(self.mlflow_warning)}[/dim]")

        content = "\n".join(lines)
        try:
            info.update(content)
        except Exception as exc:
            self.log(f"Task info markup error: {exc}")
            plain = re.sub(r"\[(\/?)((?:bold|dim))\]", "", content)
            info.update(Text(plain))

    def _render_lm_table(self, task: Dict[str, any]) -> None:
        table = self.query_one("#lm-table", DataTable)
        table.clear()
        if task.get("is_execution_summary"):
            traces = list(self.lm_traces)
        else:
            task_id = task.get("task_id")
            traces = [lt for lt in self.lm_traces if lt.get("task_id") == task_id]
        if not traces:
            table.add_row("(none)", "", "", "", "", "")
            return
        for call in traces:
            preview = self._short_snippet(call.get("prompt"), width=80) if call.get("prompt") else ""
            if self.show_io and call.get("response"):
                preview = self._short_snippet(call.get("response"), width=80)
            table.add_row(
                call.get("module_name") or "",
                call.get("model") or "",
                str(call.get("total_tokens") or ""),
                f"${float(call.get('cost_usd') or 0.0):.4f}",
                str(call.get("latency_ms") or ""),
                preview,
            )

    def _render_summary_tab(self) -> None:
        summary = self.mlflow_data.get("summary", {})
        info = self.query_one("#summary-info", Static)
        lines = [
            f"[bold]Execution:[/bold] {self._escape_text(self.execution_id)}",
            f"[bold]Experiment:[/bold] {self._escape_text(self.mlflow_data.get('experiment') or '-')}",
            "",
            f"[bold]Total Tasks:[/bold] {summary.get('total_tasks', '-')}",
            f"[bold]Total Spans:[/bold] {summary.get('total_spans', '-')}",
            f"[bold]Total Duration:[/bold] {summary.get('total_duration', 0.0):.2f}s",
            f"[bold]Total Tokens:[/bold] {summary.get('total_tokens', 0)}",
            f"[bold]Total Cost:[/bold] ${summary.get('total_cost', 0.0):.4f}",
            "",
            f"[bold]Data Sources:[/bold] {self._escape_text(self._data_sources_summary())}",
        ]
        metrics = self.metrics or {}
        if metrics:
            lines.append("")
            lines.append("[bold]Checkpoint Metrics[/bold]")
            lines.append(f"Total LM Calls: {metrics.get('total_lm_calls', '-')}")
            lines.append(f"Average Latency: {metrics.get('average_latency_ms', 0.0):.2f} ms")
            lines.append(f"Total Tokens (DB): {metrics.get('total_tokens', 0)}")
            lines.append(f"Total Cost (DB): ${metrics.get('total_cost_usd', 0.0):.4f}")

        if self.mlflow_data.get("fallback_spans"):
            lines.append("")
            lines.append("[dim]Some tasks were reconstructed from checkpoints (no MLflow spans available).[/dim]")
        content = "\n".join(lines)
        try:
            info.update(content)
        except Exception as exc:
            self.log(f"Summary markup error: {exc}")
            plain = re.sub(r"\[(\/?)((?:bold|dim))\]", "", content)
            info.update(Text(plain))

    async def on_data_loaded(self, message: DataLoaded) -> None:
        if message.success:
            task = self.get_selected_task()
            if task:
                self._render_task_views(task)
        else:
            summary = self.query_one("#summary-info", Static)
            summary.update(f"Failed to load data: {message.error}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:  # pragma: no cover - UI
        if event.data_table.id != "trace-table":
            return
        if not self.current_spans:
            return
        idx = self._resolve_event_index(event.row_key, event, default=0)
        if idx < 0 or idx >= len(self.current_spans):
            return
        self.selected_span_index = idx
        self._update_trace_detail(self.current_spans[idx])
        self._show_span_detail(self.current_spans[idx])

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:  # pragma: no cover - UI
        if event.data_table.id != "trace-table":
            return
        if not self.current_spans:
            return
        idx = self._resolve_event_index(event.row_key, event, default=self.selected_span_index or 0)
        if idx < 0 or idx >= len(self.current_spans):
            return
        self.selected_span_index = idx
        self._update_trace_detail(self.current_spans[idx])

    def _show_span_detail(self, span: Dict[str, any]) -> None:
        body = self._format_span_detail(span)
        self.push_screen(SpanDetailModal(span.get('name', 'Span Detail'), body))


def run_viz_app(execution_id: str, profile: Optional[str] = None, base_url: str = "http://localhost:8000") -> None:
    """Helper to launch the Textual TUI."""
    app = RomaVizApp(execution_id=execution_id, profile=profile, base_url=base_url)
    app.run()
class SpanDetailModal(ModalScreen[None]):
    """Modal dialog showing detailed information for a span."""

    CSS = """
    Screen {
        align: center middle;
    }
    #span-detail-container {
        width: 90%;
        max-width: 100;
        border: tall $accent;
        background: $panel;
        padding: 1 2;
    }
    #span-detail-body {
        height: auto;
        overflow: auto;
    }
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.dialog_body = body

    def compose(self) -> ComposeResult:
        title_text = Text(self.dialog_title, style="bold")
        body_text = Text()
        try:
            body_text = Text.from_markup(self.dialog_body)
        except Exception:
            body_text = Text(self.dialog_body)
        content = Text()
        content.append_text(title_text)
        content.append("\n\n")
        content.append_text(body_text)
        yield Container(
            Static(content, id="span-detail-body"),
            id="span-detail-container"
        )

    def on_key(self, event: events.Key) -> None:  # pragma: no cover - UI only
        if event.key in ("escape", "q", "enter"):
            self.dismiss(None)
