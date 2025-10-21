"""Interactive Textual-based visualization for ROMA-DSPy executions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

# Setup file logging for debugging crashes
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/roma_tui_app.log'),
    ]
)
logger = logging.getLogger(__name__)

from textual import events
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.cache import LRUCache
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static, TabPane, TabbedContent, Tree
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.message import Message

from .client import VizApiClient
from .models import AgentGroupViewModel, ExecutionViewModel, TaskViewModel, TraceViewModel
from .transformer import DataTransformer
from .widgets.tree_table import TreeTable
from .detail_view import (
    GenericDetailModal,
    LMCallDetailParser,
    SpanDetailParser,
    ToolCallDetailParser,
)


# ROMA Logo ASCII Art (from file)
DEBUG_LOG_PATH = Path("/tmp/tui_watermark_debug.log")


def _debug_log(message: str) -> None:
    """Write verbose debugging information to a temp log file."""
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as debug_file:
            debug_file.write(f"{datetime.utcnow().isoformat()}Z {message}\n")
    except Exception:
        # Avoid crashing the TUI if the debug log can't be written
        pass


def _load_ascii_art(asset_name: str) -> str:
    """Load ASCII art asset by name from the project assets directory."""
    try:
        asset_path = Path(__file__).parent.parent.parent.parent / "assets" / asset_name
        if asset_path.exists():
            content = asset_path.read_text(encoding="utf-8").strip()
            _debug_log(f"Loaded asset '{asset_name}' ({len(content)} chars) from {asset_path}")
            return content
        _debug_log(f"Asset '{asset_name}' missing at {asset_path}")
    except Exception as exc:
        # Textual UI should continue rendering even if the logo is unavailable
        _debug_log(f"Failed loading asset '{asset_name}': {exc}")
        return ""
    return ""


SENTIENT_LOGO = _load_ascii_art("sentient_logo_text_ascii.txt")
ROMA_LOGO = _load_ascii_art("roma_logo_simple_visible.txt")


class DataLoaded(Message):
    """Message emitted when remote data finished loading."""

    def __init__(self, success: bool, error: Optional[str] = None) -> None:
        self.success = success
        self.error = error
        super().__init__()


class WelcomeScreen(ModalScreen[None]):
    """Welcome/splash screen shown while data loads."""

    CSS = """
    WelcomeScreen {
        align: center middle;
        background: $surface;
    }

    #welcome-container {
        width: auto;
        height: auto;
        padding: 2 4;
        background: $surface;
        border: tall $accent;
    }

    #welcome-logo {
        color: $accent;
        text-style: bold;
        text-align: center;
    }

    #welcome-message {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self, execution_id: str) -> None:
        super().__init__()
        self.execution_id = execution_id
        self.data_loaded = False

    def compose(self) -> ComposeResult:
        """Compose the welcome screen with Sentient and ROMA logos."""
        # Show Sentient logo, then ROMA logo underneath
        sentient_logo = SENTIENT_LOGO if SENTIENT_LOGO else "SENTIENT"
        roma_logo = ROMA_LOGO if ROMA_LOGO else "ROMA-DSPy"

        combined_logo = f"{sentient_logo}\n\n{roma_logo}"

        with Container(id="welcome-container"):
            yield Static(combined_logo, id="welcome-logo")
            yield Static(
                f"Loading execution {self.execution_id[:8]}...",
                id="welcome-message"
            )

    def on_key(self, event: events.Key) -> None:
        """Handle key press - dismiss on Enter after data loads, quit on Q."""
        if event.key == "enter" and self.data_loaded:
            self.dismiss()
        elif event.key == "q":
            # Allow quitting from welcome screen
            self.app.exit()

    def mark_data_loaded(self) -> None:
        """Update message to prompt for user input."""
        self.data_loaded = True
        message_widget = self.query_one("#welcome-message", Static)
        message_widget.update("Press [bold]Enter[/bold] to continue or [bold]Q[/bold] to quit")


class RomaVizApp(App[None]):
    """Sentient ROMA-DSPy Interactive Visualizer - Hierarchical Task Execution Explorer"""

    TITLE = "🔷 Sentient ROMA-DSPy Visualizer"
    SUB_TITLE = "Hierarchical Task Execution Explorer"

    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        background: $accent;
        color: $text;
    }

    Header .header--title {
        color: $text;
        text-style: bold;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #task-tree {
        width: 40%;
        min-width: 30;
        border: tall $accent;
        background: $surface;
    }

    #detail-tabs-wrapper {
        width: 1fr;
        height: 100%;
    }

    #detail-tabs {
        border: tall $accent-darken-1;
        width: 100%;
        height: 100%;
    }

    #spans-wrapper {
        height: 100%;
        layout: vertical;
    }

    DataTable {
        height: 1fr;
    }

    #spans-container {
        layout: vertical;
        height: 100%;
        min-height: 30;
    }

    #spans-heading {
        height: auto;
        padding: 0 1;
    }

    #spans-table {
        height: 50%;
        min-height: 15;
        overflow-y: auto;
    }

    #timeline-graph {
        height: 50%;
        min-height: 15;
        border-top: wide $surface-lighten-2;
        padding: 1 1;
        overflow-y: auto;
        overflow-x: auto;
        scrollbar-gutter: stable;
    }

    #task-info {
        width: 100%;
        padding: 1 1;
    }

    #summary-info {
        width: 100%;
        padding: 1 1;
    }

    Footer {
        background: $accent-darken-2;
    }

    .footer--highlight {
        color: $accent;
    }

    .footer--key {
        background: $accent;
    }

    #logo-footer {
        color: $accent 40%;
        text-style: bold;
    }

    #welcome-screen {
        align: center middle;
        background: $surface;
    }

    #welcome-logo {
        width: auto;
        height: auto;
        color: $accent;
        text-style: bold;
        text-align: center;
        padding: 2 4;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload"),
        ("t", "toggle_io", "Toggle I/O"),
        ("l", "toggle_live", "Toggle Live"),
        ("enter", "open_span_modal", "Span Detail"),
    ]

    show_io = reactive(False)
    live_mode = reactive(False)

    def __init__(
        self,
        execution_id: str,
        base_url: str = "http://localhost:8000",
        live: bool = False,
        poll_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self.execution_id = execution_id
        self.client = VizApiClient(base_url=base_url)
        self.live_mode = live
        self.poll_interval = poll_interval

        # NEW: Clean view model (no messy lookups!)
        self.execution: Optional[ExecutionViewModel] = None
        self.transformer = DataTransformer()

        # UI state
        self.selected_task: Optional[TaskViewModel] = None
        # Note: current_spans and selected_span_index removed - TreeTable stores spans in node.data["span_obj"]

        # LM table row mapping (row_key -> trace object)
        self._lm_table_row_map: Dict[Any, TraceViewModel] = {}

        # Tool calls table row mapping (row_key -> tool call dict)
        self._tool_table_row_map: Dict[Any, Dict[str, Any]] = {}

        # Live mode state
        self._poll_task: Optional[asyncio.Task] = None
        self._last_update: Optional[datetime] = None

        # Performance optimizations
        self._span_tree_cache: LRUCache[str, List[Tuple[TraceViewModel, int]]] = LRUCache(maxsize=20)
        self._active_render_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            yield Tree("Loading…", id="task-tree")

            # Wrapper container for tabs
            with Container(id="detail-tabs-wrapper"):
                # Content with tabs
                with TabbedContent(id="detail-tabs"):
                    with TabPane("Spans", id="tab-spans"):
                        with Container(id="spans-container"):
                            yield Static("", id="spans-heading")
                            yield Static("", id="spans-summary")  # Summary row (TOTAL)

                            spans_table = TreeTable(
                                columns=["Start Time", "Duration", "Model", "Tools"],
                                id="spans-table",
                            )
                            # Note: "Span" is the tree column (implicit)
                            # show_header=True and zebra_stripes=True are defaults in TreeTable
                            yield spans_table

                            with VerticalScroll(id="timeline-graph"):
                                yield Static("[dim](no timeline data)[/dim]", id="timeline-graph-content")
                    with TabPane("Task Info", id="tab-info"):
                        with VerticalScroll():
                            yield Static("Select a task to view details", id="task-info")
                    with TabPane("Run Summary", id="tab-summary"):
                        with VerticalScroll():
                            yield Static("Loading run summary…", id="summary-info")
                    with TabPane("LM Calls", id="tab-lm"):
                        lm_table = DataTable(id="lm-table", cursor_type="row")
                        lm_table.add_columns("Module", "Model", "Tokens", "Latency (ms)", "Preview")
                        yield lm_table
                    with TabPane("Tool Calls", id="tab-tools"):
                        tool_table = DataTable(id="tool-table", cursor_type="row")
                        tool_table.add_columns("Tool", "Toolkit", "Duration (ms)", "Status", "Preview")
                        yield tool_table
        yield Footer()

    async def on_mount(self) -> None:
        # Show welcome screen with loading task running in background
        welcome_screen = WelcomeScreen(self.execution_id)

        async def load_and_prompt() -> None:
            """Load data then prompt user to continue."""
            await self._load_data()
            # Update welcome screen to show "Press Enter to continue"
            welcome_screen.mark_data_loaded()
            # Start live polling if enabled (but don't dismiss screen yet)
            if self.live_mode:
                await self._start_live_polling()

        # Push welcome screen and start loading in background
        self.push_screen(welcome_screen)
        # Schedule data loading
        asyncio.create_task(load_and_prompt())

    async def on_unmount(self) -> None:
        """Stop live polling when app closes."""
        await self._stop_live_polling()

    async def action_reload(self) -> None:
        await self._load_data()

    async def action_toggle_live(self) -> None:
        """Toggle live mode on/off."""
        self.live_mode = not self.live_mode
        if self.live_mode:
            await self._start_live_polling()
        else:
            await self._stop_live_polling()
        # Update header to show live mode status
        self._update_live_mode_indicator()

    async def action_toggle_io(self) -> None:
        try:
            self.show_io = not self.show_io
            self.log(f"Toggle I/O: show_io = {self.show_io}")
            logger.info(f"TOGGLE: show_io={self.show_io}")

            # Cancel any in-progress render to prevent race conditions
            if self._active_render_task and not self._active_render_task.done():
                self._active_render_task.cancel()
                try:
                    await self._active_render_task
                except asyncio.CancelledError:
                    pass  # Expected
                except Exception:
                    pass  # Ignore errors from cancelled task

            selected = self.get_selected_task()
            if selected:
                logger.info(f"TOGGLE: Rendering task views for task {selected.task_id[:8]}")
                await self._render_task_views_async(selected)
                logger.info(f"TOGGLE: Task views rendered successfully")
            else:
                # Only update execution info when no task is selected
                if getattr(self, "is_mounted", False):
                    try:
                        logger.info("TOGGLE: Rendering execution info")
                        self._render_execution_info()
                    except Exception as exc:  # pragma: no cover - UI refresh guard
                        self.log(f"Failed to refresh execution info after toggle: {exc}")
                        logger.error(f"TOGGLE ERROR: Execution info render failed: {exc}")

            # Always update summary tab
            if getattr(self, "is_mounted", False):
                try:
                    logger.info("TOGGLE: Rendering summary tab")
                    self._render_summary_tab()
                except Exception as exc:  # pragma: no cover - UI refresh guard
                    self.log(f"Failed to refresh summary tab after toggle: {exc}")
                    logger.error(f"TOGGLE ERROR: Summary tab render failed: {exc}")

            logger.info("TOGGLE: Completed successfully")
        except asyncio.CancelledError:
            # Gracefully handle cancellation from rapid toggling - DO NOT re-raise or app will exit
            self.log("Toggle I/O was cancelled (rapid toggle detected)")
            logger.warning("TOGGLE: Cancelled (rapid toggle)")
            return  # Just return, don't crash the app
        except Exception as exc:
            self.log(f"Critical error in action_toggle_io: {exc}")
            logger.error(f"TOGGLE CRITICAL ERROR: {exc}", exc_info=True)
            import traceback
            self.log(traceback.format_exc())
            # Don't crash the app, just log the error

    def _data_sources_summary(self) -> str:
        """Generate summary of available data sources."""
        if not self.execution:
            return "none"

        labels: List[str] = []
        sources = self.execution.data_sources

        if sources.get("mlflow"):
            labels.append("MLflow spans")
        if sources.get("checkpoint"):
            label = "Checkpoint snapshot"
            if not sources.get("mlflow"):
                label += " (primary)"
            labels.append(label)
        if sources.get("lm_traces") and not sources.get("mlflow"):
            labels.append("LM traces (fallback)")

        if self.execution.warnings:
            labels.append(f"({len(self.execution.warnings)} warnings)")

        return ", ".join(labels) if labels else "none"

    async def _load_data(self) -> None:
        """
        Simplified data loading - transformer handles all complexity!

        OLD: 100+ lines of manual merging
        NEW: 30 lines, transformer does the work
        """
        tree = self.query_one("#task-tree", Tree)
        tree.root.label = f"Execution {self.execution_id} (loading…)"
        try:
            # 1. Fetch raw data (parallel)
            # FIXED: Using new /data endpoint which includes task hierarchy
            # This preserves all agent executions instead of losing them
            results = await asyncio.gather(
                self.client.fetch_execution_data(self.execution_id),
                self.client.fetch_lm_traces(self.execution_id),
                self.client.fetch_metrics(self.execution_id),
                return_exceptions=True,
            )
            mlflow_data, lm_traces, metrics = results

            # Handle errors
            if isinstance(mlflow_data, Exception):
                self.log(f"MLflow fetch failed: {mlflow_data}")
                mlflow_data = {}
            if isinstance(lm_traces, Exception):
                self.log(f"LM traces fetch failed: {lm_traces}")
                lm_traces = []
            if isinstance(metrics, Exception):
                self.log(f"Metrics fetch failed: {metrics}")
                metrics = {}

            # Parse JSON if needed
            if isinstance(mlflow_data, str):
                try:
                    mlflow_data = json.loads(mlflow_data)
                except json.JSONDecodeError:
                    mlflow_data = {}

            # Build checkpoint_data (empty since we don't have checkpoint data)
            # Transformer will use mlflow_data for task structure and spans
            checkpoint_data = {
                "execution_id": self.execution_id,
                "tasks": {},  # Empty - let transformer extract from mlflow_data
                "root_goal": mlflow_data.get("summary", {}).get("root_goal", "") if isinstance(mlflow_data, dict) else "",
                "status": "unknown",
                "checkpoints": []
            }

            # 2. Transform to clean view model (MAGIC HAPPENS HERE!)
            self.execution = self.transformer.transform(
                mlflow_data=mlflow_data if isinstance(mlflow_data, dict) else {},
                checkpoint_data=checkpoint_data,
                lm_traces=lm_traces if isinstance(lm_traces, list) else [],
                metrics=metrics if isinstance(metrics, dict) else {},
            )

            # DEBUG
            debug_file = "/tmp/tui_debug.log"
            with open(debug_file, "w") as f:
                f.write(f"=== DATA LOADED ===\n")
                f.write(f"Execution has {len(self.execution.tasks)} tasks\n")
                for task_id, task in self.execution.tasks.items():
                    f.write(f"  Task {task_id[:8]}: {len(task.traces)} traces\n")

            # 3. Render UI
            self._populate_tree()
            self._render_summary_tab()
            self.post_message(DataLoaded(True))

        except Exception as exc:  # pragma: no cover - CLI diagnostic
            tree.root.label = f"Execution {self.execution_id} (load failed)"
            tree.root.set_label(f"⚠️ {exc}")
            self.post_message(DataLoaded(False, str(exc)))

    async def _start_live_polling(self) -> None:
        """Start the live polling background task."""
        if self._poll_task and not self._poll_task.done():
            return  # Already polling

        self.log("Starting live polling...")
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _stop_live_polling(self) -> None:
        """Stop the live polling background task."""
        if self._poll_task and not self._poll_task.done():
            self.log("Stopping live polling...")
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Background polling loop for live updates."""
        while True:
            try:
                await asyncio.sleep(self.poll_interval)
                await self._load_data()
                self._last_update = datetime.now()
                self._update_live_mode_indicator()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log(f"Polling error: {e}")
                # Continue polling even on errors

    def _update_live_mode_indicator(self) -> None:
        """Update the tree label to show live mode status."""
        try:
            tree = self.query_one("#task-tree", Tree)
            status = "🔴 LIVE" if self.live_mode else ""
            last_update = f" (updated {self._last_update:%H:%M:%S})" if self._last_update else ""
            tree.root.label = f"{status} Execution {self.execution_id[:8]}...{last_update}"
        except Exception:
            pass  # UI not ready yet

    def _populate_tree(self) -> None:
        """
        Simplified tree population - uses clean view models.

        OLD: Complex lookups in task_lookup, children_map, etc.
        NEW: Simple iteration over self.execution.tasks
        """
        tree = self.query_one("#task-tree", Tree)
        tree.clear()
        self._update_live_mode_indicator()  # Set label with live status

        if not self.execution or not self.execution.tasks:
            tree.root.add("No tasks available")
            tree.root.expand()
            return

        # Create execution summary as root data
        tree.root.data = self.execution

        # Sort root tasks chronologically by earliest trace
        def get_task_start_time(task_id: str) -> float:
            """Get the earliest start time for a task's traces."""
            task = self.execution.tasks.get(task_id)
            if not task or not task.traces:
                return 0.0
            return min((tr.start_ts for tr in task.traces if tr.start_ts), default=0.0)

        sorted_root_task_ids = sorted(
            self.execution.root_task_ids,
            key=get_task_start_time
        )

        # Render root tasks
        for root_task_id in sorted_root_task_ids:
            task = self.execution.tasks.get(root_task_id)
            if task:
                node = tree.root.add(self._task_label(task), data=task)
                self._populate_children(node, task)

        tree.root.expand()

        # Select first task
        if tree.root.children:
            first = tree.root.children[0]
            first.expand()
            tree.focus()
            tree.select_node(first)
            # Don't render task views immediately - let the tree render first
            # Task views will be rendered when user actually selects the node
        else:
            tree.focus()
            tree.select_node(tree.root)

    def _populate_children(self, node, task: TaskViewModel) -> None:
        """Recursively populate children and agent groups."""
        if not self.execution:
            return

        # Add agent groups for this task (if any)
        agent_groups = self._extract_agent_groups(task)
        for agent_type, agent_metrics in agent_groups.items():
            agent_label = f"🔧 {agent_type} ({agent_metrics['tokens']} tokens)"
            # Create AgentGroupViewModel with filtered traces
            agent_group = AgentGroupViewModel(
                task=task,
                agent_type=agent_type,
                traces=agent_metrics['traces'],
                tokens=agent_metrics['tokens'],
                duration=agent_metrics['duration']
            )
            node.add_leaf(agent_label, data=agent_group)

        # Add subtasks (sorted chronologically by earliest trace)
        def get_subtask_start_time(subtask_id: str) -> float:
            """Get the earliest start time for a subtask's traces."""
            subtask = self.execution.tasks.get(subtask_id)
            if not subtask or not subtask.traces:
                return 0.0
            return min((tr.start_ts for tr in subtask.traces if tr.start_ts), default=0.0)

        sorted_subtask_ids = sorted(task.subtask_ids, key=get_subtask_start_time)

        for child_id in sorted_subtask_ids:
            child = self.execution.tasks.get(child_id)
            if child:
                child_node = node.add(self._task_label(child), data=child)
                self._populate_children(child_node, child)

    def _extract_agent_groups(self, task: TaskViewModel) -> Dict[str, Dict[str, Any]]:
        """Extract agent execution groups from task traces.

        Returns dict mapping agent_type to metrics (tokens, duration, span_count).

        Strategy to avoid double-counting:
        - Duration: Use ONLY root wrapper spans (avoids double-counting nested spans)
        - Tokens: Use ONLY LM/tool spans (wrapper spans have 0 tokens)
        - Traces: Include ALL spans for display purposes
        """
        agent_groups: Dict[str, Dict[str, Any]] = {}

        # Group traces by module (which contains agent_type)
        for trace in task.traces:
            agent_type = trace.module
            if not agent_type:
                continue

            # Filter to only known agent types
            if agent_type not in ['atomizer', 'planner', 'executor', 'aggregator', 'verifier']:
                continue

            if agent_type not in agent_groups:
                agent_groups[agent_type] = {
                    'tokens': 0,
                    'duration': 0.0,
                    'span_count': 0,
                    'traces': [],
                }

            # Determine if this is a wrapper span or nested span
            is_wrapper = self._is_wrapper_span_for_metrics(trace)

            # Duration: Only count wrapper spans to avoid double-counting
            if is_wrapper:
                agent_groups[agent_type]['duration'] += trace.duration

            # Tokens: Only count LM/tool spans (wrapper spans have 0 tokens anyway)
            if not is_wrapper:
                agent_groups[agent_type]['tokens'] += trace.tokens

            # Always include in span count and traces list
            agent_groups[agent_type]['span_count'] += 1
            agent_groups[agent_type]['traces'].append(trace)

        # Sort by agent execution order
        agent_order = {'atomizer': 0, 'planner': 1, 'executor': 2, 'aggregator': 3, 'verifier': 4}
        return dict(sorted(agent_groups.items(), key=lambda x: agent_order.get(x[0], 999)))

    def _task_label(self, task: TaskViewModel) -> str:
        """Generate tree label for task."""
        goal = task.goal or f"Task {task.task_id[:8]}"

        # Status icon
        status_icon = {
            "completed": "✅",
            "failed": "❌",
            "running": "⚡",
            "pending": "⏳",
        }.get(task.status, "❓")

        # Metrics
        metrics_parts = []
        if task.total_duration > 0:
            metrics_parts.append(f"{task.total_duration:.1f}s")
        if task.total_tokens > 0:
            metrics_parts.append(f"{task.total_tokens} tok")

        # Add trace count (number of agent executions)
        trace_count = len(task.traces)
        if trace_count > 0:
            metrics_parts.append(f"{trace_count} traces")

        metrics_str = f" ({', '.join(metrics_parts)})" if metrics_parts else ""

        # Module tag
        module_tag = f"[{task.module}] " if task.module else ""

        # Error indicator
        error_icon = " ⚠️" if task.error else ""

        return f"{status_icon} {module_tag}{goal[:60]}{metrics_str}{error_icon}"

    def get_selected_task(self) -> Optional[TaskViewModel]:
        tree = self.query_one("#task-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            data = tree.cursor_node.data
            if isinstance(data, TaskViewModel):
                return data
        return None

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data

        # Cancel any in-progress render BEFORE starting new one (prevent race condition)
        if self._active_render_task and not self._active_render_task.done():
            self._active_render_task.cancel()

        # Defer rendering to avoid blocking the UI thread
        # This allows the tree selection to complete before heavy rendering starts
        if isinstance(data, TaskViewModel):
            # Store task reference immediately to prevent race condition
            self._active_render_task = asyncio.create_task(self._render_task_views_async(data))
        elif isinstance(data, AgentGroupViewModel):
            self._active_render_task = asyncio.create_task(self._render_agent_group_views_async(data))
        elif isinstance(data, ExecutionViewModel):
            self._active_render_task = asyncio.create_task(self._render_execution_summary_async())

    async def _render_task_views_async(self, task: TaskViewModel) -> None:
        """Async version with chunked rendering for large datasets."""
        # Cancellation is now handled by caller (on_tree_node_selected, action_toggle_io)
        # Just execute the render operations
        try:
            self.selected_task = task

            # Render async components
            try:
                await self._render_spans_tab_async(task)
            except Exception as exc:
                logger.error(f"Spans tab render failed: {exc}", exc_info=True)

            # Render sync components with individual error handling
            try:
                self._render_task_info(task)
            except Exception as exc:
                logger.error(f"Task info render failed: {exc}", exc_info=True)

            try:
                self._render_lm_table(task)
            except Exception as exc:
                logger.error(f"LM table render failed: {exc}", exc_info=True)

            try:
                self._render_tool_calls_table(task)
            except Exception as exc:
                logger.error(f"Tool calls table render failed: {exc}", exc_info=True)

        except asyncio.CancelledError:
            # User toggled or clicked another task, clean up gracefully - DO NOT re-raise or app will exit
            self.log(f"Render task views cancelled (task switch or rapid toggle)")
            logger.info("Render task views cancelled")
            return  # Just return, don't crash the app
        except Exception as exc:
            self.log(f"Critical error rendering task views: {exc}")
            logger.error(f"CRITICAL render task views error: {exc}", exc_info=True)
            import traceback
            self.log(traceback.format_exc())
            # Don't re-raise - log and continue

    async def _render_agent_group_views_async(self, agent_group: AgentGroupViewModel) -> None:
        """Async version with chunked rendering for agent groups."""
        # Cancellation is now handled by caller (on_tree_node_selected)
        # Just execute the render operations
        try:
            self.selected_task = agent_group.task

            # Render async components
            try:
                await self._render_trace_tab_for_traces_async(
                    agent_group.traces,
                    title=f"{agent_group.agent_type.title()} Agent - {agent_group.task.goal[:50]}"
                )
            except Exception as exc:
                logger.error(f"Trace tab render failed: {exc}", exc_info=True)

            # Render sync components with individual error handling
            try:
                self._render_task_info(agent_group.task)
            except Exception as exc:
                logger.error(f"Task info render failed: {exc}", exc_info=True)

            try:
                self._render_lm_table_for_agent(agent_group)
            except Exception as exc:
                logger.error(f"LM table render failed: {exc}", exc_info=True)

            try:
                self._render_tool_calls_table_for_agent(agent_group)
            except Exception as exc:
                logger.error(f"Tool calls table render failed: {exc}", exc_info=True)

        except asyncio.CancelledError:
            # User switched to another view, clean up gracefully
            logger.info("Render agent group views cancelled")
            return  # Don't re-raise, just return
        except Exception as exc:
            logger.error(f"CRITICAL render agent group error: {exc}", exc_info=True)
            # Don't re-raise - log and continue

    async def _render_execution_summary_async(self) -> None:
        """Async version for rendering execution summary."""
        if not self.execution:
            return

        # Cancellation is now handled by caller (on_tree_node_selected)
        # Just execute the render operations
        try:
            self.selected_task = None

            # Render async components
            try:
                await self._render_trace_tab_by_tasks_async(title=f"Execution: {self.execution.root_goal}")
            except Exception as exc:
                logger.error(f"Trace tab by tasks render failed: {exc}", exc_info=True)

            # Render sync components with individual error handling
            try:
                self._render_execution_info()
            except Exception as exc:
                logger.error(f"Execution info render failed: {exc}", exc_info=True)

            try:
                self._render_lm_table_all()
            except Exception as exc:
                logger.error(f"LM table all render failed: {exc}", exc_info=True)

            try:
                self._render_tool_calls_table_all()
            except Exception as exc:
                logger.error(f"Tool calls table all render failed: {exc}", exc_info=True)

        except asyncio.CancelledError:
            # User switched to another view, clean up gracefully
            logger.info("Render execution summary cancelled")
            return  # Don't re-raise, just return
        except Exception as exc:
            logger.error(f"CRITICAL render execution summary error: {exc}", exc_info=True)
            # Don't re-raise - log and continue

    def action_open_span_modal(self) -> None:
        """Open span detail modal for currently selected TreeTable node."""
        table = self.query_one("#spans-table", TreeTable)
        node = table.get_selected_node()
        if node:
            span = node.data.get("span_obj")
            if span:
                self._show_span_detail(span)

    def _get_task_spans(self, task: TaskViewModel) -> List[TraceViewModel]:
        """Get traces for a task. Traces are already deduplicated by transformer."""
        return task.traces

    def _render_timeline_graph(self, spans: List[TraceViewModel]) -> None:
        """Render timeline bar graph for given spans.

        Limited to 50 bars for performance - too many bars are hard to read anyway.
        """
        from textual.css.query import NoMatches

        try:
            graph = self.query_one("#timeline-graph-content", Static)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("Timeline graph widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying timeline graph widget: {exc}", exc_info=True)
            raise

        if not spans:
            graph.update("[dim](no timeline data)[/dim]")
            return

        # Filter out wrapper spans - only show actual execution spans
        non_wrapper_spans = [sp for sp in spans if not self._is_wrapper_span(sp)]

        if not non_wrapper_spans:
            graph.update("[dim](no spans to display)[/dim]")
            return

        # PERFORMANCE: Limit to 50 bars max (too many bars are unreadable anyway)
        MAX_BARS = 50
        was_truncated = len(non_wrapper_spans) > MAX_BARS

        # Sort spans by start time and take top 50 by duration if truncating
        sorted_spans = sorted(non_wrapper_spans, key=lambda sp: self._get_span_start(sp) or 0)
        if was_truncated:
            # Keep the longest-running spans for better visualization
            sorted_spans = sorted(sorted_spans, key=lambda sp: sp.duration, reverse=True)[:MAX_BARS]
            # Re-sort by start time for timeline display
            sorted_spans = sorted(sorted_spans, key=lambda sp: self._get_span_start(sp) or 0)

        max_duration = max((span.duration for span in sorted_spans), default=1.0)
        lines: List[str] = []

        # Build timeline graph
        has_start = any(self._get_span_start(span) is not None for span in sorted_spans)
        label_width = 30  # Increased from 22 to prevent truncation
        graph_width = 60  # Increased from 48 for better visualization

        if has_start:
            starts = [self._get_span_start(span) for span in sorted_spans if self._get_span_start(span) is not None]
            earliest = min(starts) if starts else 0.0
            ends: List[float] = []
            for span in sorted_spans:
                start = self._get_span_start(span) or earliest
                duration = span.duration
                ends.append(start + duration)
            max_end = max(ends) if ends else earliest + max_duration
            total_span = max(max_end - earliest, max_duration)
            if total_span <= 0:
                total_span = max_duration or 1.0
            total_label = f"{total_span:.2f}s"
            spacer = max(0, graph_width - len(total_label) - 2)
            lines.append(" " * (label_width + 1) + f"[dim]0s{' ' * spacer}{total_label}[/dim]")
            missing_start = False
            for span in sorted_spans:
                # Use simple name for timeline (no icons, just module prefix)
                simple_prefix = f"[{span.module}] " if span.module else ""
                name = f"{simple_prefix}{self._escape_text(span.name)}"[:label_width - 1]
                start = self._get_span_start(span)
                if start is None:
                    missing_start = True
                offset_cols = 0
                if start is not None:
                    offset_cols = int(((start - earliest) / total_span) * graph_width)
                offset_cols = max(0, min(offset_cols, graph_width - 1))
                duration = span.duration
                width_cols = max(1, int((duration / total_span) * graph_width)) if duration else 1
                if offset_cols + width_cols > graph_width:
                    width_cols = max(1, graph_width - offset_cols)
                bar = (" " * offset_cols + "█" * width_cols).ljust(graph_width)
                lines.append(f"{name:<{label_width}} {bar} {duration:.2f}s")
            if missing_start:
                lines.append("[dim]* spans without start time are aligned to 0s.[/dim]")
        else:
            for span in sorted_spans:
                duration = span.duration
                width_cols = max(1, int((duration / max_duration) * graph_width)) if duration else 1
                bar = ("█" * width_cols).ljust(graph_width)
                name = self._span_label(span, width=label_width - 1)
                lines.append(f"{name:<{label_width}} {bar} {duration:.2f}s")
            lines.append("[dim]Start timestamps unavailable; bars scaled by duration only.[/dim]")

        # Add note if truncated
        if was_truncated:
            lines.append(f"[dim]Showing top {MAX_BARS} longest spans (out of {len(non_wrapper_spans)} total)[/dim]")

        graph.update("\n".join(lines) if lines else "[dim](no timeline data)[/dim]")

    def _escape_text(self, value: Any) -> str:
        if value is None:
            return ""
        return escape(str(value))

    def _stringify_raw(self, value: Any, max_width: int = 80) -> str:
        """Convert value to string with text wrapping for long lines."""
        if isinstance(value, (dict, list)):
            try:
                # Create JSON with indentation
                text = json.dumps(value, indent=2, ensure_ascii=False)
                # Wrap long lines to max_width (simple line-by-line wrapping)
                lines = text.splitlines()
                wrapped_lines = []
                for line in lines:
                    if len(line) <= max_width:
                        wrapped_lines.append(line)
                    else:
                        # For lines that are too long, break them intelligently
                        # Preserve leading whitespace
                        leading_spaces = len(line) - len(line.lstrip())
                        indent_str = ' ' * leading_spaces

                        # Remove leading space for wrapping, add it back after
                        line_content = line.lstrip()

                        # Simple chunking - break at max_width
                        while line_content:
                            if len(line_content) <= max_width - leading_spaces:
                                wrapped_lines.append(indent_str + line_content)
                                break
                            else:
                                # Find a good break point (space, comma, etc.)
                                break_at = max_width - leading_spaces
                                for char in [', ', ' ', ',', ':', '=']:
                                    last_break = line_content[:break_at].rfind(char)
                                    if last_break > break_at // 2:  # Don't break too early
                                        break_at = last_break + len(char)
                                        break

                                wrapped_lines.append(indent_str + line_content[:break_at])
                                line_content = line_content[break_at:].lstrip()
                                # Increase indent for continuation lines
                                indent_str = ' ' * (leading_spaces + 2)

                text = '\n'.join(wrapped_lines)
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

    def _is_wrapper_span_for_metrics(self, trace: TraceViewModel) -> bool:
        """
        Determine if a trace is a root wrapper span for metrics calculation.

        This is used to prevent double-counting in agent group metrics.
        Wrapper spans contain nested LM/tool spans, so we count:
        - Wrapper duration (total execution time)
        - Nested span tokens (actual LM usage)

        Strategy:
        - Wrapper spans have agent names and typically 0 tokens
        - LM spans have agent names but non-zero tokens
        - Alternatively, check parent_trace_id (wrapper = None, nested = has parent)

        Returns:
            True if this is a wrapper span, False if it's a nested LM/tool span
        """
        name = trace.name.lower() if trace.name else ""
        wrapper_names = {"atomizer", "planner", "executor", "aggregator", "verifier"}

        if name not in wrapper_names:
            return False

        # Wrapper spans have agent names and 0 tokens
        # LM spans have agent names but non-zero tokens
        return trace.tokens == 0

    def _is_wrapper_name(self, span: TraceViewModel) -> bool:
        """Check if span is a wrapper that should be hidden in the span table.

        Note: We WANT to show agent wrappers (atomizer, planner, executor, aggregator, verifier)
        Only hide old-style generic wrappers like "agent executor".
        """
        name = span.name.lower() if span.name else ""
        # Don't hide agent-type wrappers - they should be visible!
        agent_types = {"atomizer", "planner", "executor", "aggregator", "verifier"}
        if name in agent_types:
            return False
        # Only hide generic wrapper names
        return name in {"agent executor", "agent_wrapper", "module_wrapper"}

    def _is_wrapper_span(self, span: TraceViewModel) -> bool:
        """Check if span is a wrapper that should be hidden in the TIMELINE.

        Different from _is_wrapper_name - this hides ALL wrapper spans including
        agent-type wrappers (atomizer, planner, etc.) from the timeline to avoid
        duplicating what's already shown in the table hierarchy.
        """
        # Check if this span has the is_wrapper flag from ExecutionDataService
        # (not yet implemented in transformer, but will be)

        # For now, identify wrappers by their names
        name = span.name.lower() if span.name else ""
        wrapper_names = {"atomizer", "planner", "executor", "aggregator", "verifier",
                        "agent executor", "agent_wrapper", "module_wrapper"}
        return name in wrapper_names

    async def _build_tree_table_nodes_async(self, spans: List[TraceViewModel], progress_callback=None) -> List:
        """Convert flat span list to TreeTableNode tree for TreeTable widget (ASYNC with progress).

        Args:
            spans: List of TraceViewModel objects
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            List of root TreeTableNode objects (with children recursively added)
        """
        from roma_dspy.tui.widgets.tree_table import TreeTableNode

        if not spans:
            return []

        total_spans = len(spans)
        processed = 0

        # Two-pass approach to handle out-of-order spans
        # Pass 1: Build id -> span mapping
        by_id: Dict[str, TraceViewModel] = {}
        for idx, span in enumerate(spans):
            by_id[span.trace_id] = span
            # Yield every 20 iterations to keep UI responsive
            if idx % 20 == 0:
                if progress_callback:
                    progress_callback(idx, total_spans, "Indexing spans...")
                await asyncio.sleep(0)

        # Pass 2: Build parent-child relationships
        children: Dict[str, List[TraceViewModel]] = defaultdict(list)
        roots: List[TraceViewModel] = []

        for idx, span in enumerate(spans):
            # Check if this span has a valid parent in the span list
            if span.parent_trace_id and span.parent_trace_id in by_id:
                children[span.parent_trace_id].append(span)
            else:
                roots.append(span)

            # Yield every 20 iterations to keep UI responsive
            if idx % 20 == 0:
                if progress_callback:
                    progress_callback(idx, total_spans, "Building hierarchy...")
                await asyncio.sleep(0)

        # Sort children and roots by start time
        def sort_key(span: TraceViewModel) -> Tuple[float, str]:
            return (self._get_span_start(span) or 0.0, span.name)

        for child_list in children.values():
            child_list.sort(key=sort_key)
        roots.sort(key=sort_key)

        # Build TreeTableNode tree recursively with yielding
        async def build_node_async(span: TraceViewModel, depth: int = 0):
            """Build a TreeTableNode with all its children (async)."""
            nonlocal processed

            # Format span data for columns
            start_display = "-"
            if span.start_time:
                try:
                    dt = datetime.fromisoformat(span.start_time.replace('Z', '+00:00'))
                    start_display = dt.strftime("%H:%M:%S.%f")[:-3]
                except:
                    start_display = span.start_time or "-"

            duration_display = f"{span.duration:.2f}s" if span.duration > 0 else "-"
            model_display = span.model if span.model else "-"
            tool_count = len(span.tool_calls) if span.tool_calls else 0
            tools_display = str(tool_count) if tool_count > 0 else "-"

            # Create node
            node = TreeTableNode(
                id=span.trace_id,
                label=self._span_label(span, width=40),
                data={
                    "Start Time": start_display,
                    "Duration": duration_display,
                    "Model": model_display,
                    "Tools": tools_display,
                    "span_obj": span,  # Store full span for event handlers
                },
            )

            # Add children recursively
            for child_span in children.get(span.trace_id, []):
                child_node = await build_node_async(child_span, depth + 1)
                node.add_child(child_node)

                processed += 1
                # Yield every 20 nodes to keep UI responsive
                if processed % 20 == 0:
                    if progress_callback:
                        progress_callback(processed, total_spans, "Building tree nodes...")
                    await asyncio.sleep(0)

            return node

        # Build root nodes (filter out wrappers if needed)
        root_nodes = []
        for root_span in roots:
            # Check if this is a wrapper that should be hidden
            has_children = bool(children.get(root_span.trace_id))
            is_wrapper = has_children and self._is_wrapper_name(root_span)

            if is_wrapper:
                # Skip wrapper, add its children as roots instead
                for child_span in children.get(root_span.trace_id, []):
                    root_nodes.append(await build_node_async(child_span))
            else:
                root_nodes.append(await build_node_async(root_span))

        return root_nodes

    def _add_tree_nodes_to_table(self, table, nodes: List) -> None:
        """Add TreeTableNode tree to TreeTable widget.

        This is a helper that recursively adds a tree structure to the TreeTable.

        Args:
            table: TreeTable widget
            nodes: List of root TreeTableNode objects
        """
        from roma_dspy.tui.widgets.tree_table import TreeTableNode

        for node in nodes:
            # Add root with all data
            root = table.add_root(node.label, node.data, node_id=node.id)

            # Recursively add children
            def add_children(parent_table_node, tree_node):
                for child in tree_node.children:
                    child_table_node = parent_table_node.add_child(TreeTableNode(
                        id=child.id,
                        label=child.label,
                        data=child.data,
                    ))
                    # Recursively add grandchildren
                    add_children(child_table_node, child)

            add_children(root, node)

        # Rebuild visible rows to ensure display is updated
        table.rebuild_visible_rows()

    async def _build_span_tree_internal(self, spans: List[TraceViewModel], hide_wrappers: bool = False) -> List[Tuple[TraceViewModel, int]]:
        """Build hierarchical span tree with depth information (async with yielding).

        Args:
            spans: List of spans to build tree from
            hide_wrappers: If True, hide wrapper spans from the output (used when showing module headers)
        """
        if not spans:
            return []

        by_id: Dict[str, TraceViewModel] = {}
        for idx, span in enumerate(spans):
            by_id[span.trace_id] = span
            # Yield every 20 iterations to keep UI responsive
            if idx % 20 == 0:
                await asyncio.sleep(0)

        children: Dict[str, List[TraceViewModel]] = defaultdict(list)
        roots: List[TraceViewModel] = []

        def sort_key(span: TraceViewModel) -> Tuple[float, str]:
            return (self._get_span_start(span) or 0.0, span.name)

        for idx, span in enumerate(spans):
            parent = span.parent_trace_id
            if parent and parent in by_id:
                children[parent].append(span)
            else:
                roots.append(span)
            # Yield every 20 iterations
            if idx % 20 == 0:
                await asyncio.sleep(0)

        # Sort child lists (yield after each batch)
        for idx, child_list in enumerate(children.values()):
            child_list.sort(key=sort_key)
            if idx % 20 == 0:
                await asyncio.sleep(0)

        roots.sort(key=sort_key)

        ordered: List[Tuple[TraceViewModel, int]] = []
        visit_count = 0

        async def visit_async(span: TraceViewModel, depth: int) -> None:
            nonlocal visit_count
            # Yield periodically to keep UI responsive
            visit_count += 1
            if visit_count % 20 == 0:
                await asyncio.sleep(0)

            has_children = bool(children.get(span.trace_id))
            # Check if this is a wrapper span that should be hidden
            is_wrapper = depth == 0 and has_children and (
                self._is_wrapper_name(span) if not hide_wrappers else self._is_wrapper_span(span)
            )
            next_depth = depth if is_wrapper else depth + 1

            if not is_wrapper:
                ordered.append((span, depth))

            for child in children.get(span.trace_id, []):
                await visit_async(child, next_depth)

        for root in roots:
            await visit_async(root, 0)

        return ordered

    async def _ordered_span_tree_async(self, spans: List[TraceViewModel], hide_wrappers: bool = False) -> List[Tuple[TraceViewModel, int]]:
        """Cached async wrapper for tree building.

        Args:
            spans: List of spans to build tree from
            hide_wrappers: If True, hide wrapper spans from the output
        """
        # Generate efficient cache key: hash of frozenset for order-independence
        span_ids = frozenset(s.trace_id for s in spans)
        cache_key = f"{hash(span_ids)}:{hide_wrappers}"

        # Check cache
        cached = self._span_tree_cache.get(cache_key)
        if cached is not None:
            return cached

        # Build tree
        result = await self._build_span_tree_internal(spans, hide_wrappers)

        # Store in cache
        self._span_tree_cache[cache_key] = result

        return result

    def _span_label(self, span: TraceViewModel, width: int = 36, show_task: bool = False) -> str:
        """Generate label for a span/trace.

        Note: No module prefix needed since TreeTable shows module in hierarchy.
        """
        label = self._escape_text(span.name)

        if show_task and span.task_id:
            # For execution summary view, show which task this trace belongs to
            if self.execution and span.task_id in self.execution.tasks:
                task = self.execution.tasks[span.task_id]
                goal = self._short_snippet(task.goal, width=width)
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

    def _format_span_detail(self, span: TraceViewModel) -> str:
        """Format detailed information for a span/trace (respects show_io toggle)."""
        lines: List[str] = []
        lines.append(f"[bold]Span:[/bold] {self._escape_text(span.name)}")

        # Show task info if in execution summary view
        if self.selected_task is None and self.execution and span.task_id in self.execution.tasks:
            task = self.execution.tasks[span.task_id]
            lines.append(f"[bold]Task:[/bold] {self._short_snippet(task.goal, width=160)}")
            lines.append(f"[bold]Task ID:[/bold] {self._escape_text(span.task_id)}")

        if span.start_time:
            lines.append(f"[bold]Start:[/bold] {self._escape_text(span.start_time)}")
        if span.duration > 0:
            lines.append(f"[bold]Duration:[/bold] {span.duration:.4f}s")
        if span.tokens > 0:
            lines.append(f"[bold]Tokens:[/bold] {span.tokens}")
        if span.model:
            lines.append(f"[bold]Model:[/bold] {self._escape_text(span.model)}")
        if span.source:
            lines.append(f"[bold]Source:[/bold] {span.source.value}")

        if span.tool_calls:
            lines.append("")
            lines.append("[bold]Tool Calls:[/bold]")
            for call in span.tool_calls:
                formatted = self._format_tool_call(call)
                if formatted:
                    lines.append(f"[dim]{formatted}[/dim]")

        # Show Input/Output only if toggle is ON
        if self.show_io:
            if span.inputs:
                lines.append("")
                lines.append("[bold]Input:[/bold]")
                lines.append(self._stringify_block(span.inputs))
            if span.outputs:
                lines.append("")
                lines.append("[bold]Output:[/bold]")
                lines.append(self._stringify_block(span.outputs))

        return "\n".join(lines) if lines else "[dim]No details for this span.[/dim]"

    def _update_trace_detail(self, span: Optional[TraceViewModel]) -> None:
        """No-op: Span details are now shown only in modal (via ENTER key)."""
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

    def _get_span_start(self, span: TraceViewModel) -> Optional[float]:
        """Get span start timestamp."""
        if span.start_ts is not None:
            return span.start_ts
        return self._parse_timestamp(span.start_time)

    def _get_span_duration(self, span: TraceViewModel) -> float:
        """Get span duration in seconds."""
        return span.duration

    def _render_timeline(self, task: TaskViewModel) -> None:
        """Legacy method - timeline is now merged into Spans tab."""
        pass  # Timeline graph is now rendered by _render_spans_tab_async()

    def _render_task_info(self, task: TaskViewModel) -> None:
        """Render task info tab for a selected task."""
        try:
            info = self.query_one("#task-info", Static)
        except Exception as exc:
            self.log(f"Failed to query #task-info widget: {exc}")
            return

        # Log current toggle state for debugging
        self.log(f"Rendering task info with show_io={self.show_io}")

        # Collect ROMA modules from task and traces
        roma_modules = set()
        if task.module:
            roma_modules.add(task.module)

        # Get modules from traces (prefer trace.module over span names)
        for span in task.traces:
            if span.module:  # Prefer explicit module from roma.module attribute
                roma_modules.add(span.module)

        # Build info lines
        id_line = f"[bold]Task ID:[/bold] {self._escape_text(task.task_id)}"
        parent_text = task.parent_task_id or "ROOT"
        parent_render = self._escape_text(parent_text)

        # Use inferred ROMA modules if task.module not set
        if task.module:
            module_render = self._escape_text(task.module)
        elif roma_modules:
            module_render = self._escape_text(', '.join(sorted(roma_modules)))
        else:
            module_render = "-"

        goal_render = self._escape_text(task.goal or "(unknown)")

        lines = [
            id_line,
            f"[bold]Parent:[/bold] {parent_render}",
            f"[bold]Goal:[/bold] {goal_render}",
            "",
            f"[bold]Module:[/bold] {module_render}",
            f"[bold]ROMA Modules:[/bold] {self._escape_text(', '.join(sorted(roma_modules))) if roma_modules else '-'}",
            f"[bold]Task Type:[/bold] {self._escape_text(task.task_type) if task.task_type else '-'}",
            f"[bold]Node Type:[/bold] {self._escape_text(task.node_type) if task.node_type else '-'}",
            f"[bold]Status:[/bold] {self._escape_text(task.status) if task.status else 'unknown'}",
            f"[bold]Depth:[/bold] {task.depth}",
            "",
            f"[bold]Duration:[/bold] {task.total_duration:.2f}s",
            f"[bold]Tokens:[/bold] {task.total_tokens}",
            f"[bold]Traces:[/bold] {len(task.traces)} spans",
            "",
        ]

        # Add agent execution breakdown
        agent_groups = self._extract_agent_groups(task)
        if agent_groups:
            lines.append("[bold]Agent Executions:[/bold]")
            for agent_type, metrics in agent_groups.items():
                lines.append(f"  🔧 {agent_type}:")
                lines.append(f"     Tokens: {metrics['tokens']}")
                lines.append(f"     Duration: {metrics['duration']:.2f}s")
                lines.append(f"     Spans: {metrics['span_count']}")
            lines.append("")

        lines.append(f"[bold]Data Sources:[/bold] {self._escape_text(self._data_sources_summary())}")

        if task.error:
            lines.insert(9, f"[bold red]Error:[/bold red] {self._escape_text(task.error)}")  # Insert after status

        # Show detailed result and tool calls only if toggle is ON
        if self.show_io:
            self.log(f"show_io=ON: Adding result and tool calls to task info")
            if task.result:
                lines.append("")
                lines.append("[bold]Result:[/bold]")
                lines.append(self._short_snippet(task.result, width=200))
                self.log(f"Added result section (length: {len(task.result)})")

            # Collect tool calls from traces
            tool_summaries: List[str] = []
            for span in task.traces:
                for call in span.tool_calls:
                    tool_summaries.append(self._summarize_tool_call(call, width=80))

            if tool_summaries:
                lines.append("")
                lines.append(f"[bold]Tool Calls ({len(tool_summaries)} total):[/bold]")
                # Show first 10 instead of 5
                for summary in tool_summaries[:10]:
                    lines.append(f"• {summary}")
                if len(tool_summaries) > 10:
                    lines.append(f"[dim]… and {len(tool_summaries) - 10} more[/dim]")
                self.log(f"Added {len(tool_summaries)} tool calls")
        else:
            self.log(f"show_io=OFF: Skipping result and tool calls")

        if self.execution and self.execution.warnings:
            lines.append(f"[dim]Warnings: {self._escape_text('; '.join(self.execution.warnings))}[/dim]")

        # Add toggle hint showing current state
        lines.append("")
        lines.append(
            f"[dim]Press 't' to toggle detailed span I/O (currently {'ON' if self.show_io else 'OFF'}). "
            "Highlight spans in the Spans tab to inspect them, or press Enter for a pop-out view.[/dim]"
        )

        try:
            content = "\n".join(lines)
            info.update(content)
        except Exception as exc:
            self.log(f"Task info update error: {exc}")
            import traceback
            self.log(traceback.format_exc())
            try:
                # Fallback: strip markup and try plain text
                plain = re.sub(r"\[(\/?)((?:bold|dim|red))\]", "", content)
                info.update(Text(plain))
            except Exception as exc2:
                self.log(f"Task info fallback update failed: {exc2}")
                info.update("[red]Error rendering task info[/red]")

    def _render_lm_table(self, task: TaskViewModel) -> None:
        """Render LM calls table for a selected task."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#lm-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("LM table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying LM table widget: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._lm_table_row_map.clear()  # Clear row mapping

        # Show traces for this task only - FILTER to only LM calls
        traces = task.traces
        # Filter to only LM call spans (those with tokens or model, typically named "LM.__call__")
        lm_traces = [t for t in traces if (t.tokens > 0 or t.model) and ("lm" in (t.name or "").lower() or "call" in (t.name or "").lower())]

        if not lm_traces:
            table.add_row("(none)", "", "", "", "")
            return

        # Display each LM call trace
        for trace in lm_traces:
            # Build preview from inputs/outputs
            preview = ""
            if trace.inputs:
                preview = self._short_snippet(trace.inputs, width=80)
            if self.show_io and trace.outputs:
                preview = self._short_snippet(trace.outputs, width=80)

            # Calculate latency from duration
            latency_ms = int(trace.duration * 1000) if trace.duration > 0 else 0

            row_key = table.add_row(
                trace.module or trace.name or "",
                trace.model or "",
                str(trace.tokens) if trace.tokens > 0 else "",
                str(latency_ms) if latency_ms > 0 else "",
                preview,
            )
            # Map row key to trace object
            self._lm_table_row_map[row_key] = trace

    def _render_lm_table_for_agent(self, agent_group: AgentGroupViewModel) -> None:
        """Render LM calls table filtered to a specific agent group."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#lm-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("LM table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying LM table widget for agent: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._lm_table_row_map.clear()  # Clear row mapping

        if not agent_group.traces:
            table.add_row("(none)", "", "", "", "")
            return

        # Filter to only LM call spans
        lm_traces = [t for t in agent_group.traces if (t.tokens > 0 or t.model) and ("lm" in (t.name or "").lower() or "call" in (t.name or "").lower())]

        if not lm_traces:
            table.add_row("(none)", "", "", "", "")
            return

        # Display LM call traces for this agent only
        for trace in lm_traces:
            preview = ""
            if trace.inputs:
                preview = self._short_snippet(trace.inputs, width=80)
            if self.show_io and trace.outputs:
                preview = self._short_snippet(trace.outputs, width=80)

            latency_ms = int(trace.duration * 1000) if trace.duration > 0 else 0

            row_key = table.add_row(
                trace.module or trace.name or "",
                trace.model or "",
                str(trace.tokens) if trace.tokens > 0 else "",
                str(latency_ms) if latency_ms > 0 else "",
                preview,
            )
            # Map row key to trace object
            self._lm_table_row_map[row_key] = trace

    def _render_summary_tab(self) -> None:
        """Render run summary tab."""
        info = self.query_one("#summary-info", Static)

        if not self.execution:
            info.update("[dim]No execution data available.[/dim]")
            return

        # Count total traces across all tasks
        total_traces = sum(len(task.traces) for task in self.execution.tasks.values())

        lines = [
            f"[bold]Execution:[/bold] {self._escape_text(self.execution_id)}",
            f"[bold]Root Goal:[/bold] {self._escape_text(self.execution.root_goal or '-')}",
            "",
            f"[bold]Total Tasks:[/bold] {len(self.execution.tasks)}",
            f"[bold]Total Traces:[/bold] {total_traces}",
            f"[bold]Total LM Calls:[/bold] {self.execution.metrics.total_calls}",
            f"[bold]Total Duration:[/bold] {self.execution.metrics.total_duration:.2f}s",
            f"[bold]Total Tokens:[/bold] {self.execution.metrics.total_tokens}",
            f"[bold]Total Cost:[/bold] ${self.execution.metrics.total_cost:.4f}",
            "",
            f"[bold]Data Sources:[/bold] {self._escape_text(self._data_sources_summary())}",
        ]

        # Module breakdown if available
        if self.execution.metrics.by_module:
            lines.append("")
            lines.append("[bold]By Module:[/bold]")
            for module_name, module_metrics in sorted(self.execution.metrics.by_module.items()):
                calls = module_metrics.get("calls", 0)
                tokens = module_metrics.get("tokens", 0)
                cost = module_metrics.get("cost", 0.0)
                lines.append(f"  • {module_name}: {calls} calls, {tokens} tokens, ${cost:.4f}")

        # Warnings
        if self.execution.warnings:
            lines.append("")
            lines.append("[bold]Warnings:[/bold]")
            for warning in self.execution.warnings:
                lines.append(f"  • {self._escape_text(warning)}")

        content = "\n".join(lines)
        try:
            info.update(content)
        except Exception as exc:
            self.log(f"Summary markup error: {exc}")
            plain = re.sub(r"\[(\/?)((?:bold|dim))\]", "", content)
            info.update(Text(plain))

    def _render_execution_info(self) -> None:
        """Render task info tab for execution summary."""
        info = self.query_one("#task-info", Static)

        if not self.execution:
            info.update("[dim]No execution data available.[/dim]")
            return

        # Collect all modules from all tasks
        modules = set()
        for task in self.execution.tasks.values():
            if task.module:
                modules.add(task.module)
            for span in task.traces:
                if span.name:
                    modules.add(span.name.split('.')[0])

        lines = [
            f"[bold]Execution ID:[/bold] {self._escape_text(self.execution_id)}",
            f"[bold]Root Goal:[/bold] {self._escape_text(self.execution.root_goal or '(unknown)')}",
            f"[bold]Status:[/bold] {self._escape_text(self.execution.status or 'unknown')}",
            f"[bold]Data Sources:[/bold] {self._escape_text(self._data_sources_summary())}",
            "",
            f"[bold]Total Tasks:[/bold] {len(self.execution.tasks)}",
            f"[bold]Root Tasks:[/bold] {len(self.execution.root_task_ids)}",
            f"[bold]Modules:[/bold] {self._escape_text(', '.join(sorted(modules))) if modules else '-'}",
            "",
            f"[bold]Total Duration:[/bold] {sum(task.total_duration for task in self.execution.tasks.values()):.2f}s",
            f"[bold]Total Tokens:[/bold] {sum(task.total_tokens for task in self.execution.tasks.values())}",
            f"[bold]Total Cost:[/bold] ${sum(task.total_cost for task in self.execution.tasks.values()):.4f}",
        ]

        lines.append("")
        lines.append(
            f"[dim]Press 't' to toggle detailed span I/O (currently {'ON' if self.show_io else 'OFF'}). "
            "Highlight spans in the Trace tab to inspect them below the table, or press Enter for a pop-out view.[/dim]"
        )

        if self.execution.warnings:
            lines.append(f"[dim]Warnings: {self._escape_text('; '.join(self.execution.warnings))}[/dim]")

        content = "\n".join(lines)
        try:
            info.update(content)
        except Exception as exc:
            self.log(f"Execution info markup error: {exc}")
            plain = re.sub(r"\[(\/?)((?:bold|dim))\]", "", content)
            info.update(Text(plain))

    def _render_lm_table_all(self) -> None:
        """Render LM calls table for execution summary (all traces)."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#lm-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("LM table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying LM table widget for all: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._lm_table_row_map.clear()

        if not self.execution:
            table.add_row("(none)", "", "", "", "")
            return

        # Collect all traces from all tasks and filter to only LM calls
        all_traces: List[TraceViewModel] = []
        for task in self.execution.tasks.values():
            all_traces.extend(task.traces)

        # Filter to only LM call spans
        lm_traces = [t for t in all_traces if (t.tokens > 0 or t.model) and ("lm" in (t.name or "").lower() or "call" in (t.name or "").lower())]

        if not lm_traces:
            table.add_row("(none)", "", "", "", "")
            return

        # Display each LM call trace
        for trace in lm_traces:
            # Build preview from inputs/outputs
            preview = ""
            if trace.inputs:
                preview = self._short_snippet(trace.inputs, width=80)
            if self.show_io and trace.outputs:
                preview = self._short_snippet(trace.outputs, width=80)

            # Calculate latency from duration
            latency_ms = int(trace.duration * 1000) if trace.duration > 0 else 0

            row_key = table.add_row(
                trace.module or trace.name or "",
                trace.model or "",
                str(trace.tokens) if trace.tokens > 0 else "",
                str(latency_ms) if latency_ms > 0 else "",
                preview,
            )
            # Map row key to trace object for click event handling
            self._lm_table_row_map[row_key] = trace

    def _render_tool_calls_table(self, task: TaskViewModel) -> None:
        """Render tool calls table for a selected task."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#tool-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("Tool table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying tool table widget: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._tool_table_row_map.clear()  # Clear row mapping

        # Extract all tool calls from task traces
        tool_calls = []
        for trace in task.traces:
            if trace.tool_calls:
                # DEBUG: Log the structure of tool_calls
                logger.debug(f"Tool calls for trace {trace.name}: {trace.tool_calls}")
                for call in trace.tool_calls:
                    logger.debug(f"Individual tool call: {call}")
                    # Add trace context to tool call
                    tool_calls.append({
                        "call": call,
                        "trace": trace,
                        "module": trace.module or trace.name,
                    })

        if not tool_calls:
            table.add_row("(none)", "", "", "", "")
            return

        # Display each tool call
        for item in tool_calls:
            call = item["call"]
            trace = item["trace"]

            # Extract tool name and toolkit
            tool_name = self._extract_tool_name(call)
            toolkit = self._extract_toolkit_name(call)

            # Calculate duration if available
            duration_ms = int(trace.duration * 1000) if trace.duration > 0 else 0

            # Determine status (success/error)
            status = "✓" if self._tool_call_successful(call) else "✗"

            # Build preview from arguments/output based on toggle
            preview = ""
            if self.show_io:
                # Show output if toggle is ON
                output = self._extract_tool_output(call)

                # If no output in call, try trace outputs as fallback
                if output is None and trace and trace.outputs:
                    output = trace.outputs

                if output is not None:
                    preview = self._short_snippet(output, width=80)
                else:
                    # Fallback to arguments if still no output
                    args = self._extract_tool_arguments(call)
                    if args is not None:
                        preview = f"[dim]{self._short_snippet(args, width=80)}[/dim]"
            else:
                # Show arguments if toggle is OFF (default)
                args = self._extract_tool_arguments(call)
                if args is not None:
                    preview = self._short_snippet(args, width=80)

            row_key = table.add_row(
                tool_name,
                toolkit,
                str(duration_ms) if duration_ms > 0 else "",
                status,
                preview,
            )
            # Map row key to tool call dict
            self._tool_table_row_map[row_key] = item

    def _render_tool_calls_table_for_agent(self, agent_group: AgentGroupViewModel) -> None:
        """Render tool calls table filtered to a specific agent group."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#tool-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("Tool table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying tool table widget for agent: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._tool_table_row_map.clear()

        if not agent_group.traces:
            table.add_row("(none)", "", "", "", "")
            return

        # Extract tool calls from agent group traces
        tool_calls = []
        for trace in agent_group.traces:
            if trace.tool_calls:
                for call in trace.tool_calls:
                    tool_calls.append({
                        "call": call,
                        "trace": trace,
                        "module": trace.module or trace.name,
                    })

        if not tool_calls:
            table.add_row("(none)", "", "", "", "")
            return

        # Display each tool call
        for item in tool_calls:
            call = item["call"]
            trace = item["trace"]

            tool_name = self._extract_tool_name(call)
            toolkit = self._extract_toolkit_name(call)
            duration_ms = int(trace.duration * 1000) if trace.duration > 0 else 0
            status = "✓" if self._tool_call_successful(call) else "✗"

            preview = ""
            if self.show_io:
                # Show output if toggle is ON
                output = self._extract_tool_output(call)

                # Fallback: If no output in call, try trace outputs
                if output is None and trace and trace.outputs:
                    output = trace.outputs

                if output is not None:
                    preview = self._short_snippet(output, width=80)
                else:
                    # Fallback to arguments if still no output
                    args = self._extract_tool_arguments(call)
                    if args is not None:
                        preview = f"[dim]{self._short_snippet(args, width=80)}[/dim]"
            else:
                # Show arguments if toggle is OFF (default)
                args = self._extract_tool_arguments(call)
                if args is not None:
                    preview = self._short_snippet(args, width=80)

            row_key = table.add_row(
                tool_name,
                toolkit,
                str(duration_ms) if duration_ms > 0 else "",
                status,
                preview,
            )
            self._tool_table_row_map[row_key] = item

    def _render_tool_calls_table_all(self) -> None:
        """Render tool calls table for execution summary (all tool calls)."""
        from textual.css.query import NoMatches

        try:
            table = self.query_one("#tool-table", DataTable)
        except NoMatches:
            # Widget doesn't exist - user is on a different tab - skip rendering
            logger.debug("Tool table widget not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying tool table widget for all: {exc}", exc_info=True)
            raise

        # Clear rows only (default columns=False keeps column headers)
        table.clear()
        self._tool_table_row_map.clear()

        if not self.execution:
            table.add_row("(none)", "", "", "", "")
            return

        # Collect all tool calls from all tasks
        tool_calls = []
        for task in self.execution.tasks.values():
            for trace in task.traces:
                if trace.tool_calls:
                    for call in trace.tool_calls:
                        tool_calls.append({
                            "call": call,
                            "trace": trace,
                            "module": trace.module or trace.name,
                        })

        if not tool_calls:
            table.add_row("(none)", "", "", "", "")
            return

        # Display each tool call
        for item in tool_calls:
            call = item["call"]
            trace = item["trace"]

            tool_name = self._extract_tool_name(call)
            toolkit = self._extract_toolkit_name(call)
            duration_ms = int(trace.duration * 1000) if trace.duration > 0 else 0
            status = "✓" if self._tool_call_successful(call) else "✗"

            preview = ""
            if self.show_io:
                # Show output if toggle is ON
                output = self._extract_tool_output(call)

                # Fallback: If no output in call, try trace outputs
                if output is None and trace and trace.outputs:
                    output = trace.outputs

                if output is not None:
                    preview = self._short_snippet(output, width=80)
                else:
                    # Fallback to arguments if still no output
                    args = self._extract_tool_arguments(call)
                    if args is not None:
                        preview = f"[dim]{self._short_snippet(args, width=80)}[/dim]"
            else:
                # Show arguments if toggle is OFF (default)
                args = self._extract_tool_arguments(call)
                if args is not None:
                    preview = self._short_snippet(args, width=80)

            row_key = table.add_row(
                tool_name,
                toolkit,
                str(duration_ms) if duration_ms > 0 else "",
                status,
                preview,
            )
            self._tool_table_row_map[row_key] = item

    def _extract_tool_name(self, call: Dict[str, Any]) -> str:
        """Extract tool name from tool call dict."""
        if not isinstance(call, dict):
            return str(call)

        # Try function object first (OpenAI format)
        func = call.get("function")
        if isinstance(func, dict):
            func_name = func.get("name")
            if func_name:
                return func_name

        # Try various field names
        name = (call.get("tool") or call.get("tool_name") or
                call.get("name") or call.get("type") or
                call.get("id"))

        return name or "unknown"

    def _extract_toolkit_name(self, call: Dict[str, Any]) -> str:
        """Extract toolkit name from tool call dict."""
        if not isinstance(call, dict):
            return "-"

        toolkit = call.get("toolkit") or call.get("toolkit_class")
        return toolkit or "-"

    def _extract_tool_arguments(self, call: Dict[str, Any]) -> Any:
        """Extract arguments from tool call dict."""
        if not isinstance(call, dict):
            return None

        # Try function.arguments first (OpenAI format)
        func = call.get("function")
        if isinstance(func, dict):
            args = func.get("arguments")
            if args is not None:
                return args

        # Try direct arguments
        args = call.get("arguments") or call.get("args") or call.get("input") or call.get("parameters")
        if args is not None:
            return args

        # Fallback: return whole call dict minus known metadata fields
        excluded_keys = {"tool", "tool_name", "name", "type", "id", "function", "output", "result", "return", "error", "status", "toolkit", "toolkit_class"}
        filtered = {k: v for k, v in call.items() if k not in excluded_keys}
        return filtered if filtered else None

    def _extract_tool_output(self, call: Dict[str, Any]) -> Any:
        """Extract output/result from tool call dict."""
        if not isinstance(call, dict):
            return None

        # Try various output field names in the call itself
        output = call.get("output") or call.get("result") or call.get("return") or call.get("response")
        if output is not None:
            return output

        # Check function.output (OpenAI format)
        func = call.get("function")
        if isinstance(func, dict):
            func_output = func.get("output") or func.get("result")
            if func_output is not None:
                return func_output

        # Check for content field (some frameworks use this)
        content = call.get("content")
        if content is not None:
            return content

        return None

    def _tool_call_successful(self, call: Dict[str, Any]) -> bool:
        """Check if tool call was successful."""
        if not isinstance(call, dict):
            return True

        # Check for error field - if present, call failed
        if call.get("error"):
            return False

        # Check for explicit status field
        status = call.get("status")
        if status:
            status_str = str(status).lower()
            if status_str in ("failed", "error", "failure"):
                return False
            if status_str in ("success", "ok", "completed"):
                return True

        # If no error and no explicit failure status, assume success
        return True

    async def on_data_loaded(self, message: DataLoaded) -> None:
        if message.success:
            task = self.get_selected_task()
            if task:
                await self._render_task_views_async(task)
        else:
            summary = self.query_one("#summary-info", Static)
            summary.update(f"Failed to load data: {message.error}")

    def on_tree_table_node_selected(self, event: TreeTable.NodeSelected) -> None:  # pragma: no cover - UI
        """Handle TreeTable node selection - show span detail modal."""
        span = event.node.data.get("span_obj")
        if span:
            self._show_span_detail(span)

    def on_tree_table_node_toggled(self, event: TreeTable.NodeToggled) -> None:  # pragma: no cover - UI
        """Handle TreeTable node expand/collapse."""
        # Optional: Log or track expand/collapse events
        self.log(f"Node {'expanded' if event.expanded else 'collapsed'}: {event.node.label}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:  # pragma: no cover - UI
        """Handle data table row selection - show detail modal for LM calls or tool calls."""
        # Check if this is an LM call
        trace = self._lm_table_row_map.get(event.row_key)
        if trace:
            self._show_span_detail(trace)
            return

        # Check if this is a tool call
        tool_item = self._tool_table_row_map.get(event.row_key)
        if tool_item:
            self._show_tool_call_detail(tool_item)

    def _show_span_detail(self, span: TraceViewModel) -> None:
        """Show span detail in a modal dialog with full I/O."""
        try:
            # Use the new parser system with toggle support
            parser = LMCallDetailParser()
            self.push_screen(GenericDetailModal(
                source_obj=span,
                parser=parser,
                show_io=self.show_io  # Respect current app toggle state
            ))
        except Exception as e:
            logger.error(f"Failed to show span detail: {e}", exc_info=True)
            self._show_error_fallback("Span", span, e)

    def _show_tool_call_detail(self, tool_item: Dict[str, Any]) -> None:
        """Show tool call detail in a modal dialog with full arguments/output."""
        try:
            # Use the new parser system with toggle support
            parser = ToolCallDetailParser()
            self.push_screen(GenericDetailModal(
                source_obj=tool_item,
                parser=parser,
                show_io=self.show_io  # Respect current app toggle state
            ))
        except Exception as e:
            logger.error(f"Failed to show tool call detail: {e}", exc_info=True)
            self._show_error_fallback("Tool Call", tool_item, e)

    def _show_error_fallback(self, object_type: str, obj: Any, error: Exception) -> None:
        """Show simple error modal when parsing fails."""
        import json

        # Try to show raw data
        try:
            if isinstance(obj, dict):
                raw_data = json.dumps(obj, indent=2, default=str)
            else:
                raw_data = str(obj)
        except Exception:
            raw_data = f"Could not serialize object: {type(obj)}"

        error_body = f"""[bold red]Error:[/bold red] {escape(str(error))}

[bold]Raw Data:[/bold]
{raw_data[:5000]}"""

        # Show notification
        self.notify(f"Failed to display {object_type}: {str(error)[:100]}", severity="error", timeout=5)

        # Show simple modal with error + raw data
        class SimpleErrorModal(ModalScreen):
            def compose(self) -> ComposeResult:
                with Container():
                    yield Label(f"Error Displaying {object_type}", classes="modal-title")
                    yield VerticalScroll(Static(error_body))

            def on_key(self, event: events.Key) -> None:
                if event.key in ("escape", "q", "enter"):
                    self.dismiss()

        self.push_screen(SimpleErrorModal())

    async def _render_spans_tab_async(self, task: TaskViewModel) -> None:
        """Async version with TreeTable widget and progress indicators."""
        from textual.css.query import NoMatches

        try:
            heading = self.query_one("#spans-heading", Static)
            summary = self.query_one("#spans-summary", Static)
            table = self.query_one("#spans-table", TreeTable)
        except NoMatches:
            # Widgets don't exist - user is on a different tab - skip rendering
            logger.debug("Spans tab widgets not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying spans tab widgets: {exc}", exc_info=True)
            raise

        heading_text = f"Task: {task.goal or task.task_id}"

        # Show loading indicator
        heading.update(f"[dim]Loading... {self._escape_text(heading_text)}[/dim]")

        # Clear table
        table.clear()

        traces = task.traces

        if not traces:
            heading.update(self._escape_text(heading_text))
            summary.update("")
            self._render_timeline_graph([])
            return

        # Progress callback
        def progress(current, total, message):
            pct = int((current / total) * 100) if total > 0 else 0
            heading.update(f"[dim]{message} {pct}% ({current}/{total})[/dim]")

        # Build TreeTableNode tree with progress (ASYNC)
        root_nodes = await self._build_tree_table_nodes_async(traces, progress_callback=progress)

        # Calculate metrics for summary
        # IMPORTANT: Only sum root traces to avoid double-counting nested spans
        root_traces = [t for t in traces if not t.parent_trace_id]
        total_duration = sum(t.duration for t in root_traces)
        summary_text = f"[bold]TOTAL[/bold]: {len(traces)} spans, {total_duration:.2f}s"
        summary.update(summary_text)

        # Add nodes to table
        self._add_tree_nodes_to_table(table, root_nodes)

        # Restore heading
        heading.update(self._escape_text(heading_text))

        # Render timeline graph
        self._render_timeline_graph(traces)

    async def _render_trace_tab_for_traces_async(self, traces: List[TraceViewModel], title: str = "All Traces") -> None:
        """Async version for agent groups with TreeTable and progress."""
        from textual.css.query import NoMatches

        try:
            heading = self.query_one("#spans-heading", Static)
            summary = self.query_one("#spans-summary", Static)
            table = self.query_one("#spans-table", TreeTable)
        except NoMatches:
            # Widgets don't exist - user is on a different tab - skip rendering
            logger.debug("Trace tab widgets not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying trace tab widgets for traces: {exc}", exc_info=True)
            raise

        # Show loading indicator
        heading.update(f"[dim]Loading... {self._escape_text(title)}[/dim]")

        # Clear table
        table.clear()

        if not traces:
            heading.update(self._escape_text(title))
            summary.update("")
            return

        # Progress callback
        def progress(current, total, message):
            pct = int((current / total) * 100) if total > 0 else 0
            heading.update(f"[dim]{message} {pct}% ({current}/{total})[/dim]")

        # Build TreeTableNode tree with progress (ASYNC)
        root_nodes = await self._build_tree_table_nodes_async(traces, progress_callback=progress)

        # Calculate metrics for summary
        # IMPORTANT: Only sum root traces to avoid double-counting nested spans
        root_traces = [t for t in traces if not t.parent_trace_id]
        total_duration = sum(t.duration for t in root_traces)
        summary_text = f"[bold]{title}[/bold]: {len(traces)} spans, {total_duration:.2f}s"
        summary.update(summary_text)

        # Add nodes to table
        self._add_tree_nodes_to_table(table, root_nodes)

        # Restore heading
        heading.update(self._escape_text(title))

        # Render timeline graph
        self._render_timeline_graph(traces)

    async def _render_trace_tab_by_tasks_async(self, title: str = "All Tasks") -> None:
        """Async version for execution summary with TreeTable (goals/modules/spans hierarchy)."""
        from textual.css.query import NoMatches
        from roma_dspy.tui.widgets.tree_table import TreeTableNode

        try:
            heading = self.query_one("#spans-heading", Static)
            summary = self.query_one("#spans-summary", Static)
            table = self.query_one("#spans-table", TreeTable)
        except NoMatches:
            # Widgets don't exist - user is on a different tab - skip rendering
            logger.debug("Trace tab widgets not found (on different tab), skipping render")
            return
        except Exception as exc:
            # Real error - log it and re-raise
            logger.error(f"CRITICAL: Error querying trace tab widgets by tasks: {exc}", exc_info=True)
            raise

        # Show loading indicator
        heading.update(f"[dim]Loading... {self._escape_text(title)}[/dim]")
        table.clear()

        if not self.execution or not self.execution.tasks:
            heading.update(self._escape_text(title))
            summary.update("")
            return

        # Collect all tasks
        def collect_all_tasks(task_id: str, collected: List[TaskViewModel]) -> None:
            task = self.execution.tasks.get(task_id)
            if task:
                collected.append(task)
                for subtask_id in task.subtask_ids:
                    collect_all_tasks(subtask_id, collected)

        all_tasks_list: List[TaskViewModel] = []
        for root_id in self.execution.root_task_ids:
            collect_all_tasks(root_id, all_tasks_list)

        # Group by goal
        tasks_by_goal: Dict[str, List[TaskViewModel]] = defaultdict(list)
        for task in all_tasks_list:
            goal_key = task.goal if task.goal else task.task_id
            tasks_by_goal[goal_key].append(task)

        # Calculate total metrics
        all_spans_total: List[TraceViewModel] = []
        for task in all_tasks_list:
            all_spans_total.extend(task.traces)

        # IMPORTANT: Only sum root traces to avoid double-counting
        root_spans_total = [s for s in all_spans_total if not s.parent_trace_id]
        total_duration = sum(s.duration for s in root_spans_total)

        # Update summary widget
        summary_text = f"[bold]TOTAL[/bold]: {len(all_tasks_list)} tasks, {len(all_spans_total)} spans, {total_duration:.2f}s"
        summary.update(summary_text)

        # Progress callback
        total_goals = len(tasks_by_goal)
        current_goal = 0

        # Sort goals chronologically by earliest trace start time
        def get_goal_start_time(goal_tasks: List[TaskViewModel]) -> float:
            """Get the earliest start time for all traces in a goal's tasks."""
            earliest_time = float('inf')
            for task in goal_tasks:
                for trace in task.traces:
                    if trace.start_ts and trace.start_ts < earliest_time:
                        earliest_time = trace.start_ts
            return earliest_time if earliest_time != float('inf') else 0.0

        sorted_goals = sorted(
            tasks_by_goal.items(),
            key=lambda x: get_goal_start_time(x[1])
        )

        # Build tree structure: Goals -> Modules -> Spans (with async yields)
        for goal, tasks in sorted_goals:
            current_goal += 1
            heading.update(f"[dim]Building tree... Goal {current_goal}/{total_goals}[/dim]")

            # Sort tasks within goal chronologically by earliest trace
            tasks_sorted = sorted(
                tasks,
                key=lambda t: min((tr.start_ts for tr in t.traces if tr.start_ts), default=0.0)
            )

            # Collect traces for this goal
            all_traces: List[TraceViewModel] = []
            for task in tasks_sorted:
                all_traces.extend(task.traces)

            # Calculate goal metrics
            # IMPORTANT: Only sum root traces to avoid double-counting
            root_traces_goal = [t for t in all_traces if not t.parent_trace_id]
            goal_duration = sum(t.duration for t in root_traces_goal)
            goal_tool_calls = sum(len(t.tool_calls) if t.tool_calls else 0 for t in all_traces)

            # Goal node (root) with summary data
            goal_label = f"📋 {goal[:60] if len(goal) > 60 else goal}"
            goal_node = table.add_root(
                goal_label,
                {
                    "Start Time": "-",
                    "Duration": f"{goal_duration:.1f}s ({len(tasks)}t/{len(all_traces)}s)",
                    "Model": "-",
                    "Tools": str(goal_tool_calls) if goal_tool_calls > 0 else "-",
                },
                node_id=f"goal-{goal[:30]}"
            )

            # Group by module
            spans_by_module: Dict[str, List[TraceViewModel]] = defaultdict(list)
            for span in all_traces:
                module_name = span.module or "Other"
                spans_by_module[module_name].append(span)

            # Add modules as children of goal
            for module_name in sorted(spans_by_module.keys()):
                module_spans = spans_by_module[module_name]

                # Calculate module metrics
                # IMPORTANT: Only sum root traces to avoid double-counting
                root_module_spans = [s for s in module_spans if not s.parent_trace_id]
                module_duration = sum(s.duration for s in root_module_spans)
                module_tool_calls = sum(len(s.tool_calls) if s.tool_calls else 0 for s in module_spans)

                # Module node (child of goal) with summary data
                module_label = f"🔧 {module_name}"
                module_node = goal_node.add_child(TreeTableNode(
                    id=f"module-{module_name}",
                    label=module_label,
                    data={
                        "Start Time": "-",
                        "Duration": f"{module_duration:.1f}s ({len(module_spans)}s)",
                        "Model": "-",
                        "Tools": str(module_tool_calls) if module_tool_calls > 0 else "-",
                    },
                ))

                # Progress for this module
                def module_progress(current, total, message):
                    heading.update(f"[dim]Building spans for {module_name}... {current}/{total}[/dim]")

                # Build span tree for this module (ASYNC)
                span_nodes = await self._build_tree_table_nodes_async(module_spans, progress_callback=module_progress)

                # Add spans as children of module
                for span_node in span_nodes:
                    # Add span root node
                    span_root = module_node.add_child(TreeTableNode(
                        id=span_node.id,
                        label=span_node.label,
                        data=span_node.data,
                    ))

                    # Recursively add span children
                    def add_span_children(parent, tree_node):
                        for child in tree_node.children:
                            child_node = parent.add_child(TreeTableNode(
                                id=child.id,
                                label=child.label,
                                data=child.data,
                            ))
                            add_span_children(child_node, child)

                    add_span_children(span_root, span_node)

                # Yield after each module to keep UI responsive
                await asyncio.sleep(0)

        # Rebuild visible rows to update display
        table.rebuild_visible_rows()

        # Restore heading
        heading.update(self._escape_text(title))

        # Render timeline graph
        self._render_timeline_graph(all_spans_total)


def run_viz_app(
    execution_id: str,
    base_url: str = "http://localhost:8000",
    live: bool = False,
    poll_interval: float = 2.0,
) -> None:
    """
    Helper to launch the Textual TUI.

    Args:
        execution_id: Execution ID to visualize
        base_url: API server URL
        live: Enable live mode with automatic polling
        poll_interval: Polling interval in seconds (default: 2.0)
    """
    app = RomaVizApp(
        execution_id=execution_id,
        base_url=base_url,
        live=live,
        poll_interval=poll_interval,
    )
    app.run()
# Old SpanDetailModal removed - replaced with GenericDetailModal from detail_view.py