"""Generic detail view system for TUI.

This module provides a reusable, tree-based detail view system for displaying
complex nested data structures (spans, LM calls, tool calls, etc.) in a user-friendly way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Collapsible, Label, Static, Tree
from textual.widgets.tree import TreeNode


class DataType(Enum):
    """Type of data in a section."""

    NESTED = "nested"  # Dicts, lists - use tree view
    TEXT = "text"  # Plain text - use text area
    CODE = "code"  # Code snippets - use syntax highlighting
    MARKDOWN = "markdown"  # Rich text - use markdown renderer
    UNKNOWN = "unknown"  # Auto-detect


class ViewMode(Enum):
    """How to render the data."""

    AUTO = "auto"  # Auto-detect best view
    TREE = "tree"  # Interactive tree
    RAW = "raw"  # Raw text/JSON
    FORMATTED = "formatted"  # Pretty-printed
    CUSTOM = "custom"  # Custom renderer


@dataclass
class DetailSection:
    """A collapsible section in the detail view."""

    id: str  # "input", "output", "reasoning"
    title: str  # "Input", "Output", "Reasoning"
    icon: str  # "📥", "📤", "🧠"
    data: Any  # The actual data
    data_type: DataType  # Type of data
    collapsed: bool = False  # Initial collapsed state
    view_mode: ViewMode = ViewMode.AUTO  # How to display
    renderer_hint: Optional[str] = None  # "json", "python", "markdown"


@dataclass
class DetailViewData:
    """Normalized container for any detail view."""

    title: str  # "Span: ChainOfThought.forward"
    metadata: Dict[str, Any]  # Simple key-value pairs (always visible)
    sections: List[Optional[DetailSection]]  # Collapsible sections (None = skip)
    source_object: Any  # Original object for reference


class DetailViewParser(ABC):
    """Base parser - converts objects to DetailViewData."""

    @abstractmethod
    def parse(self, obj: Any, context: str = "unknown", show_io: bool = True) -> DetailViewData:
        """Parse any object into normalized detail view data.

        Args:
            obj: Source object to parse
            context: Context string for logging/debugging
            show_io: Whether to include I/O sections (default: True)
        """
        pass

    def _detect_type(self, data: Any) -> DataType:
        """Auto-detect data type."""
        if data is None:
            return DataType.TEXT

        if isinstance(data, (dict, list)):
            return DataType.NESTED

        if isinstance(data, str):
            # Check if it's JSON, code, markdown, etc.
            stripped = data.strip()

            # JSON-like
            if stripped.startswith(("{", "[")):
                return DataType.CODE

            # Multi-line text
            if "\n" in data and len(data) > 200:
                return DataType.TEXT

            # Short text
            return DataType.TEXT

        # Numbers, booleans, etc.
        return DataType.TEXT


class DataRenderer(ABC):
    """Base class for data renderers."""

    @abstractmethod
    def render(self, data: Any, **kwargs) -> Widget:
        """Render data as a Textual widget."""
        pass


class TreeRenderer(DataRenderer):
    """Renders nested data as interactive tree."""

    def render(self, data: Any, **kwargs) -> Widget:
        """Render data as tree widget."""
        tree = Tree("Data", id=kwargs.get("section_id", "tree"))
        tree.show_root = False
        self._build_tree(tree.root, data)
        return tree

    def _build_tree(self, parent: TreeNode, data: Any, key: str = "root") -> None:
        """Recursively build tree nodes."""
        if isinstance(data, dict):
            if not data:
                parent.add_leaf("[dim]{empty}[/dim]")
                return

            for k, v in data.items():
                if isinstance(v, (dict, list)) and v:
                    # Nested structure - create expandable node
                    preview = self._get_preview(v)
                    label = f"{self._format_key(k)} [dim]{preview}[/dim]"
                    child_node = parent.add(label, expand=False)
                    self._build_tree(child_node, v, k)
                elif self._is_long_value(v):
                    # Long string or multiline - make it expandable
                    preview = self._format_value(v)
                    label = f"{self._format_key(k)}: {preview}"
                    child_node = parent.add(label, expand=False)
                    # Add full text as child
                    self._add_full_text(child_node, v)
                else:
                    # Short leaf node - show inline
                    label = f"{self._format_key(k)}: {self._format_value(v)}"
                    parent.add_leaf(label)

        elif isinstance(data, list):
            if not data:
                parent.add_leaf("[dim][] empty[/dim]")
                return

            for i, item in enumerate(data):
                if isinstance(item, (dict, list)) and item:
                    preview = self._get_preview(item)
                    label = f"[{i}] [dim]{preview}[/dim]"
                    child_node = parent.add(label, expand=False)
                    self._build_tree(child_node, item, f"[{i}]")
                elif self._is_long_value(item):
                    # Long string or multiline - make it expandable
                    preview = self._format_value(item)
                    label = f"[{i}]: {preview}"
                    child_node = parent.add(label, expand=False)
                    # Add full text as child
                    self._add_full_text(child_node, item)
                else:
                    label = f"[{i}]: {self._format_value(item)}"
                    parent.add_leaf(label)
        else:
            # Scalar value
            parent.add_leaf(self._format_value(data))

    def _get_preview(self, data: Any) -> str:
        """Generate preview for collapsed nodes."""
        if isinstance(data, dict):
            return f"{{{len(data)} keys}}"
        elif isinstance(data, list):
            return f"[{len(data)} items]"
        else:
            return ""

    def _is_long_value(self, value: Any) -> bool:
        """Check if value should be expandable (long text or multiline)."""
        if isinstance(value, str):
            return len(value) > 100 or "\n" in value
        return False

    def _add_full_text(self, parent: TreeNode, text: str) -> None:
        """Add full text content as child nodes (chunked for readability)."""
        if "\n" in text:
            # Multi-line: add each line as a child
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                if line:  # Skip empty lines
                    escaped_line = escape(line)
                    parent.add_leaf(f"[dim]L{i}:[/dim] [green]{escaped_line}[/green]")
                else:
                    parent.add_leaf(f"[dim]L{i}:[/dim] [dim](empty line)[/dim]")
        else:
            # Long single line: chunk into 200-char segments for readability
            chunk_size = 200
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                escaped_chunk = escape(chunk)
                chunk_label = f"[dim][{i}:{i+len(chunk)}][/dim] [green]{escaped_chunk}[/green]"
                parent.add_leaf(chunk_label)

    def _format_key(self, key: Any) -> str:
        """Format dict key."""
        # Convert to string and escape (keys can be int, tuple, etc.)
        return f"[bold cyan]{escape(str(key))}[/bold cyan]"

    def _format_value(self, value: Any) -> str:
        """Format value with syntax highlighting."""
        if isinstance(value, str):
            if len(value) > 100:
                return f'[green]"{escape(value[:100])}..."[/green]'
            elif "\n" in value:
                lines = value.split("\n")
                return f'[green]"{escape(lines[0])}..." ({len(lines)} lines)[/green]'
            else:
                return f'[green]"{escape(value)}"[/green]'
        elif isinstance(value, bool):
            return f"[blue]{value}[/blue]"
        elif isinstance(value, (int, float)):
            return f"[yellow]{value}[/yellow]"
        elif value is None:
            return "[dim]null[/dim]"
        else:
            return escape(str(value))


class RawRenderer(DataRenderer):
    """Renders data as formatted JSON/text."""

    def render(self, data: Any, **kwargs) -> Widget:
        """Render as formatted JSON or string."""
        import json

        if isinstance(data, (dict, list)):
            try:
                formatted = json.dumps(data, indent=2)
                # Escape JSON output to prevent markup interpretation
                return Static(escape(formatted), id=kwargs.get("section_id", "raw"))
            except Exception:
                # Escape fallback string representation
                return Static(escape(str(data)), id=kwargs.get("section_id", "raw"))
        else:
            # Escape string representation
            text = str(data) if data is not None else ""
            return Static(escape(text), id=kwargs.get("section_id", "raw"))


class TextRenderer(DataRenderer):
    """Renders plain text with wrapping."""

    def render(self, data: Any, **kwargs) -> Widget:
        """Render as plain text."""
        text = str(data) if data is not None else ""
        # Escape text to prevent markup interpretation
        return Static(escape(text), id=kwargs.get("section_id", "text"))


class CodeRenderer(DataRenderer):
    """Renders code with syntax highlighting."""

    def render(self, data: Any, language: str = "json", **kwargs) -> Widget:
        """Render with syntax highlighting."""
        from rich.syntax import Syntax

        code = str(data) if data is not None else ""

        try:
            syntax = Syntax(code, language, theme="monokai", line_numbers=False)
            return Static(syntax, id=kwargs.get("section_id", "code"))
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            return Static(code, id=kwargs.get("section_id", "code"))


# =============================================================================
# PARSERS - Convert domain objects to DetailViewData
# =============================================================================


class SpanDetailParser(DetailViewParser):
    """Parser for TraceViewModel objects (spans/LM calls)."""

    def parse(self, obj: Any, context: str = "span", show_io: bool = True) -> DetailViewData:
        """Parse a TraceViewModel into detail view data."""
        from roma_dspy.tui.models import TraceViewModel

        if not isinstance(obj, TraceViewModel):
            raise TypeError(f"Expected TraceViewModel, got {type(obj)}")

        trace = obj

        # Build title
        module_name = trace.module or trace.name
        title = f"Span: {escape(module_name)}"

        # Build metadata (always visible at top)
        metadata = {}
        if trace.duration > 0:
            metadata["Duration"] = f"{trace.duration:.3f}s"
        if trace.tokens > 0:
            metadata["Tokens"] = str(trace.tokens)
        if trace.cost > 0:
            metadata["Cost"] = f"${trace.cost:.6f}"
        if trace.model:
            metadata["Model"] = escape(trace.model)
        if trace.temperature is not None:
            metadata["Temperature"] = str(trace.temperature)
        if trace.start_time:
            metadata["Start Time"] = escape(trace.start_time)
        if trace.trace_id:
            metadata["Trace ID"] = escape(trace.trace_id)

        # Add I/O summary to metadata (ALWAYS add - this ensures metadata is never completely empty)
        available_sections = []
        if trace.inputs:
            available_sections.append("Input")
        if trace.outputs:
            available_sections.append("Output")
        if trace.reasoning:
            available_sections.append("Reasoning")

        # ALWAYS set I/O metadata, even if empty dict
        if available_sections:
            sections_list = ", ".join(available_sections)
            if show_io:
                metadata["I/O"] = f"✓ Showing: {sections_list}"
            else:
                metadata["I/O"] = f"✗ Hidden: {sections_list}"
        else:
            metadata["I/O"] = "No data available"

        # Ensure metadata is never empty by adding a fallback
        if not metadata:
            metadata["Status"] = "Active"

        # Build sections
        sections = []

        # Only add I/O sections if show_io is True
        if show_io:
            # Section 1: Input
            if trace.inputs:
                sections.append(
                    DetailSection(
                        id="input",
                        title="Input",
                        icon="📥",
                        data=trace.inputs,
                        data_type=self._detect_type(trace.inputs),
                        collapsed=False,
                        view_mode=ViewMode.AUTO,
                    )
                )

            # Section 2: Output
            if trace.outputs:
                sections.append(
                    DetailSection(
                        id="output",
                        title="Output",
                        icon="📤",
                        data=trace.outputs,
                        data_type=self._detect_type(trace.outputs),
                        collapsed=False,
                        view_mode=ViewMode.AUTO,
                    )
                )

            # Section 3: Reasoning (if present)
            if trace.reasoning:
                sections.append(
                    DetailSection(
                        id="reasoning",
                        title="Reasoning",
                        icon="🧠",
                        data=trace.reasoning,
                        data_type=DataType.TEXT,
                        collapsed=True,
                        view_mode=ViewMode.AUTO,
                    )
                )

        # Section 4: Tool Calls (always visible - metadata, not I/O)
        if trace.tool_calls:
            sections.append(
                DetailSection(
                    id="tool_calls",
                    title="Tool Calls",
                    icon="🔧",
                    data=trace.tool_calls,
                    data_type=DataType.NESTED,
                    collapsed=True,
                    view_mode=ViewMode.TREE,
                )
            )

        return DetailViewData(
            title=title, metadata=metadata, sections=sections, source_object=trace
        )


class ToolCallDetailParser(DetailViewParser):
    """Parser for tool call dictionaries."""

    def parse(self, obj: Any, context: str = "tool", show_io: bool = True) -> DetailViewData:
        """Parse a tool call dict into detail view data.

        Args:
            obj: Tool item dict with keys: 'call', 'trace', 'module'
            context: Additional context string
            show_io: Whether to include I/O sections (default: True)
        """
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict, got {type(obj)}")

        call = obj.get("call", {})
        trace = obj.get("trace")
        module_name = obj.get("module", "unknown")

        # Extract tool name
        tool_name = self._extract_tool_name(call)

        # Build title
        title = f"Tool Call: {escape(tool_name)}"

        # Build metadata
        metadata = {}

        # Tool info
        toolkit = call.get("toolkit") or call.get("source") or "unknown"
        metadata["Tool"] = escape(tool_name)
        metadata["Toolkit"] = escape(toolkit)

        # Duration (if available)
        duration_ms = call.get("duration_ms") or call.get("duration")
        if duration_ms:
            metadata["Duration"] = f"{duration_ms}ms"

        # Status
        success = self._tool_call_successful(call)
        metadata["Status"] = "✓ Success" if success else "✗ Failed"

        # Module
        if module_name and module_name != "unknown":
            metadata["Module"] = escape(module_name)

        # Add I/O summary to metadata (ALWAYS add - this ensures metadata is never completely empty)
        # PRIMARY SOURCE: Use trace inputs/outputs (they have the actual data)
        # FALLBACK: Use call arguments/output (often just schema)
        args = None
        output = None

        if trace:
            # Try trace inputs first (this is where actual data is stored)
            if hasattr(trace, 'inputs') and trace.inputs:
                args = trace.inputs
            # Try trace outputs
            if hasattr(trace, 'outputs') and trace.outputs:
                output = trace.outputs
                # If output is a JSON string, parse it
                if isinstance(output, str):
                    try:
                        import json
                        output = json.loads(output)
                    except (json.JSONDecodeError, ValueError):
                        # If it's not valid JSON, keep as string
                        pass

        # Fallback to call if trace doesn't have the data
        if args is None:
            args = self._extract_tool_arguments(call)
        if output is None:
            output = self._extract_tool_output(call)

        available_sections = []
        if args is not None:
            available_sections.append("Arguments")
        if output is not None:
            available_sections.append("Output")

        # ALWAYS set I/O metadata
        if available_sections:
            sections_list = ", ".join(available_sections)
            if show_io:
                metadata["I/O"] = f"✓ Showing: {sections_list}"
            else:
                metadata["I/O"] = f"✗ Hidden: {sections_list}"
        else:
            metadata["I/O"] = "No data available"

        # Ensure metadata is never empty by adding a fallback
        if not metadata:
            metadata["Status"] = "Active"

        # Build sections
        sections = []

        # Only add I/O sections if show_io is True
        if show_io:
            # Section 1: Arguments (already extracted from trace or call above)
            if args is not None:
                sections.append(
                    DetailSection(
                        id="arguments",
                        title="Arguments",
                        icon="📝",
                        data=args,
                        data_type=self._detect_type(args),
                        collapsed=False,
                        view_mode=ViewMode.AUTO,
                    )
                )

            # Section 2: Output (already extracted from trace or call above)
            # For code execution tools, restructure output to highlight stdout
            if output is not None:
                output_to_display = output

                # Check if this is a code execution output with stdout
                if isinstance(output, dict) and "stdout" in output:
                    stdout = output.get("stdout", [])

                    # If stdout is a list, join it
                    if isinstance(stdout, list):
                        stdout_text = "".join(stdout)
                    else:
                        stdout_text = str(stdout)

                    # Create restructured output that shows stdout first
                    if stdout_text:
                        output_to_display = {
                            "stdout": stdout_text,
                            "success": output.get("success"),
                            "results": output.get("results"),
                            "stderr": output.get("stderr"),
                            "error": output.get("error"),
                            "sandbox_id": output.get("sandbox_id"),
                        }
                        # Remove None values to keep it clean
                        output_to_display = {k: v for k, v in output_to_display.items() if v is not None and v != [] and v != ""}

                sections.append(
                    DetailSection(
                        id="output",
                        title="Output",
                        icon="📤",
                        data=output_to_display,
                        data_type=self._detect_type(output_to_display),
                        collapsed=False,
                        view_mode=ViewMode.AUTO,
                    )
                )

        # Section 3: Error (always visible - important diagnostic info)
        error = call.get("error") or call.get("exception")
        if error:
            sections.append(
                DetailSection(
                    id="error",
                    title="Error",
                    icon="❌",
                    data=error,
                    data_type=DataType.TEXT,
                    collapsed=False,
                    view_mode=ViewMode.AUTO,
                )
            )

        return DetailViewData(
            title=title, metadata=metadata, sections=sections, source_object=obj
        )

    def _extract_tool_name(self, call: Dict[str, Any]) -> str:
        """Extract tool name from call dict."""
        # Try function object first (OpenAI format)
        func = call.get("function")
        if isinstance(func, dict):
            func_name = func.get("name")
            if func_name:
                return func_name

        # Try various field names
        name = (
            call.get("tool")
            or call.get("tool_name")
            or call.get("name")
            or call.get("type")
            or call.get("id")
        )
        return name or "unknown"

    def _extract_tool_arguments(self, call: Dict[str, Any]) -> Any:
        """Extract arguments from call dict."""
        # Try various argument field names
        args = (
            call.get("arguments")
            or call.get("args")
            or call.get("input")
            or call.get("params")
            or call.get("parameters")
        )

        # Check function.arguments (OpenAI format)
        if args is None:
            func = call.get("function")
            if isinstance(func, dict):
                args = func.get("arguments") or func.get("args")

        # Handle special case: code stored as dict key (MLflow artifact format)
        if isinstance(args, dict):
            # Check if this is just type metadata without actual content
            if set(args.keys()) == {"type"} or (set(args.keys()) == {"code", "type"} and isinstance(args.get("code"), dict) and not args.get("code")):
                # Just metadata like {"type": "string"} or {"code": {}, "type": "string"}
                # Not useful to display
                return None

            # For code execution tools, check if code is stored as a dict key
            if "code" in args:
                code_val = args["code"]
                if isinstance(code_val, dict):
                    if len(code_val) == 1:
                        # Code is stored as: {"code": {"<actual_code_string>": None}}
                        # Extract the dict key as the actual code
                        actual_code = list(code_val.keys())[0]
                        # Replace the dict with the actual code string
                        args = dict(args)  # Make a copy
                        args["code"] = actual_code
                        # Remove type field if it exists, as it's redundant
                        args.pop("type", None)
                    elif len(code_val) == 0:
                        # Empty dict, check if there's a 'type' sibling that has the code
                        if "type" in args and isinstance(args["type"], str):
                            # Sometimes the structure is: {"code": {}, "type": "actual_code_here"}
                            # But more likely, if code is empty, there's no code
                            pass
                elif isinstance(code_val, str):
                    # Code is already a string, good!
                    # Remove redundant type field
                    if "type" in args:
                        args = dict(args)
                        args.pop("type", None)

        return args

    def _extract_tool_output(self, call: Dict[str, Any]) -> Any:
        """Extract output from call dict."""
        # Try various output field names
        output = (
            call.get("output")
            or call.get("result")
            or call.get("return")
            or call.get("response")
        )

        if output is not None:
            # For code execution tools, prioritize stdout if available
            if isinstance(output, dict) and "stdout" in output:
                stdout = output.get("stdout", "")
                # If stdout is the main content, return a cleaned version
                if stdout and isinstance(stdout, str):
                    # Keep the full dict but ensure stdout is easily visible
                    return output
            return output

        # Check function.output (OpenAI format)
        func = call.get("function")
        if isinstance(func, dict):
            func_output = func.get("output") or func.get("result")
            if func_output is not None:
                return func_output

        # Check for content field
        content = call.get("content")
        if content is not None:
            return content

        return None

    def _tool_call_successful(self, call: Dict[str, Any]) -> bool:
        """Check if tool call was successful."""
        # Check for error field
        if call.get("error") or call.get("exception"):
            return False

        # Check for explicit status field
        status = call.get("status")
        if status:
            status_str = str(status).lower()
            if status_str in ("failed", "error", "failure"):
                return False
            if status_str in ("success", "ok", "completed"):
                return True

        # If no error and no explicit failure, assume success
        return True


class LMCallDetailParser(SpanDetailParser):
    """Parser for LM calls - just an alias for SpanDetailParser."""

    def parse(self, obj: Any, context: str = "lm_call", show_io: bool = True) -> DetailViewData:
        """Parse an LM call (which is just a TraceViewModel)."""
        # Just delegate to parent
        result = super().parse(obj, context, show_io=show_io)
        # Update title to say "LM Call" instead of "Span"
        result.title = result.title.replace("Span:", "LM Call:")
        return result


# =============================================================================
# WIDGETS - UI components for displaying detail views
# =============================================================================


class GenericDetailView(VerticalScroll):
    """Generic detail view widget that displays normalized DetailViewData.

    This widget handles rendering of:
    - Title
    - Metadata (key-value pairs, always visible)
    - Collapsible sections (each with appropriate renderer)
    """

    DEFAULT_CSS = """
    GenericDetailView {
        background: $panel;
        border: solid $primary;
        padding: 1;
    }

    GenericDetailView .detail-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    GenericDetailView .detail-metadata {
        background: $boost;
        padding: 1;
        margin-bottom: 1;
        border: round $primary-lighten-1;
        height: auto;
        min-height: 3;
    }

    GenericDetailView .metadata-row {
        margin-bottom: 0;
        height: auto;
    }

    GenericDetailView Collapsible {
        margin-bottom: 1;
        border: round $primary-darken-1;
    }

    GenericDetailView Tree {
        height: auto;
        scrollbar-size: 1 1;
    }

    GenericDetailView Static {
        height: auto;
    }
    """

    def __init__(
        self,
        data: DetailViewData,
        renderer_registry: Optional[Dict[DataType, DataRenderer]] = None,
        **kwargs,
    ):
        """Initialize the detail view.

        Args:
            data: The normalized detail view data
            renderer_registry: Optional custom renderers for data types
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.data = data

        # Set up default renderers
        if renderer_registry is None:
            self.renderers = {
                DataType.NESTED: TreeRenderer(),
                DataType.TEXT: TextRenderer(),
                DataType.CODE: CodeRenderer(),
                DataType.MARKDOWN: TextRenderer(),  # TODO: Add markdown renderer
                DataType.UNKNOWN: RawRenderer(),
            }
        else:
            self.renderers = renderer_registry

    def compose(self) -> ComposeResult:
        """Compose the detail view layout."""
        # Title
        yield Label(self.data.title, classes="detail-title")

        # Metadata section (always visible if not None)
        if self.data.metadata is not None and len(self.data.metadata) > 0:
            with Container(classes="detail-metadata"):
                for key, value in self.data.metadata.items():
                    yield Label(
                        f"[bold]{key}:[/bold] {value}", classes="metadata-row"
                    )
        elif self.data.metadata is not None:
            # Metadata dict exists but is empty - show placeholder
            with Container(classes="detail-metadata"):
                yield Label("[dim]No metadata available[/dim]", classes="metadata-row")

        # Collapsible sections
        for section in self.data.sections:
            if section is None:
                continue

            # Create collapsible section
            with Collapsible(
                title=f"{section.icon} {section.title}",
                collapsed=section.collapsed,
                id=f"section-{section.id}",
            ):
                # Render the data using appropriate renderer
                renderer = self._select_renderer(section)
                if renderer:
                    # Build render kwargs
                    render_kwargs = {"section_id": f"data-{section.id}"}

                    # Add language hint if renderer supports it
                    if isinstance(renderer, CodeRenderer) and section.renderer_hint:
                        render_kwargs["language"] = section.renderer_hint

                    widget = renderer.render(section.data, **render_kwargs)
                    yield widget

    def _select_renderer(self, section: DetailSection) -> Optional[DataRenderer]:
        """Select the appropriate renderer for a section."""
        # If view mode is custom, try to use a custom renderer
        if section.view_mode == ViewMode.CUSTOM:
            # TODO: Support custom renderers
            pass

        # Map view mode to renderer
        if section.view_mode == ViewMode.TREE:
            return self.renderers.get(DataType.NESTED)
        elif section.view_mode == ViewMode.RAW:
            return RawRenderer()
        elif section.view_mode == ViewMode.FORMATTED:
            return CodeRenderer()

        # Auto mode - use data type
        return self.renderers.get(section.data_type, RawRenderer())


class GenericDetailModal(ModalScreen):
    """Modal dialog that displays a GenericDetailView.

    Supports:
    - ESC / q to close
    - 't' to toggle I/O display
    - Automatic sizing
    """

    DEFAULT_CSS = """
    GenericDetailModal {
        align: center middle;
    }

    GenericDetailModal > Container {
        width: 90%;
        height: 90%;
        background: $panel;
        border: thick $primary;
    }

    GenericDetailModal .modal-title {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }

    GenericDetailModal .toggle-hint {
        dock: bottom;
        height: 1;
        background: $panel-darken-1;
        color: $text-muted;
        content-align: center middle;
        text-style: italic;
    }

    GenericDetailModal GenericDetailView {
        height: 1fr;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape,q", "dismiss", "Close", show=True),
    ]

    def __init__(
        self,
        source_obj: Any,
        parser: DetailViewParser,
        show_io: bool = True,
        renderer_registry: Optional[Dict[DataType, DataRenderer]] = None,
        **kwargs,
    ):
        """Initialize the modal.

        Args:
            source_obj: The original object to parse
            parser: Parser instance to use for rendering
            show_io: Initial I/O display state (default: True)
            renderer_registry: Optional custom renderers
            **kwargs: Additional screen arguments
        """
        super().__init__(**kwargs)
        self.source_obj = source_obj
        self.parser = parser
        self.show_io = show_io
        self.renderer_registry = renderer_registry
        self._current_data = parser.parse(source_obj, show_io=show_io)
        self._view_counter = 0  # Counter for unique IDs

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="modal-container"):
            yield Label(self._current_data.title, classes="modal-title")
            yield GenericDetailView(
                self._current_data,
                renderer_registry=self.renderer_registry,
                id=f"detail-view-{self._view_counter}",
            )
            # Footer with toggle hint
            hint_text = self._get_toggle_hint_text()
            yield Label(hint_text, classes="toggle-hint")

    def _get_toggle_hint_text(self) -> str:
        """Get the toggle hint text based on current state."""
        if self.show_io:
            return "I/O Display: ON • Press 't' to hide detailed I/O sections"
        else:
            return "I/O Display: OFF • Press 't' to show detailed I/O sections"

    def on_key(self, event: events.Key) -> None:
        """Handle key events - intercept 't' before it bubbles to app."""
        if event.key == "t":
            # Stop event from propagating to parent app
            event.stop()
            # Toggle I/O display
            self._toggle_io()

    def _toggle_io(self) -> None:
        """Toggle I/O display and refresh view."""
        # Store old state for rollback
        old_show_io = self.show_io
        old_counter = self._view_counter

        self.show_io = not self.show_io
        self._view_counter += 1  # Increment for unique ID

        try:
            # Re-parse with new state (can raise exceptions)
            self._current_data = self.parser.parse(self.source_obj, show_io=self.show_io)

            # Get container
            container = self.query_one("#modal-container", Container)

            # Remove all children from container (clears everything)
            container.remove_children()

            # Mount fresh content with unique IDs (no ID on label to avoid conflicts)
            container.mount(Label(self._current_data.title, classes="modal-title"))
            container.mount(
                GenericDetailView(
                    self._current_data,
                    renderer_registry=self.renderer_registry,
                    id=f"detail-view-{self._view_counter}",  # Unique ID each time
                )
            )
            # Mount footer hint with updated state
            hint_text = self._get_toggle_hint_text()
            container.mount(Label(hint_text, classes="toggle-hint"))

            # Only show success notification if we got here
            status = "ON" if self.show_io else "OFF"
            self.notify(f"I/O Display: {status}", severity="information", timeout=1)

        except Exception as e:
            # Rollback state on any error
            self.show_io = old_show_io
            self._view_counter = old_counter
            # Show error notification
            self.notify(f"Failed to toggle I/O: {str(e)[:100]}", severity="error", timeout=3)

    def action_dismiss(self) -> None:
        """Close the modal."""
        self.dismiss()
