"""TreeTable widget - A table with collapsible tree hierarchy.

Combines the best of Tree (hierarchy, expand/collapse) and DataTable (columns, sorting).
Renders tree guides and columns in a unified scrollable view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Static


@dataclass
class TreeTableNode:
    """A node in the tree table.

    Attributes:
        id: Unique identifier for this node
        label: Display text for the tree column
        data: Dictionary of column_name -> value for data columns
        children: List of child nodes
        parent: Reference to parent node (None for root)
        expanded: Whether children are visible
        depth: Nesting level (0 for root)
        is_last_sibling: Whether this is the last child of its parent
    """

    id: str
    label: str
    data: Dict[str, Any] = field(default_factory=dict)
    children: List[TreeTableNode] = field(default_factory=list)
    parent: Optional[TreeTableNode] = None
    expanded: bool = True
    depth: int = 0
    is_last_sibling: bool = False

    def add_child(self, child: TreeTableNode) -> TreeTableNode:
        """Add a child node and update its depth and parent.

        **IMPORTANT**: This method only modifies the TreeTableNode structure.
        If this node is part of a TreeTable widget, you must call
        `table.rebuild_visible_rows()` after adding children to update
        the display.

        Args:
            child: The node to add as a child

        Returns:
            The child node (for chaining). If the child is already in the
            children list, returns it without adding again.

        Example:
            # Adding children through the node
            root.add_child(child1)
            root.add_child(child2)
            table.rebuild_visible_rows()  # Required to update display!
        """
        # Prevent duplicate children
        if child in self.children:
            return child

        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        self._update_sibling_flags()
        return child

    def _update_sibling_flags(self) -> None:
        """Update is_last_sibling flags for all children."""
        for i, child in enumerate(self.children):
            child.is_last_sibling = (i == len(self.children) - 1)

    def toggle_expanded(self) -> None:
        """Toggle the expanded state."""
        if self.children:
            self.expanded = not self.expanded

    def get_visible_descendants(self) -> List[TreeTableNode]:
        """Get all visible descendant nodes (respecting expanded state)."""
        result = []
        for child in self.children:
            result.append(child)
            if child.expanded and child.children:
                result.extend(child.get_visible_descendants())
        return result


class TreeTable(ScrollView, can_focus=True):
    """A table widget with collapsible tree hierarchy in the first column.

    Features:
    - Tree-style hierarchy with expand/collapse
    - Multiple data columns aligned with tree rows
    - Keyboard navigation (arrows, space/enter to toggle)
    - Click to expand/collapse or select
    - Zebra striping for readability

    Usage:
        table = TreeTable(columns=["Duration", "Model", "Tools"])
        root = table.add_root("Root Item", {"Duration": "1.5s", "Model": "gpt-4"})
        child = root.add_child(TreeTableNode(
            id="child-1",
            label="Child Item",
            data={"Duration": "0.5s", "Model": "gpt-4"}
        ))
    """

    BINDINGS = [
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("left", "collapse", "Collapse"),
        ("right", "expand", "Expand"),
        ("space", "toggle", "Toggle"),
        ("enter", "select", "Select"),
    ]

    DEFAULT_CSS = """
    TreeTable {
        background: $surface;
        color: $text;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    TreeTable:focus {
        border: tall $accent;
    }

    TreeTable > .tree-table--header {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    """

    # Reactive attributes
    cursor_row: reactive[int] = reactive(0)
    show_header: reactive[bool] = reactive(True)
    zebra_stripes: reactive[bool] = reactive(True)

    # Column configuration
    TREE_COLUMN_WIDTH = 60  # Width of the tree/hierarchy column (increased for deep hierarchies)
    DATA_COLUMN_WIDTH = 30  # Width of each data column (increased for compound summary values)

    class NodeSelected(Message):
        """Posted when a node is selected."""

        def __init__(self, node: TreeTableNode) -> None:
            self.node = node
            super().__init__()

    class NodeToggled(Message):
        """Posted when a node is expanded or collapsed."""

        def __init__(self, node: TreeTableNode, expanded: bool) -> None:
            self.node = node
            self.expanded = expanded
            super().__init__()

    def __init__(
        self,
        columns: List[str],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        """Initialize TreeTable.

        Args:
            columns: List of column names for data columns (tree column is implicit)
            name: The name of the widget
            id: The ID of the widget in the DOM
            classes: The CSS classes for the widget
            disabled: Whether the widget is disabled
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.columns = columns
        self.roots: List[TreeTableNode] = []
        self._visible_rows: List[TreeTableNode] = []
        self._row_to_node: Dict[int, TreeTableNode] = {}
        self.cursor_row = 0
        self._next_node_id = 0  # Counter for auto-generated node IDs (never resets)

    def add_root(self, label: str, data: Dict[str, Any], node_id: Optional[str] = None) -> TreeTableNode:
        """Add a root-level node.

        Args:
            label: Display text for the tree column
            data: Dictionary of column_name -> value
            node_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The created TreeTableNode
        """
        if node_id is None:
            node_id = f"node-{self._next_node_id}"
            self._next_node_id += 1

        node = TreeTableNode(id=node_id, label=label, data=data, depth=0)
        self.roots.append(node)
        self._update_sibling_flags()
        self.rebuild_visible_rows()
        return node

    def clear(self) -> None:
        """Remove all nodes."""
        self.roots.clear()
        self._visible_rows.clear()
        self._row_to_node.clear()
        self.cursor_row = 0
        self.refresh()

    def _update_sibling_flags(self) -> None:
        """Update is_last_sibling flags for root nodes."""
        for i, root in enumerate(self.roots):
            root.is_last_sibling = (i == len(self.roots) - 1)

    def rebuild_visible_rows(self) -> None:
        """Rebuild the list of visible rows based on expand/collapse state."""
        self._visible_rows.clear()
        self._row_to_node.clear()

        for root in self.roots:
            self._visible_rows.append(root)
            if root.expanded:
                self._visible_rows.extend(root.get_visible_descendants())

        # Build row index mapping
        for idx, node in enumerate(self._visible_rows):
            self._row_to_node[idx] = node

        # Constrain cursor
        if self.cursor_row >= len(self._visible_rows):
            self.cursor_row = max(0, len(self._visible_rows) - 1)

        # Update virtual size for scrolling
        row_count = len(self._visible_rows) + (1 if self.show_header else 0)
        self.virtual_size = Size(self.size.width, row_count)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        """Calculate the width of the content."""
        return self.TREE_COLUMN_WIDTH + (len(self.columns) * self.DATA_COLUMN_WIDTH)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Calculate the height of the content."""
        row_count = len(self._visible_rows) + (1 if self.show_header else 0)
        return row_count

    def render_line(self, y: int) -> Strip:
        """Render a single line of the table.

        Args:
            y: Line number (0 = header if show_header, otherwise first data row)

        Returns:
            Strip containing the rendered line
        """
        scroll_y = self.scroll_offset.y
        line_y = y + scroll_y

        # Render header
        if self.show_header and line_y == 0:
            return self._render_header()

        # Calculate row index
        row_idx = line_y - (1 if self.show_header else 0)

        # Render data row
        if row_idx < 0 or row_idx >= len(self._visible_rows):
            return Strip.blank(self.size.width)

        node = self._visible_rows[row_idx]
        is_selected = (row_idx == self.cursor_row)
        is_even = (row_idx % 2 == 0)

        return self._render_row(node, is_selected, is_even)

    def _truncate_text(self, text: str, max_width: int) -> str:
        """Truncate text to fit within max_width, adding ellipsis if needed.

        Uses cell_len() for accurate display width accounting for Unicode.

        Args:
            text: Text to truncate
            max_width: Maximum width in display cells

        Returns:
            Truncated text with ellipsis if needed
        """
        text_width = cell_len(text)
        if text_width <= max_width:
            return text
        # Reserve space for ellipsis (1 cell)
        if max_width <= 1:
            return "…"

        # Binary search to find the right truncation point
        # since Unicode characters can have different widths
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid] + "…"
            if cell_len(candidate) <= max_width:
                left = mid
            else:
                right = mid - 1

        return text[:left] + "…" if left < len(text) else text

    def _render_header(self) -> Strip:
        """Render the header row."""
        segments = []

        # Tree column header - pad to exact width using cell_len()
        header_text = "Span"
        header_width = cell_len(header_text)
        padding = self.TREE_COLUMN_WIDTH - header_width
        tree_header_str = header_text + (" " * max(0, padding))
        tree_header = Text(tree_header_str, style="bold")
        segments.extend(tree_header.render(self.app.console))

        # Data column headers
        for col_name in self.columns:
            # Truncate column name if too long (reserve 1 cell for leading space)
            col_name_truncated = self._truncate_text(col_name, self.DATA_COLUMN_WIDTH - 1)
            # Manually pad to ensure exact width using cell_len()
            col_display_width = cell_len(col_name_truncated)
            padding_needed = self.DATA_COLUMN_WIDTH - 1 - col_display_width
            padded_col = " " + col_name_truncated + (" " * max(0, padding_needed))
            col_text = Text(padded_col, style="bold")
            segments.extend(col_text.render(self.app.console))

        # Fill remaining width (defensive: use max to prevent negative width)
        total_width = self.TREE_COLUMN_WIDTH + (len(self.columns) * self.DATA_COLUMN_WIDTH)
        fill_width = max(0, self.size.width - total_width)
        if fill_width > 0:
            segments.append(Segment(" " * fill_width))

        return Strip(segments, self.size.width)

    def _render_row(self, node: TreeTableNode, is_selected: bool, is_even: bool) -> Strip:
        """Render a single data row.

        Args:
            node: The node to render
            is_selected: Whether this row is selected (cursor on it)
            is_even: Whether this is an even row (for zebra striping)

        Returns:
            Strip containing the rendered row
        """
        segments = []

        # Determine style
        if is_selected:
            style = Style(bgcolor="blue", color="white", bold=True)
        elif self.zebra_stripes and is_even:
            style = Style(bgcolor="grey11")
        else:
            style = Style()

        # Render tree column with guides
        tree_text = self._get_tree_cell_text(node)
        # Note: No set_length() call - we manually pad in _get_tree_cell_text()
        tree_text.stylize(style)
        segments.extend(tree_text.render(self.app.console))

        # Render data columns
        for col_name in self.columns:
            value = node.data.get(col_name, "")
            value_str = str(value)
            # Truncate value if too long (reserve 1 cell for leading space)
            value_truncated = self._truncate_text(value_str, self.DATA_COLUMN_WIDTH - 1)
            # Manually pad to ensure exact width using cell_len()
            value_display_width = cell_len(value_truncated)
            padding_needed = self.DATA_COLUMN_WIDTH - 1 - value_display_width
            padded_value = " " + value_truncated + (" " * max(0, padding_needed))
            col_text = Text(padded_value)
            col_text.stylize(style)
            segments.extend(col_text.render(self.app.console))

        # Fill remaining width with style (defensive: use max to prevent negative width)
        total_width = self.TREE_COLUMN_WIDTH + (len(self.columns) * self.DATA_COLUMN_WIDTH)
        fill_width = max(0, self.size.width - total_width)
        if fill_width > 0:
            segments.append(Segment(" " * fill_width, style))

        return Strip(segments, self.size.width)

    def _get_tree_cell_text(self, node: TreeTableNode) -> Text:
        """Generate the tree column text with guides and icons.

        Args:
            node: The node to render

        Returns:
            Rich Text object with tree guides and label
        """
        # Build plain string first to calculate actual display width
        parts = []
        current_width = 0

        # Build tree guides (│, ├──, └──)
        ancestors = self._get_ancestors(node)
        for ancestor in ancestors[:-1]:  # Exclude the node itself
            if self._has_sibling_below(ancestor):
                guide = "│   "
                parts.append((guide, "dim"))
                current_width += cell_len(guide)
            else:
                guide = "    "
                parts.append((guide, "dim"))
                current_width += cell_len(guide)

        # Add branch connector
        if node.depth > 0:
            if node.is_last_sibling:
                branch = "└── "
            else:
                branch = "├── "
            parts.append((branch, "dim"))
            current_width += cell_len(branch)

        # Add expand/collapse icon
        if node.children:
            if node.expanded:
                icon = "⊟ "
                parts.append((icon, "bold cyan"))
            else:
                icon = "⊞ "
                parts.append((icon, "bold yellow"))
            current_width += cell_len(icon)
        else:
            icon = "  "
            parts.append((icon, None))
            current_width += cell_len(icon)

        # Calculate available width for label using actual display width
        available_width = self.TREE_COLUMN_WIDTH - current_width

        # Truncate label to fit available width
        label_truncated = self._truncate_text(node.label, available_width)
        # Pad label to exact width
        label_display_width = cell_len(label_truncated)
        padding_needed = available_width - label_display_width
        label_padded = label_truncated + (" " * max(0, padding_needed))
        parts.append((label_padded, None))

        # Build Text object
        text = Text()
        for content, style in parts:
            if style:
                text.append(content, style=style)
            else:
                text.append(content)

        return text

    def _get_ancestors(self, node: TreeTableNode) -> List[TreeTableNode]:
        """Get list of ancestors from root to node (inclusive).

        Optimized to O(n) by appending and reversing instead of repeated insert(0).
        """
        ancestors = []
        current = node
        while current is not None:
            ancestors.append(current)  # O(1) append instead of O(n) insert(0)
            current = current.parent
        return list(reversed(ancestors))  # O(n) reverse once at end

    def _has_sibling_below(self, node: TreeTableNode) -> bool:
        """Check if a node has siblings below it.

        Optimized to use try/except instead of 'in' check + index() for better performance.
        """
        if node.parent is None:
            # Root node - check in roots list
            try:
                idx = self.roots.index(node)
                return idx < len(self.roots) - 1
            except ValueError:
                # Node not in roots list (shouldn't happen, but be defensive)
                return False
        else:
            # Child node - check in parent's children
            try:
                idx = node.parent.children.index(node)
                return idx < len(node.parent.children) - 1
            except ValueError:
                # Node not in parent's children (shouldn't happen, but be defensive)
                return False

    def on_click(self, event: events.Click) -> None:
        """Handle click events."""
        # Calculate which row was clicked
        scroll_y = self.scroll_offset.y
        clicked_y = event.y + scroll_y

        # Account for header
        if self.show_header:
            if clicked_y == 0:
                return  # Clicked on header
            clicked_y -= 1

        # Check if valid row
        if clicked_y < 0 or clicked_y >= len(self._visible_rows):
            return

        node = self._visible_rows[clicked_y]

        # Check if clicked on expand/collapse icon area
        # Calculate icon position based on tree structure:
        # - Root (depth 0): just icon (2 chars: "⊟ " or "⊞ ")
        # - Nested (depth > 0): tree_guides (depth*4) + branch (4) + icon (2)
        if node.depth == 0:
            icon_x_end = 2
        else:
            icon_x_end = (node.depth * 4) + 6

        if event.x < icon_x_end and node.children:
            # Clicked on icon area - toggle
            self.action_toggle_at_row(clicked_y)
        else:
            # Clicked on row - select
            self.cursor_row = clicked_y
            self.post_message(self.NodeSelected(node))

    def action_cursor_up(self) -> None:
        """Move cursor up one row."""
        if self.cursor_row > 0:
            self.cursor_row -= 1
            self.scroll_to_cursor()

    def action_cursor_down(self) -> None:
        """Move cursor down one row."""
        if self.cursor_row < len(self._visible_rows) - 1:
            self.cursor_row += 1
            self.scroll_to_cursor()

    def action_expand(self) -> None:
        """Expand the current node."""
        if self.cursor_row < len(self._visible_rows):
            node = self._visible_rows[self.cursor_row]
            if node.children and not node.expanded:
                node.expanded = True
                self.rebuild_visible_rows()
                self.post_message(self.NodeToggled(node, True))
                self.refresh()

    def action_collapse(self) -> None:
        """Collapse the current node."""
        if self.cursor_row < len(self._visible_rows):
            node = self._visible_rows[self.cursor_row]
            if node.children and node.expanded:
                node.expanded = False
                self.rebuild_visible_rows()
                self.post_message(self.NodeToggled(node, False))
                self.refresh()

    def action_toggle(self) -> None:
        """Toggle expand/collapse of the current node."""
        if self.cursor_row < len(self._visible_rows):
            node = self._visible_rows[self.cursor_row]
            if node.children:
                node.toggle_expanded()
                self.rebuild_visible_rows()
                self.post_message(self.NodeToggled(node, node.expanded))
                self.refresh()

    def action_toggle_at_row(self, row: int) -> None:
        """Toggle expand/collapse at a specific row."""
        if row < len(self._visible_rows):
            node = self._visible_rows[row]
            if node.children:
                node.toggle_expanded()
                self.rebuild_visible_rows()
                self.post_message(self.NodeToggled(node, node.expanded))
                self.refresh()

    def action_select(self) -> None:
        """Select the current node and post message."""
        if self.cursor_row < len(self._visible_rows):
            node = self._visible_rows[self.cursor_row]
            self.post_message(self.NodeSelected(node))

    def scroll_to_cursor(self) -> None:
        """Scroll to make the cursor visible."""
        # Calculate the y position of the cursor row
        cursor_y = self.cursor_row + (1 if self.show_header else 0)

        # Scroll if cursor is out of view
        viewport_top = self.scroll_offset.y
        viewport_bottom = viewport_top + self.size.height - 1

        if cursor_y < viewport_top:
            self.scroll_to(y=cursor_y, animate=False)
        elif cursor_y > viewport_bottom:
            self.scroll_to(y=cursor_y - self.size.height + 1, animate=False)

    def watch_cursor_row(self, old_row: int, new_row: int) -> None:
        """React to cursor row changes."""
        self.refresh()

    def get_selected_node(self) -> Optional[TreeTableNode]:
        """Get the currently selected node."""
        if 0 <= self.cursor_row < len(self._visible_rows):
            return self._visible_rows[self.cursor_row]
        return None
