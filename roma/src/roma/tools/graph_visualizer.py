"""
Graph Visualization Tools for ROMA v2.0

Provides visual debugging capabilities for dynamic task graphs.
Supports multiple output formats including ASCII art, Graphviz, and Matplotlib.
"""

import asyncio
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

from ..domain.graph.dynamic_task_graph import DynamicTaskGraph
from ..domain.entities.task_node import TaskNode
from ..domain.value_objects.task_status import TaskStatus
from ..domain.value_objects.task_type import TaskType
from ..domain.value_objects.node_type import NodeType


class GraphVisualizer:
    """
    Comprehensive graph visualization for ROMA task graphs.
    
    Features:
    - ASCII art for console output
    - Graphviz DOT format for professional diagrams
    - Matplotlib for interactive visualization
    - Status-based coloring and styling
    - Export to various formats
    """
    
    # Color schemes for different statuses
    STATUS_COLORS = {
        TaskStatus.PENDING: "#E8E8E8",      # Light gray
        TaskStatus.READY: "#FFF2CC",        # Light yellow  
        TaskStatus.EXECUTING: "#D4E1F5",    # Light blue
        TaskStatus.COMPLETED: "#D5E8D4",    # Light green
        TaskStatus.FAILED: "#F8CECC",       # Light red
        TaskStatus.AGGREGATING: "#E1D5E7",  # Light purple
    }
    
    TASK_TYPE_SHAPES = {
        TaskType.THINK: "ellipse",
        TaskType.WRITE: "box", 
        TaskType.RETRIEVE: "diamond"
    }
    
    def __init__(self, graph: DynamicTaskGraph):
        """Initialize visualizer with task graph."""
        self.graph = graph
    
    def render_ascii(self, max_width: int = 80, show_details: bool = False) -> str:
        """
        Render graph as ASCII art tree.
        
        Args:
            max_width: Maximum width for text wrapping
            show_details: Include status and metadata
            
        Returns:
            ASCII art representation of the graph
        """
        if not self.graph.nodes:
            return "Empty graph"
        
        # Find root nodes (nodes with no parents)
        root_nodes = [
            node for node in self.graph.get_all_nodes() 
            if node.parent_id is None
        ]
        
        if not root_nodes:
            return "No root nodes found"
        
        lines = []
        for root in root_nodes:
            lines.extend(self._render_node_ascii(root, "", True, show_details))
        
        return "\n".join(lines)
    
    def _render_node_ascii(
        self, 
        node: TaskNode, 
        prefix: str, 
        is_last: bool, 
        show_details: bool
    ) -> List[str]:
        """Recursively render node and children as ASCII."""
        lines = []
        
        # Node symbol based on status
        status_symbol = {
            TaskStatus.PENDING: "⚪",
            TaskStatus.READY: "🟡", 
            TaskStatus.EXECUTING: "🔵",
            TaskStatus.COMPLETED: "🟢",
            TaskStatus.FAILED: "🔴",
            TaskStatus.AGGREGATING: "🟣"
        }.get(node.status, "⚪")
        
        # Node type symbol
        type_symbol = {
            TaskType.THINK: "🧠",
            TaskType.WRITE: "✏️", 
            TaskType.RETRIEVE: "📥"
        }.get(node.task_type, "")
        
        # Build node line
        connector = "└── " if is_last else "├── "
        node_info = f"{status_symbol} {type_symbol} {node.goal[:40]}..."
        
        if show_details:
            details = f" [{node.status.value}, {node.task_type.value}]"
            node_info += details
        
        lines.append(prefix + connector + node_info)
        
        # Get children
        children_ids = self.graph.get_children(node.task_id)
        children = [self.graph.get_node(child_id) for child_id in children_ids]
        children = [child for child in children if child is not None]
        
        # Render children
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(
                self._render_node_ascii(child, child_prefix, is_child_last, show_details)
            )
        
        return lines
    
    def render_graphviz(
        self, 
        output_format: str = "png",
        include_metadata: bool = True,
        cluster_by_type: bool = False
    ) -> str:
        """
        Render graph using Graphviz.
        
        Args:
            output_format: Output format (png, svg, pdf, dot)
            include_metadata: Include status and timing info
            cluster_by_type: Group nodes by task type
            
        Returns:
            DOT format string or rendered output path
            
        Raises:
            ImportError: If graphviz not available
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("Graphviz not available. Install with: pip install graphviz")
        
        dot = graphviz.Digraph(comment='ROMA Task Graph')
        dot.attr(rankdir='TB', size='12,8!')
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='8')
        
        # Add title
        title = f"ROMA Task Graph - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        dot.attr(label=title, fontsize='14', fontname='Arial Bold')
        
        if cluster_by_type:
            # Create subgraphs for each task type
            self._add_clustered_nodes(dot, include_metadata)
        else:
            # Add all nodes without clustering
            self._add_nodes_to_dot(dot, include_metadata)
        
        # Add edges
        for node in self.graph.get_all_nodes():
            children_ids = self.graph.get_children(node.task_id)
            for child_id in children_ids:
                dot.edge(node.task_id, child_id)
        
        # Add legend
        self._add_legend_to_dot(dot)
        
        if output_format == "dot":
            return dot.source
        else:
            return dot.render(format=output_format, cleanup=True)
    
    def _add_nodes_to_dot(self, dot: Any, include_metadata: bool) -> None:
        """Add nodes to graphviz dot object."""
        for node in self.graph.get_all_nodes():
            label = self._create_node_label(node, include_metadata)
            color = self.STATUS_COLORS.get(node.status, "#E8E8E8")
            shape = self.TASK_TYPE_SHAPES.get(node.task_type, "ellipse")
            
            dot.node(
                node.task_id,
                label=label,
                fillcolor=color,
                style='filled',
                shape=shape
            )
    
    def _add_clustered_nodes(self, dot: Any, include_metadata: bool) -> None:
        """Add nodes grouped by task type."""
        for task_type in TaskType:
            type_nodes = [n for n in self.graph.get_all_nodes() if n.task_type == task_type]
            if not type_nodes:
                continue
            
            with dot.subgraph(name=f'cluster_{task_type.value}') as cluster:
                cluster.attr(label=f'{task_type.value} Tasks', style='dashed')
                
                for node in type_nodes:
                    label = self._create_node_label(node, include_metadata)
                    color = self.STATUS_COLORS.get(node.status, "#E8E8E8")
                    shape = self.TASK_TYPE_SHAPES.get(node.task_type, "ellipse")
                    
                    cluster.node(
                        node.task_id,
                        label=label,
                        fillcolor=color,
                        style='filled',
                        shape=shape
                    )
    
    def _create_node_label(self, node: TaskNode, include_metadata: bool) -> str:
        """Create formatted label for node."""
        # Truncate goal for readability
        goal = node.goal[:30] + "..." if len(node.goal) > 30 else node.goal
        
        if not include_metadata:
            return goal
        
        # Add metadata
        lines = [goal, f"ID: {node.task_id[:8]}"]
        lines.append(f"Status: {node.status.value}")
        
        if node.started_at:
            elapsed = datetime.now() - node.started_at.replace(tzinfo=None)
            lines.append(f"Time: {elapsed.total_seconds():.1f}s")
        
        return "\\n".join(lines)
    
    def _add_legend_to_dot(self, dot: Any) -> None:
        """Add status legend to graph."""
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Status Legend', style='dashed')
            
            for i, (status, color) in enumerate(self.STATUS_COLORS.items()):
                legend.node(
                    f'legend_{i}',
                    label=status.value,
                    fillcolor=color,
                    style='filled',
                    shape='box',
                    fontsize='8'
                )
    
    def render_matplotlib(
        self, 
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Render graph using Matplotlib with NetworkX layout.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure if save_path provided
            
        Raises:
            ImportError: If matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib")
        
        import networkx as nx
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use NetworkX graph for layout
        pos = nx.spring_layout(self.graph._graph, k=1, iterations=50)
        
        # Draw nodes by status
        for status in TaskStatus:
            status_nodes = [
                node.task_id for node in self.graph.get_all_nodes() 
                if node.status == status
            ]
            if status_nodes:
                nx.draw_networkx_nodes(
                    self.graph._graph, 
                    pos,
                    nodelist=status_nodes,
                    node_color=self.STATUS_COLORS.get(status, "#E8E8E8"),
                    node_size=1000,
                    ax=ax
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph._graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Draw labels
        labels = {
            node.task_id: node.goal[:20] + "..." if len(node.goal) > 20 else node.goal
            for node in self.graph.get_all_nodes()
        }
        nx.draw_networkx_labels(
            self.graph._graph,
            pos,
            labels,
            font_size=8,
            ax=ax
        )
        
        # Add title
        ax.set_title(f"ROMA Task Graph - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=color, label=status.value)
            for status, color in self.STATUS_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics for analysis.
        
        Returns:
            Dictionary with graph metrics
        """
        nodes = self.graph.get_all_nodes()
        
        # Basic counts
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(list(self.graph._graph.edges())),
        }
        
        # Status breakdown
        status_counts = {}
        for status in TaskStatus:
            count = sum(1 for node in nodes if node.status == status)
            status_counts[status.value] = count
        stats["status_breakdown"] = status_counts
        
        # Type breakdown  
        type_counts = {}
        for task_type in TaskType:
            count = sum(1 for node in nodes if node.task_type == task_type)
            type_counts[task_type.value] = count
        stats["type_breakdown"] = type_counts
        
        # Graph structure metrics
        if nodes:
            depths = []
            for node in nodes:
                ancestors = self.graph.get_ancestors(node.task_id)
                depths.append(len(ancestors))
            
            stats["max_depth"] = max(depths) if depths else 0
            stats["avg_depth"] = sum(depths) / len(depths) if depths else 0
            
            # Fan-out metrics
            fanouts = [len(self.graph.get_children(node.task_id)) for node in nodes]
            stats["max_fanout"] = max(fanouts) if fanouts else 0
            stats["avg_fanout"] = sum(fanouts) / len(fanouts) if fanouts else 0
        
        return stats
    
    def export_summary(self, output_path: str) -> str:
        """
        Export comprehensive graph summary to file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            Path to created file
        """
        stats = self.get_graph_statistics()
        ascii_art = self.render_ascii(show_details=True)
        
        summary_lines = [
            "ROMA Task Graph Analysis",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "STATISTICS",
            "-" * 20,
        ]
        
        # Add statistics
        for key, value in stats.items():
            if isinstance(value, dict):
                summary_lines.append(f"{key.upper()}:")
                for subkey, subvalue in value.items():
                    summary_lines.append(f"  {subkey}: {subvalue}")
            else:
                summary_lines.append(f"{key}: {value}")
        
        summary_lines.extend([
            "",
            "GRAPH STRUCTURE", 
            "-" * 20,
            ascii_art
        ])
        
        # Write to file
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        
        return str(output_path)


# Utility functions
def visualize_graph(
    graph: DynamicTaskGraph,
    method: str = "ascii",
    **kwargs
) -> str:
    """
    Quick visualization function.
    
    Args:
        graph: DynamicTaskGraph to visualize
        method: Visualization method ("ascii", "graphviz", "matplotlib")
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Visualization output or path to saved file
    """
    visualizer = GraphVisualizer(graph)
    
    if method == "ascii":
        return visualizer.render_ascii(**kwargs)
    elif method == "graphviz":
        return visualizer.render_graphviz(**kwargs)
    elif method == "matplotlib":
        return visualizer.render_matplotlib(**kwargs)
    else:
        raise ValueError(f"Unknown visualization method: {method}")


async def interactive_graph_monitor(
    graph: DynamicTaskGraph,
    refresh_interval: float = 1.0,
    max_iterations: int = 100
) -> None:
    """
    Interactive console monitor for real-time graph changes.
    
    Args:
        graph: DynamicTaskGraph to monitor
        refresh_interval: Seconds between updates
        max_iterations: Maximum refresh cycles
    """
    import os
    
    visualizer = GraphVisualizer(graph)
    
    for i in range(max_iterations):
        # Clear console
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Show current state
        print(f"ROMA Task Graph Monitor - Iteration {i+1}/{max_iterations}")
        print("=" * 60)
        
        stats = visualizer.get_graph_statistics()
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"{key}: {value}")
        
        print("\nGraph Structure:")
        print(visualizer.render_ascii(show_details=True))
        
        # Wait for next iteration
        await asyncio.sleep(refresh_interval)