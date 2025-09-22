"""
Graph Traversal Service for ROMA v2.0.

Provides graph analysis and traversal capabilities leveraging NetworkX algorithms.
Focuses on reusing proven graph algorithms rather than reimplementation.
"""

from typing import List, Set, Dict, Optional
import networkx as nx

from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.task_status import TaskStatus


class GraphTraversalService:
    """
    Graph traversal and analysis service using NetworkX algorithms.
    
    Key Design Philosophy:
    - Leverage NetworkX's proven, optimized algorithms
    - Do not reimplement standard graph algorithms
    - Provide domain-specific analysis on top of NetworkX primitives
    - Maintain high performance through efficient graph operations
    """
    
    def __init__(self, graph: DynamicTaskGraph):
        """
        Initialize GraphTraversalService.
        
        Args:
            graph: DynamicTaskGraph to analyze and traverse
        """
        self.graph = graph
    
    def get_topological_order(self) -> List[str]:
        """
        Get topological ordering of nodes using NetworkX.
        
        Returns:
            List of task IDs in topological order
            
        Raises:
            NetworkXError: If graph contains cycles
        """
        try:
            return list(nx.topological_sort(self.graph._graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, return empty list
            return []
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the graph using NetworkX.
        
        Returns:
            List of cycles, where each cycle is a list of task IDs.
            Empty list if no cycles found.
        """
        try:
            # find_cycle returns a single cycle if one exists
            cycle_edges = list(nx.find_cycle(self.graph._graph, orientation='original'))
            if cycle_edges:
                # Extract nodes from cycle edges
                cycle_nodes = [edge[0] for edge in cycle_edges]
                return [cycle_nodes]
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
        
        return []
    
    def find_parallel_execution_paths(self) -> List[Set[str]]:
        """
        Identify nodes that can execute in parallel using topological levels.
        
        Returns:
            List of sets, where each set contains task IDs that can execute in parallel
        """
        if len(self.graph.get_all_nodes()) == 0:
            return []
        
        try:
            # Get topological sort first
            topo_order = list(nx.topological_sort(self.graph._graph))
            
            # Calculate the topological level for each node
            levels = {}
            for node_id in topo_order:
                # Level is max level of predecessors + 1, or 0 for root nodes
                predecessors = list(self.graph._graph.predecessors(node_id))
                if not predecessors:
                    levels[node_id] = 0
                else:
                    max_predecessor_level = max(levels[pred] for pred in predecessors)
                    levels[node_id] = max_predecessor_level + 1
            
            # Group nodes by level for parallel execution
            level_groups: Dict[int, Set[str]] = {}
            for node_id, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = set()
                level_groups[level].add(node_id)
            
            # Return levels in order
            max_level = max(level_groups.keys()) if level_groups else -1
            return [level_groups[i] for i in range(max_level + 1)]
            
        except nx.NetworkXUnfeasible:
            # Graph has cycles, cannot determine parallel execution
            return []
    
    def get_node_dependencies(self, task_id: str) -> List[str]:
        """
        Get direct dependencies (predecessors) of a node.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of task IDs that this task depends on
        """
        if task_id not in self.graph._graph:
            return []
        
        return list(self.graph._graph.predecessors(task_id))
    
    def get_node_dependents(self, task_id: str) -> List[str]:
        """
        Get direct dependents (successors) of a node.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of task IDs that depend on this task
        """
        if task_id not in self.graph._graph:
            return []
        
        return list(self.graph._graph.successors(task_id))
    
    def calculate_graph_depth(self) -> int:
        """
        Calculate the maximum depth of the graph.
        
        Returns:
            Maximum distance from root to any leaf node
        """
        if len(self.graph.get_all_nodes()) == 0:
            return 0
        
        try:
            # Use NetworkX's longest path algorithm on DAG
            # First, we need to find all nodes with no predecessors (roots)
            all_nodes = set(self.graph._graph.nodes())
            nodes_with_predecessors = set()
            
            for node in all_nodes:
                if list(self.graph._graph.predecessors(node)):
                    nodes_with_predecessors.add(node)
            
            root_nodes = all_nodes - nodes_with_predecessors
            
            if not root_nodes:
                # No root nodes found, might be a cycle or empty
                return 0
            
            max_depth = 0
            
            # Calculate depth from each root
            for root in root_nodes:
                try:
                    # Use DFS to find longest path from this root
                    depth = self._calculate_max_depth_from_node(root)
                    max_depth = max(max_depth, depth)
                except:
                    continue
                    
            return max_depth
            
        except nx.NetworkXUnfeasible:
            return 0
    
    def _calculate_max_depth_from_node(self, node_id: str, visited: Optional[Set[str]] = None) -> int:
        """
        Helper method to calculate maximum depth from a given node using DFS.
        
        Args:
            node_id: Starting node
            visited: Set to track visited nodes (cycle detection)
            
        Returns:
            Maximum depth from this node
        """
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return 0  # Cycle detected, stop recursion
            
        visited.add(node_id)
        
        successors = list(self.graph._graph.successors(node_id))
        if not successors:
            return 1  # Leaf node
        
        max_child_depth = 0
        for successor in successors:
            child_depth = self._calculate_max_depth_from_node(successor, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth + 1
    
    def is_node_ready_for_execution(self, task_id: str) -> bool:
        """
        Check if a node is ready for execution based on dependencies.
        
        A node is ready if:
        1. It exists in the graph
        2. It has PENDING status
        3. All its dependencies are COMPLETED
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if node is ready for execution, False otherwise
        """
        # Check if node exists
        node = self.graph.get_node(task_id)
        if node is None:
            return False
        
        # Check if node is in PENDING status
        if node.status != TaskStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        dependencies = self.get_node_dependencies(task_id)
        for dep_id in dependencies:
            dep_node = self.graph.get_node(dep_id)
            if dep_node is None or dep_node.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def get_critical_path(self) -> List[str]:
        """
        Find the critical path (longest path) through the graph.
        
        Returns:
            List of task IDs representing the critical path
        """
        if len(self.graph.get_all_nodes()) == 0:
            return []
        
        try:
            # For DAG, we can find longest path using topological sort
            topo_order = list(nx.topological_sort(self.graph._graph))
            
            if not topo_order:
                return []
            
            # Calculate longest path to each node
            longest_path_to = {}
            path_predecessor: Dict[str, Optional[str]] = {}
            
            for node_id in topo_order:
                predecessors = list(self.graph._graph.predecessors(node_id))
                
                if not predecessors:
                    # Root node
                    longest_path_to[node_id] = 1
                    path_predecessor[node_id] = None
                else:
                    # Find predecessor with longest path
                    max_length = 0
                    best_predecessor = None
                    
                    for pred in predecessors:
                        if longest_path_to[pred] > max_length:
                            max_length = longest_path_to[pred]
                            best_predecessor = pred
                    
                    longest_path_to[node_id] = max_length + 1
                    path_predecessor[node_id] = best_predecessor
            
            # Find node with maximum path length
            if not longest_path_to:
                return []
            
            end_node = max(longest_path_to.keys(), key=lambda x: longest_path_to[x])
            
            # Reconstruct path
            path = []
            current: Optional[str] = end_node
            
            while current is not None:
                path.append(current)
                current = path_predecessor.get(current)
            
            path.reverse()
            return path
            
        except nx.NetworkXUnfeasible:
            return []
