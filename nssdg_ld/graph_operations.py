import numpy as np

class GraphOperations:
    def __init__(self, num_nodes, fan_in=3):
        """
        Initialize the graph operation handler.

        Args:
            num_nodes (int): Number of nodes in the network.
            fan_in (int): Maximum allowed number of parents per node.
        """
        self.num_nodes = num_nodes
        self.fan_in = fan_in

    def propose_graph_move(self, current_graph):
        """
        Propose a local move (add/remove/reverse an edge) to the graph.

        Returns:
            new_graph (networkx.DiGraph or None): Modified graph or None if move is invalid.
        """
        new_graph = current_graph.copy()
        nodes = list(current_graph.nodes())
        
        # Select one of the three possible operations
        operation = np.random.choice(['add', 'remove', 'reverse'])

        # ADD OPERATION: Add a directed edge u → v
        if operation == 'add':
            u, v = np.random.choice(nodes, 2, replace=True)
            
            # Check that edge u → v doesn't already exist and fan-in constraint is satisfied
            if not new_graph.has_edge(u, v) and len(list(new_graph.predecessors(v))) < self.fan_in:
                new_graph.add_edge(u, v)
            else:
                return None  # Invalid move → reject by returning None

        # REMOVE OPERATION: Remove a randomly chosen edge
        elif operation == 'remove':
            edges = list(current_graph.edges())
            if edges:
                u, v = edges[np.random.randint(len(edges))]
                new_graph.remove_edge(u, v)
            else:
                return None  # No edge to remove
        return new_graph
"""
        # REVERSE OPERATION: Reverse u → v to v → u if valid
        elif operation == 'reverse':
            edges = list(current_graph.edges())
            if edges:
                u, v = edges[np.random.randint(len(edges))]

                # Only reverse if v → u doesn't exist already and fan-in constraint holds for u
                if not new_graph.has_edge(v, u) and len(list(new_graph.predecessors(u))) < self.fan_in:
                    new_graph.remove_edge(u, v)
                    new_graph.add_edge(v, u)
                else:
                    return None
            else:
                return None
"""
