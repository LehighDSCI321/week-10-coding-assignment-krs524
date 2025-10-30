"""
Implements VersatileDigraph with visualization libraries.
"""
class VersatileDigraph:
    """
    A class for representing a flexible directed graph.
    Each node has a unique ID and an optional value.
    Each edge connects two nodes, has a weight,
    and a unique name from the start node.
    """
    def __init__(self):
        """Set up the graph."""
        self.nodes = {}
        self.adj = {}

    def add_node(self, node_id: str, node_value=0):
        """Add a new node."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if not isinstance(node_value, (int, float)):
            raise TypeError("node_value must be a number.")
        if node_id in self.nodes:
            raise ValueError(f"node '{node_id}' already exists.")
        self.nodes[node_id] = node_value

    def add_edge(
        self,
        start_node_id: str,
        end_node_id: str,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """Add an edge from one node to another."""
        if not isinstance(start_node_id, str) or not isinstance(end_node_id, str):
            raise TypeError("start_node_id and end_node_id must be strings.")
        if edge_name is not None and not isinstance(edge_name, str):
            raise TypeError("edge_name must be a string or None.")
        if not isinstance(edge_weight, (int, float)):
            raise TypeError("edge_weight must be a number.")
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")
        if start_node_id not in self.nodes:
            self.add_node(start_node_id, 0 if start_node_value is None else start_node_value)
        if end_node_id not in self.nodes:
            self.add_node(end_node_id, 0 if end_node_value is None else end_node_value)
        if start_node_id not in self.adj:
            self.adj[start_node_id] = {}
        if end_node_id in self.adj[start_node_id]:
            raise ValueError(f"edge {start_node_id}->{end_node_id} already exists.")
        if edge_name is not None:
            for info in self.adj[start_node_id].values():
                if info.get('name') == edge_name:
                    raise ValueError(f"edge name '{edge_name}' already used from '{start_node_id}'")
        self.adj[start_node_id][end_node_id] = {'name': edge_name, 'weight': edge_weight}

    def get_nodes(self):
        """Return all nodes."""
        return list(self.nodes.keys())

    def get_edge_weight(self, start_node_id: str, end_node_id: str):
        """Return the weight of an edge."""
        if not isinstance(start_node_id, str) or not isinstance(end_node_id, str):
            raise TypeError("start_node_id and end_node_id must be strings.")
        if start_node_id not in self.adj or end_node_id not in self.adj[start_node_id]:
            raise KeyError(f"edge {start_node_id}->{end_node_id} not found.")
        return self.adj[start_node_id][end_node_id]['weight']

    def get_node_value(self, node_id: str):
        """Return the value of a node."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        return self.nodes[node_id]

    def print_graph(self):
        """Print nodes and edges."""
        for node_id, node_value in self.nodes.items():
            print(f"Node {node_id} with value {node_value}")
        for start_node_id, edges in self.adj.items():
            for end_node_id, edge_info in edges.items():
                name_part = f"and name {edge_info['name']}" if edge_info.get('name') else ""
                print((
                    f"Edge from {start_node_id} to {end_node_id} "
                    f"with weight {edge_info['weight']} {name_part}"
                ))

    def predecessors(self, node_id: str):
        """Return nodes that precede a node."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        return [start for start, edges in self.adj.items() if node_id in edges]

    def successors(self, node_id: str):
        """Return nodes that succeed a node."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        return list(self.adj.get(node_id, {}).keys())

    def successor_on_edge(self, node_id: str, edge_name: str):
        """Return the successor on a named edge."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        if not isinstance(edge_name, str):
            raise TypeError("edge_name must be a string.")
        return next(
            (end for end, info in self.adj.get(node_id, {}).items()
             if info.get('name') == edge_name),
            None
        )

    def in_degree(self, node_id: str):
        """Return the number of incoming edges."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        return len(self.predecessors(node_id))

    def out_degree(self, node_id: str):
        """Return the number of outgoing edges."""
        if not isinstance(node_id, str):
            raise TypeError("node_id must be a string.")
        if node_id not in self.nodes:
            raise KeyError(f"node '{node_id}' not found.")
        return len(self.adj.get(node_id, {}))

    def plot_graph(self):
        """Make a Graphviz plot of the graph."""
        try:
            from graphviz import Digraph  # pylint: disable=import-outside-toplevel
            viz_graph = Digraph()
            for node_id, value in self.nodes.items():
                viz_graph.node(node_id, f"{node_id}:{value}")
            for start_node, edges in self.adj.items():
                for end_node, info in edges.items():
                    label = (
                        f"{info['name']}:{info['weight']}"
                        if info.get('name') is not None
                        else str(info['weight'])
                    )
                    viz_graph.edge(start_node, end_node, label=label)
            return viz_graph
        except ImportError as e:
            raise ImportError(
                "Visualization requires the 'graphviz' package. "
                "Please install it with 'pip install graphviz'."
            ) from e

    def plot(self):
        """Alias for plot_graph."""
        return self.plot_graph()

    def plot_edge_weights(self):
        """Make a Bokeh bar chart of edge weights."""
        try:
            from bokeh.plotting import figure  # pylint: disable=import-outside-toplevel
            labels = []
            weights = []
            for start_node, edges in self.adj.items():
                for end_node, info in edges.items():
                    nm = info.get('name') if info.get('name') is not None else ""
                    labels.append(
                        f"{start_node}->{end_node} ({nm})"
                    )
                    weights.append(info['weight'])
            p = figure(x_range=labels, height=300, title="Edge Weights")
            p.vbar(x=labels, top=weights, width=0.9)
            p.xaxis.major_label_orientation = 1.0
            return p
        except ImportError as e:
            raise ImportError(
                "Visualization requires the 'bokeh' package. "
                "Please install it with 'pip install bokeh'."
            ) from e

if __name__ == "__main__":
    try:
        from bokeh.io import output_file  # pylint: disable=import-outside-toplevel
        from bokeh.plotting import show   # pylint: disable=import-outside-toplevel
        digraph = VersatileDigraph()

        # Define nodes
        digraph.add_node("Allentown", 66)
        digraph.add_node("Easton", 74)
        digraph.add_node("Bethlehem", 70)

        # Add directed edges with their names and weights
        digraph.add_edge("Allentown", "Easton", edge_name="US22E", edge_weight=17)
        digraph.add_edge("Easton", "Allentown", edge_name="US22W", edge_weight=17)
        digraph.add_edge("Allentown", "Bethlehem", edge_name="Hanover", edge_weight=6)
        digraph.add_edge("Bethlehem", "Allentown", edge_name="Hanover", edge_weight=6)
        digraph.add_edge("Easton", "Bethlehem", edge_name="Freemansburg", edge_weight=12)
        digraph.add_edge("Bethlehem", "Easton", edge_name="US22E", edge_weight=12)

        g = digraph.plot()
        g.render(filename="city_graph", format="png", cleanup=True)

        output_file("edge_weights.html")
        fig = digraph.plot_edge_weights()
        show(fig)
    except ImportError as e:
        raise ImportError(
            "Bokeh is required for visualization. Run 'pip install bokeh' to install it."
        ) from e

class BinaryGraph(VersatileDigraph):
    """Binary tree implementation with left/right child methods."""
    def __init__(self):
        """Initialize the binary graph with a root node."""
        super().__init__()
        self.add_node("Root", 0)

    def add_node_left(self, child_id: str, child_value=0, parent_id: str = None):
        """Add a left child node to a given parent node.
        If parent_id is None, add to 'Root'.
        """
        parent_id = "Root" if parent_id is None else parent_id
        if parent_id not in self.nodes:
            raise KeyError("parent node not found.")
        if self.successor_on_edge(parent_id, "L") is not None:
            raise ValueError("left child already exists.")
        if child_id in self.nodes and self.predecessors(child_id):
            raise ValueError("child already has a parent.")
        if child_id not in self.nodes:
            super().add_node(child_id, child_value)
        super().add_edge(parent_id, child_id, edge_name="L", edge_weight=0)

    def add_node_right(self, child_id: str, child_value=0, parent_id: str = None):
        """Add a right child node to a given parent node.
        If parent_id is None, add to 'Root'.
        """
        parent_id = "Root" if parent_id is None else parent_id
        if parent_id not in self.nodes:
            raise KeyError("parent node not found.")
        if self.successor_on_edge(parent_id, "R") is not None:
            raise ValueError("right child already exists.")
        if child_id in self.nodes and self.predecessors(child_id):
            raise ValueError("child already has a parent.")
        if child_id not in self.nodes:
            super().add_node(child_id, child_value)
        super().add_edge(parent_id, child_id, edge_name="R", edge_weight=0)

    def get_node_left(self, parent_id: str):
        """Return the left child ID of the given parent node."""
        if parent_id not in self.nodes:
            raise KeyError("parent node not found.")
        return self.successor_on_edge(parent_id, "L")

    def get_node_right(self, parent_id: str):
        """Return the right child ID of the given parent node."""
        if parent_id not in self.nodes:
            raise KeyError("parent node not found.")
        return self.successor_on_edge(parent_id, "R")

    def plot_graph(self):
        """Create and return a Graphviz Digraph of the tree."""
        try:
            from graphviz import Digraph # pylint: disable=import-outside-toplevel
            viz_graph = Digraph(
                graph_attr={'rankdir': 'TB'},
                node_attr={'shape': 'circle'},
                edge_attr={'arrowhead': 'none'}
            )
            for node_id, value in self.nodes.items():
                viz_graph.node(node_id, label=str(value))
            for start_node, edges in self.adj.items():
                for end_node in edges.keys():
                    viz_graph.edge(start_node, end_node)
            return viz_graph
        except ImportError as e:
            raise ImportError(
                "Visualization requires 'graphviz'. Please install it with 'pip install graphviz'."
            ) from e

    def plot(self):
        """Alias for plot_graph to simplify calls."""
        return self.plot_graph()

if __name__ == "__main__":
    tree = BinaryGraph()

    # root node
    tree.nodes["Root"] = 8

    # first level
    tree.add_node_left('71', 71, parent_id='Root')
    tree.add_node_right('41', 41, parent_id='Root')

    # Left Subtree
    tree.add_node_left('31', 31, parent_id='71')
    tree.add_node_right('10', 10, parent_id='71')
    tree.add_node_left('46', 46, parent_id='31')
    tree.add_node_right('51', 51, parent_id='31')
    tree.add_node_left('31_left', 31, parent_id='10')
    tree.add_node_right('21', 21, parent_id='10')

    # Right Subtree
    tree.add_node_left('11', 11, parent_id='41')
    tree.add_node_right('16', 16, parent_id='41')
    tree.add_node_left('13', 13, parent_id='11')

    dot = tree.plot_graph()
    dot.render("Binary_Graph", format="png", view=False)

class SortingTree(BinaryGraph):
    """Binary search tree that inherits from BinaryGraph."""
    def __init__(self, root_value=0):
        """Initialize the tree with a root node value."""
        super().__init__()
        self.nodes["Root"] = root_value

    def insert(self, value, node_id: str = "Root"):
        """Recursively insert value into the BST."""
        node_value = self.nodes[node_id]
        if value < node_value:
            left_child = self.get_node_left(node_id)
            if left_child is None:
                self.add_node_left(f"N{len(self.nodes)}", value, parent_id=node_id)
            else:
                self.insert(value, left_child)
        else:
            right_child = self.get_node_right(node_id)
            if right_child is None:
                self.add_node_right(f"N{len(self.nodes)}", value, parent_id=node_id)
            else:
                self.insert(value, right_child)

    def traverse(self, node_id: str = "Root"):
        """Recursively traverse the BST in order and print values."""
        left_child = self.get_node_left(node_id)
        if left_child is not None:
            self.traverse(left_child)
        print(self.nodes[node_id], end=" ")
        right_child = self.get_node_right(node_id)
        if right_child is not None:
            self.traverse(right_child)

class SortableDigraph(VersatileDigraph):
    """
    Directed graph with topological sort functionality.
    Based on Listing 4-10 from Python Algorithms (Kahn’s Algorithm).
    """
    def top_sort(self):
        """
        Return a list of nodes in topological order.
        Raises ValueError if the graph contains a cycle.
        """
        indeg = {u: 0 for u in self.nodes}
        for u in self.nodes:
            for v in self.successors(u):
                indeg[v] += 1

        zero_indegree_nodes = [u for u in self.nodes if indeg[u] == 0]
        zero_indegree_nodes.sort()

        sorted_nodes = []
        while zero_indegree_nodes:
            node = zero_indegree_nodes.pop(0)
            sorted_nodes.append(node)
            for succ in self.successors(node):
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    zero_indegree_nodes.append(succ)
            zero_indegree_nodes.sort()

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph has cycles; topological sort not possible.")
        return sorted_nodes

if __name__ == "__main__":
    g = SortableDigraph()
    for item in ["shirt", "tie", "jacket", "belt", "pants", "shoes", "socks", "vest"]:
        g.add_node(item)

    g.add_edge("shirt", "pants")
    g.add_edge("shirt", "vest")
    g.add_edge("shirt", "tie")
    g.add_edge("shirt", "jacket")
    g.add_edge("pants", "belt")
    g.add_edge("pants", "shoes")
    g.add_edge("socks", "shoes")
    g.add_edge("tie", "jacket")
    g.add_edge("belt", "jacket")
    g.add_edge("vest", "jacket")

    print(g.top_sort())

class TraversableDigraph(SortableDigraph):
    """
    Extends SortableDigraph with depth-first and breadth-first traversal.
    """
    def dfs(self, start_node_id: str):
        """
        Depth-first search starting at start_node_id.
        Yields nodes in the order they are first visited.
        """
        visited = {start_node_id}
        stack = list(self.successors(start_node_id))
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            stack.extend(self.successors(u))
            yield u

    def bfs(self, start_node_id: str):
        """
        Breadth-first search starting at start_node_id.
        Yields nodes layer by layer, using a deque for efficiency.
        """
        from collections import deque # pylint: disable=import-outside-toplevel
        visited = {start_node_id}
        queue = deque(self.successors(start_node_id))
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            for v in self.successors(u):
                if v not in visited:
                    queue.append(v)
            yield u

class DAG(TraversableDigraph):
    """
    Directed acyclic graph. Prevents cycle creation on edge insert.
    """
    def add_edge(
        self,
        start_node_id: str,
        end_node_id: str,
        start_node_value=None,
        end_node_value=None,
        edge_name=None,
        edge_weight=0,
    ):
        """
        Add a directed edge, unless it would introduce a cycle.
        Raises ValueError if the edge would create a cycle.
        """
        if start_node_id == end_node_id:
            raise ValueError("edge would create a cycle")

        if end_node_id in self.nodes:
            for reached in self.dfs(end_node_id):
                if reached == start_node_id:
                    raise ValueError("edge would create a cycle")

        return super().add_edge(
            start_node_id,
            end_node_id,
            start_node_value=start_node_value,
            end_node_value=end_node_value,
            edge_name=edge_name,
            edge_weight=edge_weight,
        )

if __name__ == "__main__":
    # --- Demo: clothing example graph (from the assignment image) ---
    g = TraversableDigraph()

    for name in ["shirt", "pants", "socks", "vest", "tie", "belt", "shoes", "jacket"]:
        g.add_node(name)

    edge_list = [
        ("shirt", "pants"),
        ("shirt", "vest"),
        ("shirt", "tie"),
        ("shirt", "jacket"),
        ("pants", "belt"),
        ("pants", "shoes"),
        ("socks", "shoes"),
        ("tie", "jacket"),
        ("belt", "jacket"),
        ("vest", "jacket"),
    ]
    for u, v in edge_list:
        g.add_edge(u, v)

    print("DFS from 'shirt':", list(g.dfs("shirt")))
    print("BFS from 'shirt':", list(g.bfs("shirt")))
    print("Topological order:", g.top_sort())
