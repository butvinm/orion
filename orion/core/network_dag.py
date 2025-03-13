import networkx as nx
import matplotlib.pyplot as plt

from orion.nn.normalization import BatchNormNd

class NetworkDAG(nx.DiGraph):
    """
    Represents a neural network as a directed acyclic graph (DAG) using 
    NetworkX. This class builds a DAG from a symbolic trace of a PyTorch 
    network, identifies residual connections, and provides several useful 
    methods that we will use in our automatic bootstrap placement algorithm. 
    """
    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.residuals = {}

    def build_dag(self):
        """Builds the DAG representation of the neural network based on 
        the provided symbolic trace."""

        for node in self.trace.graph.nodes:
            # If the user assumes a default layer parameter (e.g. bias=False)
            # this will appear as an unconnected node with node.users = 0.
            # It is fine to disregard these cases.
            if len(node.users) > 0:
                module = None
                if node.op == "call_module":
                    module = self.trace.get_submodule(node.target)

                # Insert the node into the graph
                self.add_node(node.name, fx_node=node, op=node.op, module=module)
                for input_node in node.all_input_nodes:
                    self.add_edge(input_node.name, node.name)

    def find_residuals(self):
        """Finds pairs of fork/join nodes representing residual connections. 
        We consider a fork (join) node to be any Orion module or arithmetic 
        operation in our computational graph that has two or more children 
        (parents). Each residual connection creates a pair of fork/join nodes 
        that become the start/end nodes of each subgraph that we will
        ultimately extract in our automatic bootstrap placement algorithm."""

        # Residual connections in FHE are particularly difficult to deal with. 
        # Each residual connection creates a pair of fork and join nodes in our 
        # graph. For every fork, there is a join somewhere later in the graph. 
        # Our automatic bootstrap placement algorithm relies on extracting the 
        # subgraphs between pairs of fork/join nodes. This function nicely finds 
        # fork/join pairs and stores them in the self.residuals dictionary so 
        # we can reference them later.
        topo = list(nx.topological_sort(self))
        for start_node in list(self.nodes):
            successors = list(self.successors(start_node))

            # Fork node found
            if len(successors) > 1:
                paths = []
                # For each child of the node, get a path from that child to the 
                # last node in the network.
                for source in successors:
                    path = nx.shortest_path(self, source, topo[-1])
                    paths.append(set(path))

                # By set intersecting all paths from child -> end, we can find 
                # nodes common between all paths.
                common_nodes = list(set.intersection(*paths))

                # The join node is the "first" common node of the graph in 
                # topological order.
                end_node = [node for node in topo if node in common_nodes][0]

                # Finally, we'll insert special (auxiliary) fork/join nodes into 
                # the graph. This makes our automatic bootstrap placement 
                # algorithm slightly cleaner.
                fork, join = self.insert_fork_and_join_nodes(start_node, end_node)
                self.residuals[fork] = join

    def insert_fork_and_join_nodes(self, start, end):
        """Inserts special fork/join nodes into the graph around the residual
        connection. This makes our automatic bootstrap placement algorithm 
        slightly cleaner."""

        fork = f"{start}_fork"
        join = f"{end}_join"

        # Add fork/join nodes to the network
        self.add_node(fork, op="fork", module=None)
        self.add_node(join, op="join", module=None)

        # Insert fork node and update edges
        for child in list(self.successors(start)):
            self.remove_edge(start, child)
            self.add_edge(fork, child)
        self.add_edge(start, fork)

        # Insert join node and update edges
        for parent in list(self.predecessors(end)):
            self.remove_edge(parent, end)
            self.add_edge(parent, join)
        self.add_edge(join, end)

        return fork, join

    def extract_residual_subgraph(self, fork):
        """A helper function designed to extract the subgraphs between
        the fork/join nodes of a residual connection."""

        nodes_in_residual = set()
        edges_in_residual = set()

        # Get all paths from fork -> join and build up a set of unique
        # nodes/edges in its subgraph
        join = self.residuals[fork]
        for path in nx.all_simple_paths(self, fork, join):
            nodes_in_residual.update(path)
            edges_in_residual.update(zip(path[:-1], path[1:]))

        # Rebuild the subgraph from the nodes/edges
        residual_subgraph = nx.DiGraph()
        residual_subgraph.add_nodes_from(nodes_in_residual)
        residual_subgraph.add_edges_from(edges_in_residual)

        return residual_subgraph

    def remove_fused_batchnorms(self):
        """Removes BatchNorm nodes from the graph when it is known that they
        can be fused with preceding linear layers."""

        for node in list(self.nodes):
            node_module = self.nodes[node]["module"]
            
            if isinstance(node_module, BatchNormNd):
                # Get the parents and children of the batchnorm node
                parent_nodes = list(self.predecessors(node))
                child_nodes = list(self.successors(node))

                # Our tracer has already verified that the BN node only has
                # one parent.
                parent = parent_nodes[0]

                # Our fuser will have fused this BN node if it was possible,
                # and it's fused attribute will have been set. Remove this
                # BN node so that it isn't counted when we assign levels to
                # layers further into the compilation process.
                if node_module.fused:
                    for child in child_nodes:
                        self.add_edge(parent, child)
                    self.remove_node(node)

    def topological_sort(self):
        return nx.topological_sort(self)

    def plot(self, save_path="", figsize=(10,10)):
        """Plot the network digraph. For the best visualization, please install
        Graphviz and PyGraphviz."""

        try:
            pos = nx.nx_agraph.graphviz_layout(self, prog='dot')
        except:
            print("Graphviz not installed. Defaulting to worse visualization.\n")
            pos = nx.kamada_kawai_layout(self)
        
        plt.figure(figsize=figsize)
        nx.draw(self, pos, with_labels=True, arrows=True, font_size=8)

        if save_path:
            plt.savefig(save_path)
        plt.show()