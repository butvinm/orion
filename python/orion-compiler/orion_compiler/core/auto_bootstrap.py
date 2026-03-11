from __future__ import annotations

import itertools
import logging
import math
import types
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn

from orion_compiler.nn.operations import Bootstrap

from .level_dag import LevelDAG
from .network_dag import NetworkDAG

logger = logging.getLogger(__name__)


class BootstrapSolver:
    def __init__(
        self,
        net: nn.Module,
        network_dag: NetworkDAG,
        l_eff: int,
        context: types.SimpleNamespace | None = None,
    ) -> None:
        self.net = net
        self.network_dag = network_dag
        self.l_eff = l_eff
        self.context = context
        self.full_level_dag = LevelDAG(l_eff=l_eff, network_dag=network_dag)
        self.shortest_path: set[str] = set()

    def extract_all_residual_subgraphs(self) -> list[nx.DiGraph]:
        all_residual_subgraphs = []
        for fork in self.network_dag.residuals:
            subgraph = self.network_dag.extract_residual_subgraph(fork)
            all_residual_subgraphs.append(subgraph)

        return all_residual_subgraphs

    def sort_residual_subgraphs(self) -> list[tuple[str, list[list[str]], list[list[str]]]]:
        all_residual_subgraphs = self.extract_all_residual_subgraphs()

        residuals = []
        for i, (fork, join) in enumerate(self.network_dag.residuals.items()):
            subgraph = all_residual_subgraphs[i]
            paths = list(nx.all_simple_paths(subgraph, fork, join))

            unique_paths = []
            visited_children = set()
            for path in paths:
                if path[1] not in visited_children:
                    unique_paths.append(path)
                    visited_children.add(path[1])

            residuals.append((fork, paths, unique_paths))

        sorted_subgraphs = sorted(residuals, key=lambda x: len(x[1]))

        return sorted_subgraphs

    def first_solve_residual_subgraphs(self) -> dict[str, LevelDAG]:
        sorted_residual_subgraphs = self.sort_residual_subgraphs()
        self.network_dag.solved_residual_level_dags = {}

        for fork, _, paths in sorted_residual_subgraphs:
            aggregate_level_dag = LevelDAG(
                l_eff=self.l_eff, network_dag=self.network_dag, path=None
            )
            for path in paths:
                path_dag = nx.DiGraph()
                nodes_in_path = [(node, self.network_dag.nodes[node]) for node in path]
                edges_in_path = [
                    (u, v, self.network_dag[u][v]) for u, v in itertools.pairwise(path)
                ]
                path_dag.add_nodes_from(nodes_in_path)
                path_dag.add_edges_from(edges_in_path)
                aggregate_level_dag += LevelDAG(
                    l_eff=self.l_eff, network_dag=self.network_dag, path=path_dag
                )

            self.network_dag.solved_residual_level_dags[fork] = aggregate_level_dag

        return self.network_dag.solved_residual_level_dags

    def then_build_full_level_dag(self, solved_residual_level_dags: dict[str, LevelDAG]) -> None:
        all_forks = self.network_dag.residuals.keys()

        visited = set()
        for node in nx.topological_sort(self.network_dag):
            if node not in visited:
                if node in all_forks:
                    next_level_dag = solved_residual_level_dags[node]
                    subgraph = self.network_dag.extract_residual_subgraph(node)
                    visited.update(subgraph.nodes)
                else:
                    node_dag = nx.DiGraph()
                    node_dag.add_nodes_from([(node, self.network_dag.nodes[node])])
                    next_level_dag = LevelDAG(
                        l_eff=self.l_eff, network_dag=self.network_dag, path=node_dag
                    )
                    visited.add(node)

                self.full_level_dag.append(next_level_dag)

    def finally_solve_full_level_dag(self) -> int:
        heads = self.full_level_dag.head()
        tails = self.full_level_dag.tail()

        self.full_level_dag.add_node("source", weight=0)
        self.full_level_dag.add_node("target", weight=0)

        for head, tail in zip(heads, tails, strict=False):
            self.full_level_dag.add_edge("source", head, weight=0)
            self.full_level_dag.add_edge(tail, "target", weight=0)

        shortest_path, latency = self.full_level_dag.shortest_path(
            source="source", target="target"
        )

        if latency == float("inf"):
            raise ValueError(
                "Automatic bootstrap placement failed. First try increasing "
                "the length of your LogQ moduli chain in CKKSParams. If this "
                "fails, double check that the network was instantiated properly."
            )

        shortest_path = shortest_path[1:-1]

        reconstructed_path = set()
        for u, v in itertools.pairwise(shortest_path):
            edge = self.full_level_dag[u][v]
            reconstructed_path.update(edge["path"])

        self.shortest_path = reconstructed_path

        input_level = int(shortest_path[1].split("=")[-1])
        return input_level

    def solve(self) -> tuple[int, int, list[int]]:
        solved_residual_dags = self.first_solve_residual_subgraphs()
        self.then_build_full_level_dag(solved_residual_dags)
        input_level = self.finally_solve_full_level_dag()

        self.assign_levels_to_layers()
        num_bootstraps, bootstrapper_slots = self.mark_bootstrap_locations()

        return input_level, num_bootstraps, bootstrapper_slots

    def assign_levels_to_layers(self) -> None:
        for node in self.network_dag.nodes:
            node_module = self.network_dag.nodes[node]["module"]
            for layer in self.shortest_path:
                name = layer.split("@")[0]
                level = int(layer.split("=")[-1])

                if node == name:
                    self.network_dag.nodes[node]["level"] = level
                    if node_module:
                        node_module.level = level
                    break

    def mark_bootstrap_locations(self) -> tuple[int, list[int]]:
        node_map = {}
        for node in self.shortest_path:
            name = node.split("@")[0]
            node_map[name] = node

        query = LevelDAG(l_eff=self.l_eff, network_dag=self.network_dag, path=None)

        total_bootstraps = 0
        bootstrapper_slots = []

        for node in self.network_dag.nodes:
            node_w_level = node_map[node]

            children = self.network_dag.successors(node)
            self.network_dag.nodes[node]["bootstrap"] = False

            for child in children:
                child_w_level = node_map[child]
                _, curr_boots = query.estimate_bootstrap_latency(node_w_level, child_w_level)

                total_bootstraps += curr_boots
                if curr_boots > 0:
                    self.network_dag.nodes[node]["bootstrap"] = True
                    slots = self.get_bootstrap_slots(node)

                    if slots not in bootstrapper_slots:
                        bootstrapper_slots.append(slots)
                    break

        return total_bootstraps, bootstrapper_slots

    def get_bootstrap_slots(self, node: str) -> int:
        module = self.network_dag.nodes[node]["module"]
        assert self.context is not None
        max_slots = self.context.params.get_slots()

        elements = module.fhe_output_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(max_slots, curr_slots))

        return slots

    def plot_shortest_path(self, save_path: str = "", figsize: tuple[int, int] = (10, 10)) -> None:
        nodes = {}
        for node in self.shortest_path:
            name = node.split("@")[0]
            level = node.split("=")[-1]
            nodes[name] = level

        network = nx.DiGraph(self.network_dag)
        shortest_graph = nx.DiGraph()

        for name, level in nodes.items():
            shortest_graph.add_node(name, level=level)

        for u, v in network.edges():
            if u in nodes and v in nodes:
                shortest_graph.add_edge(u, v)

        try:
            pos = nx.nx_agraph.graphviz_layout(shortest_graph, prog="dot")
        except Exception:
            logger.warning("Graphviz not installed. Defaulting to worse visualization.")
            pos = nx.kamada_kawai_layout(shortest_graph)

        plt.figure(figsize=figsize)
        nx.draw(shortest_graph, pos, with_labels=False, arrows=True, font_size=8)

        node_labels = {
            node: f"{node}\n(level: {data['level']})"
            for node, data in shortest_graph.nodes(data=True)
        }
        nx.draw_networkx_labels(shortest_graph, pos, labels=node_labels, font_size=8)

        if save_path:
            plt.savefig(save_path)
        plt.show()


class BootstrapPlacer:
    def __init__(
        self, net: nn.Module, network_dag: NetworkDAG, context: types.SimpleNamespace
    ) -> None:
        self.net = net
        self.network_dag = network_dag
        self.context = context

    def place_bootstraps(self) -> None:
        """Insert explicit bootstrap nodes into the DAG.

        For each node marked with bootstrap=True, creates a boot_{idx} node
        and inserts it between the node and all its children in the DAG.
        """
        # Snapshot node list — we mutate the graph during iteration
        nodes_to_bootstrap = [
            node
            for node in self.network_dag.nodes
            if self.network_dag.nodes[node].get("bootstrap", False)
        ]

        for boot_idx, node in enumerate(nodes_to_bootstrap):
            module = self.network_dag.nodes[node]["module"]
            bootstrapper = self._create_bootstrapper(module)

            boot_name = f"boot_{boot_idx}"
            btp_level = module.level - module.depth

            # Set level/depth on the module object so the compiler can read it
            bootstrapper.level = btp_level
            bootstrapper.depth = 0

            # Insert bootstrap node into the DAG
            children = list(self.network_dag.successors(node))
            self.network_dag.add_node(
                boot_name,
                op="bootstrap",
                module=bootstrapper,
                level=btp_level,
                depth=0,
                bootstrap=False,
            )

            # Re-link edges: node -> boot_name -> each child
            for child in children:
                self.network_dag.remove_edge(node, child)
                self.network_dag.add_edge(boot_name, child)
            self.network_dag.add_edge(node, boot_name)

    def _create_bootstrapper(self, module: Any) -> Bootstrap:
        btp_input_level = module.level - module.depth
        btp_input_min = module.output_min
        btp_input_max = module.output_max

        bootstrapper = Bootstrap(btp_input_min, btp_input_max, btp_input_level)

        bootstrapper.fhe_input_shape = module.fhe_output_shape
        bootstrapper.fit(self.context)
        # NOTE: compile() removed — v2 format reads attributes directly

        return bootstrapper
