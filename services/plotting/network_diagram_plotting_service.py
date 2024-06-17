from typing import Any, Callable, Iterable, Mapping, Sequence
import networkx as nx
from ipysigma import Sigma


class NetworkDiagramPlottingService:
    def __init__(
        self,
        layout_calculator: Callable[..., Mapping[Any, Any]],
        top_nodes_percentage: int,
    ) -> None:
        self.tags_graph: nx.Graph
        self.positions: dict[Any, Any]
        self.layout_calculator = layout_calculator
        self.top_nodes_percentage = top_nodes_percentage

    def init_tag_graph(self, tag_list_iterable: Iterable[Iterable[str]]):
        self.tags_graph = nx.Graph()
        for tag_list in tag_list_iterable:
            for tag in tag_list:
                if not self.tags_graph.has_node(tag):
                    self.tags_graph.add_node(tag)
                for other_tag in tag_list:
                    if tag != other_tag:
                        self.tags_graph.add_edge(tag, other_tag)

    def keep_top_degree_centrality_nodes(self, top_nodes_percentage: int | None):
        if top_nodes_percentage is None:
            top_nodes_percentage = self.top_nodes_percentage

        tag_degree_centralities: Mapping[Any, Any] = nx.degree_centrality(
            self.tags_graph
        )

        sorted_tag_nodes: Sequence[Any] = sorted(
            tag_degree_centralities,
            key=tag_degree_centralities.get,
            reverse=True,
        )

        top_node_count = top_nodes_percentage * len(sorted_tag_nodes) // 100

        for node in sorted_tag_nodes[top_node_count:]:
            self.tags_graph.remove_node(node)

    def compute_positions(self, iterations: int = 50):
        tag_positions = self.layout_calculator(
            self.tags_graph,
            iterations=iterations,
        )

        self.positions = {}
        for node, _ in tag_positions.items():
            self.positions[node] = {
                "x": tag_positions[node][0],
                "y": tag_positions[node][1],
            }

    def display_graph(self):
        # displaying the graph with a size mapped on degree and a color mapped on a categorical attribute of the nodes
        return Sigma(self.tags_graph, node_color="category", layout=self.positions)

    def plot_network_diagram(
        self,
        tag_list_iterable: Iterable[Iterable[str]],
        compute_position_iterations: int = 50,
        top_degree_centrality_nodes_percentage: int | None = None,
    ):
        self.init_tag_graph(tag_list_iterable)
        self.keep_top_degree_centrality_nodes(top_degree_centrality_nodes_percentage)
        self.compute_positions(compute_position_iterations)
        return self.display_graph()
