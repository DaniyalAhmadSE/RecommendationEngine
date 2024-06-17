from typing import Any, Sequence
import numpy as np
from sklearn.base import ClusterMixin

from services.clustering.clustering_service import ClusteringService

from collections import deque


class HierarchicalClusteringService(ClusteringService):
    def __init__(self, clustering_model: ClusterMixin) -> None:
        super().__init__(clustering_model)
        self._linkage_matrix = None
        self._hierarchy_tree = None

    @property
    def linkage_matrix(self):
        if self._linkage_matrix is None:
            self._linkage_matrix = self._get_linkage_matrix()
        return self._linkage_matrix

    @property
    def hierarchy_tree(self) -> dict[str, Any]:
        if not self._hierarchy_tree:
            self._hierarchy_tree = self.initialize_hierarchy_tree()
        return self._hierarchy_tree

    def _get_linkage_matrix(self):
        # create the counts of samples under each node
        counts = np.zeros(self.clustering_model.children_.shape[0])
        n_samples = len(self.clustering_model.labels_)
        for i, merge in enumerate(self.clustering_model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.clustering_model.children_, self.clustering_model.distances_, counts]
        ).astype(float)

        return linkage_matrix

    def _search_node(self, name: str) -> dict[str, int] | None:
        stack = deque([self.hierarchy_tree])
        while stack:
            current_node = stack.pop()

            if current_node["name"] == name:
                return current_node

            stack.extend(current_node["children"])
        return None

    def _add_items_to_ancestors(self, node: dict[str, Any]):
        items: list[Any] = node["items"]
        parent: dict[str, Any] | None = self._search_node(node["parent"])
        while parent is not None:
            parent["items"] += items
            parent = self._search_node(parent["parent"])

    def initialize_hierarchy_tree(self, max_depth: int | None = None):
        linkage_matrix = self.linkage_matrix

        num_rows, _ = linkage_matrix.shape
        inter = {}
        i = 0
        for row in linkage_matrix:
            i += 1
            inter[float(i + num_rows)] = [row[0], row[1]]

        tree: dict[str, Any] = {
            "name": float(i + num_rows),
            "parent": None,
            "items": [],
            "children": [],
        }

        stack: deque[tuple[dict[str, Any], int]] = deque([(tree, 0)])

        while stack:
            current_node, depth = stack.pop()
            node_name = current_node["name"]

            if depth == max_depth or node_name not in inter:
                continue

            for n in inter[node_name]:
                node: dict[str, Any] = {
                    "name": n,
                    "parent": node_name,
                    "items": [],
                    "children": [],
                }
                current_node["children"].append(node)
                stack.append((node, depth + 1))

        self._hierarchy_tree = tree
        return tree

    def add_items_to_hierarchy_tree_nodes(self, items: Sequence[Any]):
        stack = deque([self.hierarchy_tree])
        labels = list(self.clustering_model.labels_)

        while stack:
            current_node = stack.pop()

            if current_node["name"] in labels:
                current_node["items"] = [items[labels.index(current_node["name"])]]
                self._add_items_to_ancestors(current_node)
            elif not current_node["items"]:
                current_node["items"] = []

            stack.extend(current_node["children"])
