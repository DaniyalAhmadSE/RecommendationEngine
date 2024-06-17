from typing import Any, Iterable, Sequence
from sklearn.base import ClusterMixin
import numpy as np


class ClusteringService:
    def __init__(self, clustering_model: ClusterMixin) -> None:
        self._clustering_model = clustering_model
        pass

    @property
    def clustering_model(self):
        return self._clustering_model

    @staticmethod
    def get_cluster_ids(cluster_assignments: Sequence[int]):
        cluster_ids: list[int] = list(
            np.unique(np.array(cluster_assignments).flatten())
        )
        return cluster_ids

    def generate_clusters(self, items: Any):
        cluster_assignments: list[Any] = list(self.clustering_model.fit_predict(items))
        return cluster_assignments

    def get_cluster_assignments(self, items: Any):
        return self.generate_clusters(items)

    @staticmethod
    def get_raw_item_clusters_for_cluster_assignments(
        cluster_assignments: list[Any],
        raw_items: Iterable[Any],
    ) -> list[list[Any]]:
        unique_cluster_ids: list[int] = ClusteringService.get_cluster_ids(
            cluster_assignments
        )

        clusters: list[list[Any]] = [
            [
                item
                for item, cluster_assignment in zip(raw_items, cluster_assignments)
                if cluster_id == cluster_assignment
                or (
                    (type(cluster_assignment) is list)
                    and (cluster_id in cluster_assignment)
                )
            ]
            for cluster_id in unique_cluster_ids
        ]

        return clusters

    def get_raw_item_clusters_for_embedded_items(
        self,
        embeddings: Any,
        raw_items: Iterable[Any],
    ) -> list[list[Any]]:
        cluster_assignments = self.get_cluster_assignments(embeddings)
        clusters = self.get_raw_item_clusters_for_cluster_assignments(
            cluster_assignments, raw_items
        )
        return clusters
