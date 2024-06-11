from typing import Any, Literal

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame

# from sklearn._typing import ArrayLike, Float, Int
from sklearn.base import ClusterMixin
from sklearn.mixture import GaussianMixture
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import spmatrix

# import numpy as np
# from sklearn.mixture import GaussianMixture

# # Assuming you have 'embeddings' as a list of tweet embeddings
# # Each embedding is a numpy array

# # Convert the list of embeddings to a 2D NumPy array
# X = np.array(embeddings)

# # Specify the number of clusters (you can adjust this based on your requirements)
# num_clusters = 5

# # Perform Gaussian Mixture Model clustering
# gmm = GaussianMixture(n_components=num_clusters, random_state=42)
# gmm.fit(X)

# # Get the cluster assignments for each embedding
# cluster_probs = gmm.predict_proba(X)

# # Set a threshold for cluster membership
# # threshold = 0.5

# # Create a dictionary to store tweets for each cluster
# clusters = {i: [] for i in range(num_clusters)}

# # Assign tweets to clusters based on the threshold
# for tweet_index, cluster_probabilities in enumerate(cluster_probs):
#     for cluster_index, prob in enumerate(cluster_probabilities):
#         if prob > threshold:
#             clusters[cluster_index].append(tweet_index)

# # Get lists of tweets for each cluster
# tweets_for_clusters = {
#     cluster_index: [tweets[i] for i in cluster]
#     for cluster_index, cluster in clusters.items()
# }

# # Print or use the 'tweets_for_clusters' dictionary as needed
# for cluster_index, tweet_list in tweets_for_clusters.items():
#     print(f"Cluster {cluster_index}: {len(tweet_list)} tweets")
#     for tweet in tweet_list:
#         print(f"- {tweet}")


class FuzzyClusteringModel(ClusterMixin):
    # n_clusters=2, random_state
    def __init__(
        self,
        n_components: int = 1,
        cluster_similarity_threshold: float = 0.7,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 0.001,
        reg_covar: float = 0.000001,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal[
            "kmeans", "k-means++", "random", "random_from_data"
        ] = "kmeans",
        weights_init: ArrayLike | None = None,
        means_init: ArrayLike | None = None,
        precisions_init: ArrayLike | None = None,
        random_state: int | RandomState | None = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ) -> None:
        self.gaussian_mixture = GaussianMixture(
            n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.cluster_similarity_threshold = cluster_similarity_threshold

    def fit_predict(self, X: NDArray[Any] | DataFrame | spmatrix, y: Any = None):
        self.gaussian_mixture.fit(X)
        cluster_probabilities: NDArray[Any] = self.gaussian_mixture.predict_proba(X)

        cluster_assignments: list[list[Any]] = []

        for prob in cluster_probabilities:
            assigned_clusters = np.where(prob > self.cluster_similarity_threshold)[0]
            cluster_assignments.append(assigned_clusters.tolist())

        return cluster_assignments
