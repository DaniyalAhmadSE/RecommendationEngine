from umap import UMAP

from typing import Any, Sequence
from numpy import float64
from numpy.typing import NDArray


class DimensionalityReductionService:
    def __init__(self) -> None:
        self._dimensionality_reducer = UMAP(min_dist=0.25, metric="cosine")
        pass

    @property
    def dimensionality_reducer(self):
        return self._dimensionality_reducer

    def reduce_dimensions(
        self, higher_dimensional_input: Any, n_dimensions: int
    ) -> NDArray[float64]:
        self.dimensionality_reducer.n_components = n_dimensions
        return self.dimensionality_reducer.fit_transform(higher_dimensional_input)
