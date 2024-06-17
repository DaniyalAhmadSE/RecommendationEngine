from numpy import float64
from numpy.typing import NDArray
from typing import Any


class DimensionalityReductionService:
    def __init__(self, dimensionality_reducer: Any) -> None:
        self._dimensionality_reducer = dimensionality_reducer
        pass

    @property
    def dimensionality_reducer(self):
        return self._dimensionality_reducer

    def reduce_dimensions(self, higher_dimensional_input: Any) -> NDArray[float64]:
        return self.dimensionality_reducer.fit_transform(higher_dimensional_input)
