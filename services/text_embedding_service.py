from typing import Any, Iterable, Sequence
from numpy.typing import NDArray

import numpy as np

import tensorflow_hub as tf_hub
from os import getenv


class TextEmbeddingService:
    def __init__(self) -> None:
        self._text_embedding_model = tf_hub.load(
            getenv("UNIVERSAL_SENTENCE_ENCODER_PATH")
        )

    @property
    def text_embedding_model(self) -> Any:
        return self._text_embedding_model

    def embed_sentences(
        self,
        sentences: Iterable[str],
    ) -> Sequence[NDArray[np.float64]]:
        return self.text_embedding_model(sentences)
