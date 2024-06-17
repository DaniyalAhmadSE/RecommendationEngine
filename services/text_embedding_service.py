from typing import Any, Iterable, Sequence


class TextEmbeddingService:
    def __init__(self, text_embedding_model: Any) -> None:
        self._text_embedding_model = text_embedding_model
        pass

    @property
    def text_embedding_model(self) -> Any:
        return self._text_embedding_model

    def embed_sentences(self, sentences: Iterable[str]) -> Sequence[Sequence[float]]:
        return self.text_embedding_model(sentences)
