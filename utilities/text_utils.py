from math import ceil
from typing import Any


class TextUtils:
    @staticmethod
    def concatenate_as_sentences(text_list: list[str], join_string: str = ". "):
        try:
            return join_string.join(text_list)
        except TypeError as e:
            print(f"LIST: {text_list}")
            raise e

    @staticmethod
    def concatenate_list_list_str_as_list_str(
        text_list_list: list[list[str]],
        join_string: str = ". ",
    ):
        concatenated: list[str] = [""] * len(text_list_list)
        for i, text_list in enumerate(text_list_list):
            concatenated[i] = TextUtils.concatenate_as_sentences(text_list)
        return concatenated

    @staticmethod
    def keyfy(text: str, separator: str = "_"):
        return text.replace(" ", separator)

    @staticmethod
    def chunkify_text_list(cluster: list[Any], chunk_count: int):
        chunks: list[list[Any]] = [[] for _ in range(chunk_count)]
        list_char_count = len(TextUtils.concatenate_as_sentences(cluster))
        chunking_threshold = ceil(list_char_count / chunk_count)
        chunk_index = 0

        for item in cluster:
            chunk_char_count = 0
            chunk_char_count = len(
                TextUtils.concatenate_as_sentences(chunks[chunk_index])
            )
            chunks[chunk_index].append(item)
            if chunk_char_count >= chunking_threshold:
                chunk_index += 1

        return chunks
