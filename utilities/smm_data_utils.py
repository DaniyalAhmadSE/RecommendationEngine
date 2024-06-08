from string import punctuation
from typing import Iterable


class SmmDataUtils:
    @staticmethod
    def clean_tag(text: str, characters_to_remove: list[str] = []):
        return "".join(
            [
                character.strip(punctuation)
                for character in text.lower()
                if character not in characters_to_remove
            ]
        )

    @staticmethod
    def extract_tags_from_text(
        text: str,
        tag_specifier: str,
        characters_to_remove: list[str] = [],
    ):
        return [
            tag_specifier + SmmDataUtils.clean_tag(word, characters_to_remove)
            for word in text.split()
            if word.startswith(tag_specifier)
        ]

    @staticmethod
    def extract_tags_from_texts(
        texts: Iterable[str],
        tag_specifier: str,
        characters_to_remove: list[str] = [],
    ):
        return [
            SmmDataUtils.extract_tags_from_text(
                tweet, tag_specifier, characters_to_remove
            )
            for tweet in texts
        ]
