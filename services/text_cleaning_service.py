from string import punctuation
from typing import Any, Iterable
from nltk import download as nltk_download
from nltk.corpus import stopwords as nltk_stopwords


class TextCleaningService:
    def __init__(
        self,
        non_nltk_stopwords: Iterable[str],
        tokenizer: Any,
        lemmatizer: Any,
        url_indicators: list[str] = ["t.co", "http", "bit.ly", "https", "www"],
    ) -> None:
        self.url_indicators = url_indicators
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.__download_nltk_resources()
        self.stopwords: set[str] = self.__get_all_stopwords(non_nltk_stopwords)

    def __download_nltk_resources(self):
        nltk_download("stopwords")
        nltk_download("punkt")
        nltk_download("wordnet")

    def __get_all_stopwords(self, non_nltk_stopwords: Iterable[str]) -> set[str]:
        return set(non_nltk_stopwords).union(nltk_stopwords.words("english"))

    @staticmethod
    def is_url(
        text: str,
        url_indicators: list[str] = ["t.co", "http", "bit.ly", "https", "www"],
    ):
        return any(substring in text for substring in url_indicators)

    @staticmethod
    def has_any_alphabet(text: str):
        return any(letter.isalpha() for letter in text)

    def filter_words(self, words: list[str], stopwords: Iterable[str]):
        return [
            word
            for word in words
            if (
                not word in punctuation
                and not word in stopwords
                and not TextCleaningService.is_url(word, self.url_indicators)
                and TextCleaningService.has_any_alphabet(word)
            )
        ]

    def clean_text_and_get_tokens(self, text: str):
        words = text.lower().split()
        filtered_words = self.filter_words(words, self.stopwords)
        tokenized_words = self.tokenizer.tokenize(" ".join(filtered_words))

        # TODO: Check Effect
        filtered_tokens = self.filter_words(tokenized_words, self.stopwords)

        lemmatized_words: list[str] = [
            self.lemmatizer.lemmatize(word) for word in filtered_tokens
        ]

        return lemmatized_words

    def get_text_cleaned(self, text: str):
        filtered_tokens: list[str] = self.clean_text_and_get_tokens(text)
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def get_text_list_cleaned(self, text_list: list[str]):
        cleaned_text_list = [""] * len(text_list)

        for i, text in enumerate(text_list):
            cleaned_text_list[i] = self.get_text_cleaned(text)

        return cleaned_text_list
