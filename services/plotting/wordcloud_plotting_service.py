from typing import Any
from matplotlib import pyplot as plt


class WordcloudPlottingService:
    def __init__(self, wordcloud: Any) -> None:
        self._wordcloud = wordcloud

    @property
    def wordcloud(self):
        return self._wordcloud

    def plot_wordcloud_for_text(self, text: str):
        self.wordcloud.generate_from_text(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(self.wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    def plot_wordclouds_for_text_list(
        self,
        text_list: list[str],
        titles: list[str] = [],
    ):
        for i, text in enumerate(text_list):
            if titles:
                print(titles[i])
            self.plot_wordcloud_for_text(text)
            print()
