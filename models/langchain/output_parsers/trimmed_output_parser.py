from langchain.schema import BaseOutputParser


class TrimmedOutputParser(BaseOutputParser[str]):
    def parse(self, text: str):
        return text.strip("\n ")
