from typing import Any
from langchain.schema import BaseOutputParser
from json import loads


class DictOutputParser(BaseOutputParser[dict[str, Any] | str]):
    def parse(self, text: str) -> dict[str, Any] | str:
        found_opening_braces = False
        found_closing_braces = False

        while True:
            try:
                i = text.index("{")
                found_opening_braces = True
                j = text.rindex("}") + 1
                found_closing_braces = True
                return loads(text[i:j])

            except:
                if found_opening_braces and not found_closing_braces:
                    text += '"}'
                    continue
                break

        return "Text could not be parsed: \n" + text
