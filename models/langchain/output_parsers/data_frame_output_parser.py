import pandas as pd
from langchain.schema import BaseOutputParser


class DataFrameOutputParser(BaseOutputParser):
    def parse(self, text: str) -> pd.DataFrame:
        """
        Parse the output string into a pandas DataFrame.

        Args:
            output (str): The string output from the model.

        Returns:
            pd.DataFrame: The parsed output as a DataFrame.
        """
        lines = text.strip().split("\n")
        header = lines[0].split(",")
        data = [line.split(",") for line in lines[1:]]

        df = pd.DataFrame(data, columns=header)

        return df
