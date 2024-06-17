from shutil import copytree


class DataFetchingService:
    def __init__(self, source: str, destination: str) -> None:
        self.source = source
        self.destination = destination
        pass

    def fetch_latest(self):
        copytree(
            self.source,
            self.destination,
            dirs_exist_ok=True,
        )
