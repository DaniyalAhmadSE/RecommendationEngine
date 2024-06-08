import os


class FileUtils:
    @staticmethod
    def get_directory_names_at_path(path: str):
        return [
            directory_name
            for directory_name in os.listdir(path)
            if os.path.isdir(os.path.join(path, directory_name))
        ]

    @staticmethod
    def get_file_names_at_path(path: str):
        return os.listdir(path)
