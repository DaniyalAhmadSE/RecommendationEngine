import json
from typing import Any

from utilities.file_utils import FileUtils


class JsonUtils:
    @staticmethod
    def get_text_list_from_json(
        output_directory_path: str,
        filename_key: str,
        keys_tree: list[str],
    ) -> list[str]:
        account_directories = FileUtils.get_directory_names_at_path(
            output_directory_path
        )
        all_texts: list[str] = []

        for each_directory in account_directories:
            file_in_each_directory: list[str] = FileUtils.get_file_names_at_path(
                f"{output_directory_path}/{each_directory}"
            )
            key_files = [
                each_file
                for each_file in file_in_each_directory
                if filename_key in each_file
            ]

            key_files.sort(reverse=True)

            all_objects: list[dict[str, Any]]
            with open(f"{output_directory_path}/{each_directory}/{key_files[0]}") as f:
                all_objects = json.load(f)

            for data in all_objects:
                content: Any = data
                for key in keys_tree:
                    content = content[key]
                all_texts.append(content)

        return all_texts
