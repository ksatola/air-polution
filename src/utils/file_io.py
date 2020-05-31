import json
from typing import Dict, List
from pathlib import Path


def read_json(json_path: str) -> Dict:
    """
    Reads JSON file.
    :param json_path: full path to JSON file
    :return: Python dictionary
    """
    with open(json_path, 'r') as infile:
        return json.load(infile)


def get_file_paths(parent_path: str) -> List[str]:
    """
    Gets list of files in a specific folder.
    :param parent_path: path to a parent file directory
    :return: list of full paths with file names
    """
    curr_dir = Path(parent_path)
    files_list = []
    for entry in curr_dir.iterdir():
        files_list.append(str(entry))
    return files_list
