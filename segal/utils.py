import json
from typing import Any, List

import numpy as np


def is_list_of_strings(lst):
    """Check if a list only contains strings"""
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False


def is_array_of_bools(array):
    """Check if a list only contains strings"""
    if isinstance(array, np.ndarray) and len(list(array)) != []:
        return all(type(elem) != bool for elem in array)
    else:
        return False


def save_json(data: Any, save_path: str) -> None:
    """Save data to json format.
    Args:
        data (Any): The data to store.
        save_path (str): Path to save.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> List[Any]:
    """Read json data
    Args:
        path (str): Path of file to read.
    Returns:
        List[Any]: FTSE entity data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
