import json
from typing import Any, List

import numpy as np


def is_list_of_strings(lst: List[str]) -> bool:
    """Check if a list only contains strings

    Args:
        lst (List[str]): a list

    Returns:
        bool: if all element is string, return True, otherwise return False
    """
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False


def is_list_of_int(lst: List[int]) -> bool:
    """Check if a list only contains int

    Args:
        lst (List[int]): a list

    Returns:
        bool: if all element is int, return True, otherwise return False
    """
    if lst and isinstance(lst, list):
        return all(isinstance(elem, int) for elem in lst)
    else:
        return False


def is_array_of_bools(array: np.ndarray) -> bool:
    """Check if a array only contains bool

    Args:
        array (np.ndarray): numpy array

    Returns:
        bool: if all element types are bool, return True, otherwise return False
    """
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
