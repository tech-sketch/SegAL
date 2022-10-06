import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest


@pytest.fixture
def probs() -> np.array:
    """Probs for strategies test"""
    probs = np.array(
        [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.2], [0.3, 0.1]], [[0.4, 0.6], [0.4, 0.5]]]]
    )  # B,C,H,W = 1,3,2,2

    return probs
