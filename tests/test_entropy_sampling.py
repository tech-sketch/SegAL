import math
import random
from typing import Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from segal.strategies import EntropySampling


class TestEntropySampling:
    """Test EntropySampling class"""

    def test_query_return_correct_result(self, probs) -> None:
        """Test EntropySampling.cal_scores function return correct result if log_probability runs after prediction"""
        # Arrange
        expected = np.array([1.415957])

        # Act
        scores = EntropySampling.cal_scores(probs)
        scores = np.around(scores, decimals=6)

        # Assert
        assert np.array_equal(scores, expected) is True
