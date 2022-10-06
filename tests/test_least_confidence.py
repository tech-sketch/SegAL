import math
import random
from typing import Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from segal.strategies import LeastConfidence


class TestLeastConfidence:
    """Test LeastConfidence class"""

    def test_cal_scores_return_correct_result(self, probs) -> None:
        """Test LeastConfidence.cal_scores function return correct result if log_probability runs after prediction"""
        # Arrange
        expected = np.array([-0.5])

        # Act
        scores = LeastConfidence.cal_scores(probs)

        # Assert
        assert np.array_equal(scores, expected) is True
