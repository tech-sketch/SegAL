import numpy as np

from segal.strategies import MarginSampling


class TestMarginSampling:
    """Test MarginSampling class"""

    def test_cal_scores_return_correct_result(self, probs) -> None:
        """Test MarginSampling.cal_scores function return correct result if log_probability runs after prediction"""
        # Arrange
        expected = np.array([-0.175])

        # Act
        scores = MarginSampling.cal_scores(probs)

        # Assert
        assert np.array_equal(scores, expected) is True
