import numpy as np
import pytest

from selective_prediction_lab.metrics import (
    classification_metrics,
    expected_calibration_error,
    selective_metrics,
)


def test_perfect_predictions_have_zero_calibration_error() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probabilities, [0, 1]) == 0.0
    metrics = classification_metrics(probabilities, [0, 1])
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == 0.0


def test_selective_metrics_report_abstention_tradeoff() -> None:
    probabilities = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
    result = selective_metrics(probabilities, [0, 1, 0], threshold=0.8)
    assert result["accepted"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["selective_risk"] == pytest.approx(0.5)


def test_empty_selection_has_no_defined_risk() -> None:
    result = selective_metrics([[0.6, 0.4]], [0], threshold=0.9)
    assert result["accepted"] == 0
    assert result["selective_risk"] is None


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        classification_metrics([[1.1, -0.1]], [0])
