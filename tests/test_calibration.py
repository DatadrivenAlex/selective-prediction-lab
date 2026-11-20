import numpy as np
import pytest

from selective_prediction_lab.calibration import TemperatureScaler


def test_temperature_scaler_improves_nll_for_overconfident_logits() -> None:
    labels = np.tile([0, 1], 100)
    logits = np.full((200, 2), -3.0)
    logits[np.arange(200), labels] = 3.0
    # Make 20% of the examples ambiguous, so a warmer temperature is useful.
    logits[::5] = [0.4, 0.0]
    labels[::5] = 1
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated = scaler.transform(logits)
    from selective_prediction_lab.metrics import classification_metrics

    assert scaler.temperature_ is not None
    assert scaler.temperature_ > 1.0
    assert (
        classification_metrics(calibrated, labels)["nll"]
        < classification_metrics(_softmax(logits), labels)["nll"]
    )


def test_transform_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TemperatureScaler().transform([[1.0, 0.0]])


def test_rejects_invalid_temperature_bounds() -> None:
    with pytest.raises(ValueError):
        TemperatureScaler(0.0, 2.0)


def test_rejects_bad_labels() -> None:
    with pytest.raises(ValueError, match="class indices"):
        TemperatureScaler().fit([[1.0, 0.0], [0.0, 1.0]], [0, 2])


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
