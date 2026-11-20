from __future__ import annotations

from typing import Any

from ._validation import labels_array, logits_array, softmax
from .calibration import TemperatureScaler
from .conformal import SplitConformalClassifier
from .metrics import classification_metrics, selective_metrics


def audit_report(
    *,
    calibration_logits: object,
    calibration_labels: object,
    conformal_logits: object,
    conformal_labels: object,
    test_logits: object,
    test_labels: object,
    alpha: float = 0.1,
    confidence_threshold: float = 0.7,
    bins: int = 10,
) -> dict[str, Any]:
    """Fit on calibration data, conformalize on a separate split, and report test results."""
    calibration = logits_array(calibration_logits)
    conformal = logits_array(conformal_logits)
    test = logits_array(test_logits)
    class_count = calibration.shape[1]
    if conformal.shape[1] != class_count or test.shape[1] != class_count:
        raise ValueError("all splits must use the same number and order of classes")
    calibration_y = labels_array(calibration_labels, calibration.shape[0], class_count)
    conformal_y = labels_array(conformal_labels, conformal.shape[0], class_count)
    test_y = labels_array(test_labels, test.shape[0], class_count)

    scaler = TemperatureScaler().fit(calibration, calibration_y)
    raw_test = softmax(test)
    calibrated_test = scaler.transform(test)
    conformal_probs = scaler.transform(conformal)
    predictor = SplitConformalClassifier(alpha=alpha).fit(conformal_probs, conformal_y)
    return {
        "n_samples": {
            "temperature_fit": int(len(calibration_y)),
            "conformal_calibration": int(len(conformal_y)),
            "test": int(len(test_y)),
        },
        "temperature": scaler.temperature_,
        "metrics": {
            "uncalibrated": classification_metrics(raw_test, test_y, bins=bins),
            "temperature_scaled": classification_metrics(calibrated_test, test_y, bins=bins),
        },
        "selective_prediction": selective_metrics(
            calibrated_test, test_y, threshold=confidence_threshold
        ),
        "split_conformal": {
            "alpha": alpha,
            "threshold": predictor.threshold_,
            **predictor.evaluate(calibrated_test, test_y),
        },
        "notes": [
            "Temperature scaling is fitted only on the temperature-fit split.",
            "Conformal scores use a separate conformal-calibration split.",
            "Coverage is marginal under exchangeability, not a per-group guarantee.",
            "Test metrics are estimates for this test sample, not production guarantees.",
        ],
    }
