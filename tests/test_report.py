import numpy as np

from selective_prediction_lab.report import audit_report


def test_report_uses_three_disjoint_splits_and_returns_all_sections() -> None:
    rng = np.random.default_rng(7)
    splits = []
    for n in (90, 90, 120):
        labels = rng.integers(0, 3, size=n)
        logits = rng.normal(0, 0.5, size=(n, 3))
        logits[np.arange(n), labels] += 1.5
        splits.append((logits, labels))
    report = audit_report(
        calibration_logits=splits[0][0],
        calibration_labels=splits[0][1],
        conformal_logits=splits[1][0],
        conformal_labels=splits[1][1],
        test_logits=splits[2][0],
        test_labels=splits[2][1],
    )
    assert report["n_samples"] == {
        "temperature_fit": 90,
        "conformal_calibration": 90,
        "test": 120,
    }
    assert set(report["metrics"]) == {"uncalibrated", "temperature_scaled"}
    assert report["split_conformal"]["n_test"] == 120
    assert "marginal" in report["notes"][2]
