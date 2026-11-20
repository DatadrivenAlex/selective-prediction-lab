import numpy as np
import pytest

from selective_prediction_lab.conformal import SplitConformalClassifier


def test_conformal_threshold_uses_finite_sample_rank() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    labels = np.zeros(4, dtype=int)
    predictor = SplitConformalClassifier(alpha=0.25).fit(probs, labels)
    # ceil((n+1)*(1-alpha)) = ceil(3.75) = 4; score order is 0.1, 0.2, 0.3, 0.4.
    assert predictor.threshold_ == pytest.approx(0.4)
    assert predictor.n_calibration_ == 4


def test_predict_sets_and_evaluation() -> None:
    calibration = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    predictor = SplitConformalClassifier(alpha=0.25).fit(calibration, [0, 0, 0, 0])
    test = np.array([[0.65, 0.35], [0.45, 0.55]])
    sets = predictor.predict_sets(test)
    assert sets == [[0], []]
    result = predictor.evaluate(test, [0, 1])
    assert result["empirical_coverage"] == 0.5
    assert result["mean_set_size"] == 0.5
    assert result["empty_set_rate"] == 0.5


def test_rejects_invalid_alpha_and_unfitted_use() -> None:
    with pytest.raises(ValueError):
        SplitConformalClassifier(alpha=1.0)
    with pytest.raises(RuntimeError, match="fit"):
        SplitConformalClassifier().predict_sets([[0.5, 0.5]])
