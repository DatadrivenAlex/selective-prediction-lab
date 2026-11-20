from __future__ import annotations

import math

import numpy as np

from ._validation import FloatArray, IntArray, labels_array, probabilities_array


class SplitConformalClassifier:
    """Marginal split-conformal prediction sets from a disjoint calibration sample."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self.threshold_: float | None = None
        self.n_calibration_: int | None = None

    def fit(self, probabilities: object, labels: object) -> SplitConformalClassifier:
        probs: FloatArray = probabilities_array(probabilities)
        target: IntArray = labels_array(labels, probs.shape[0], probs.shape[1])
        scores = 1.0 - probs[np.arange(len(target)), target]
        n = len(scores)
        rank = min(n, math.ceil((n + 1) * (1.0 - self.alpha)))
        self.threshold_ = float(np.partition(scores, rank - 1)[rank - 1])
        self.n_calibration_ = n
        return self

    def predict_sets(self, probabilities: object) -> list[list[int]]:
        if self.threshold_ is None:
            raise RuntimeError("fit must be called before predict_sets")
        probs: FloatArray = probabilities_array(probabilities)
        if probs.shape[1] < 2:
            raise ValueError("at least two classes are required")
        included = (1.0 - probs) <= self.threshold_
        return [np.flatnonzero(row).astype(int).tolist() for row in included]

    def evaluate(self, probabilities: object, labels: object) -> dict[str, float | int]:
        probs: FloatArray = probabilities_array(probabilities)
        target = labels_array(labels, probs.shape[0], probs.shape[1])
        sets = self.predict_sets(probs)
        covered = np.array(
            [int(label) in prediction for label, prediction in zip(target, sets, strict=True)]
        )
        sizes = np.array([len(prediction) for prediction in sets], dtype=np.float64)
        return {
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "empty_set_rate": float((sizes == 0).mean()),
            "n_test": int(len(target)),
        }
