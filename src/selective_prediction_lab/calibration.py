from __future__ import annotations

import math

import numpy as np

from ._validation import FloatArray, IntArray, labels_array, logits_array, softmax


class TemperatureScaler:
    """Fit a scalar temperature by minimizing held-out multiclass negative log likelihood."""

    def __init__(self, min_temperature: float = 0.05, max_temperature: float = 20.0) -> None:
        if not (0 < min_temperature < max_temperature):
            raise ValueError("temperatures must satisfy 0 < min_temperature < max_temperature")
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.temperature_: float | None = None

    def fit(self, logits: object, labels: object, *, iterations: int = 96) -> TemperatureScaler:
        values = logits_array(logits)
        target: IntArray = labels_array(labels, values.shape[0], values.shape[1])
        if iterations < 8:
            raise ValueError("iterations must be at least 8")
        log_lo, log_hi = math.log(self.min_temperature), math.log(self.max_temperature)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left = log_hi - ratio * (log_hi - log_lo)
        right = log_lo + ratio * (log_hi - log_lo)

        def objective(log_temperature: float) -> float:
            probs = softmax(values / math.exp(log_temperature))
            return float(-np.log(np.clip(probs[np.arange(len(target)), target], 1e-15, 1.0)).mean())

        f_left, f_right = objective(left), objective(right)
        for _ in range(iterations):
            if f_left <= f_right:
                log_hi, right, f_right = right, left, f_left
                left = log_hi - ratio * (log_hi - log_lo)
                f_left = objective(left)
            else:
                log_lo, left, f_left = left, right, f_right
                right = log_lo + ratio * (log_hi - log_lo)
                f_right = objective(right)
        self.temperature_ = float(math.exp((log_lo + log_hi) / 2.0))
        return self

    def transform(self, logits: object) -> FloatArray:
        if self.temperature_ is None:
            raise RuntimeError("fit must be called before transform")
        values = logits_array(logits)
        return softmax(values / self.temperature_)

    def fit_transform(self, logits: object, labels: object, *, iterations: int = 96) -> FloatArray:
        return self.fit(logits, labels, iterations=iterations).transform(logits)
