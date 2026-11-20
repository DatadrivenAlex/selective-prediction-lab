from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def logits_array(logits: object) -> FloatArray:
    result = np.asarray(logits, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("logits must be a non-empty (n_samples, n_classes>=2) matrix")
    if not np.isfinite(result).all():
        raise ValueError("logits must contain only finite values")
    return result


def labels_array(labels: object, n_samples: int, n_classes: int) -> IntArray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape[0] != n_samples:
        raise ValueError("labels must be a one-dimensional array matching the number of rows")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.isfinite(raw.astype(np.float64)).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("labels must be integer class indices")
    result = raw.astype(np.int64)
    if (result < 0).any() or (result >= n_classes).any():
        raise ValueError("labels must be class indices in [0, n_classes)")
    return result


def probabilities_array(probabilities: object) -> FloatArray:
    result = np.asarray(probabilities, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty (n_samples, n_classes>=2) matrix")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = result.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("each probability row must have a positive sum")
    return result / totals


def softmax(logits: object) -> FloatArray:
    values = logits_array(logits)
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
