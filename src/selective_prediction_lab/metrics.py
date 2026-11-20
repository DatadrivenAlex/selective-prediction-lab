from __future__ import annotations

import numpy as np

from ._validation import FloatArray, labels_array, probabilities_array


def expected_calibration_error(probabilities: object, labels: object, *, bins: int = 10) -> float:
    probs: FloatArray = probabilities_array(probabilities)
    target = labels_array(labels, probs.shape[0], probs.shape[1])
    if bins < 1:
        raise ValueError("bins must be positive")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == target).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return ece


def classification_metrics(
    probabilities: object, labels: object, *, bins: int = 10
) -> dict[str, float]:
    probs: FloatArray = probabilities_array(probabilities)
    target = labels_array(labels, probs.shape[0], probs.shape[1])
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[target]
    true_prob = np.clip(probs[np.arange(len(target)), target], 1e-15, 1.0)
    return {
        "accuracy": float((probs.argmax(axis=1) == target).mean()),
        "nll": float(-np.log(true_prob).mean()),
        "brier_score": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probs, target, bins=bins),
    }


def selective_metrics(
    probabilities: object, labels: object, *, threshold: float
) -> dict[str, float | int | None]:
    probs: FloatArray = probabilities_array(probabilities)
    target = labels_array(labels, probs.shape[0], probs.shape[1])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    confidence = probs.max(axis=1)
    accepted = confidence >= threshold
    count = int(accepted.sum())
    errors = probs.argmax(axis=1) != target
    return {
        "threshold": float(threshold),
        "accepted": count,
        "coverage": float(accepted.mean()),
        "selective_risk": float(errors[accepted].mean()) if count else None,
    }
