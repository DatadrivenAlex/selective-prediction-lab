# Selective Prediction Lab

A small, model-agnostic Python toolkit for the part of an ML system that happens after a classifier emits logits: probability calibration, abstention metrics, and split-conformal prediction sets.

It makes it possible to answer three practical questions with disjoint data splits:

1. Are the model's confidence scores aligned with observed accuracy?
2. What coverage and error rate result if low-confidence predictions are withheld?
3. How large are conformal prediction sets, and how often do they contain the true label on a held-out test set?

This is complementary to model selection and rollout gating: it evaluates the uncertainty attached to individual predictions, rather than deciding which model version to release.

## What it includes

- Scalar temperature scaling fitted by held-out multiclass negative log likelihood, with a bounded, dependency-light golden-section optimizer.
- Accuracy, negative log likelihood, multiclass Brier score, and expected calibration error (ECE).
- A confidence-threshold selective policy with accepted count, coverage, and selective risk.
- Split-conformal prediction sets using the finite-sample quantile for the nonconformity score `1 - p(true class)`.
- A CLI that reads three separately named splits and writes a JSON report; a deterministic synthetic example; unit tests and CI for Python 3.11–3.13.

## Quick start

```bash
uv sync --extra dev
python examples/make_sample.py
uv run prediction-audit examples/sample.json --alpha 0.1 --threshold 0.7 --output report.json
```

Install with pip instead:

```bash
python -m pip install -e .
prediction-audit examples/sample.json
```

The example generator uses synthetic logits and labels. It does not download a model, use an API key, train a model, or send data anywhere.

## Input format

Provide a JSON file with `temperature_fit`, `conformal_calibration`, and `test` objects. Each object has a two-dimensional `logits` array and a one-dimensional integer `labels` array. All arrays must use the same class order.

```json
{
  "temperature_fit": {"logits": [[2.1, 0.1], [0.2, 1.4]], "labels": [0, 1]},
  "conformal_calibration": {"logits": [[1.3, 0.7], [0.1, 1.8]], "labels": [0, 1]},
  "test": {"logits": [[0.9, 0.4], [0.2, 1.0]], "labels": [0, 1]}
}
```

In real use, use three appropriately separated samples. Fit temperature scaling on one, calibrate conformal scores on another, and reserve the test set for final assessment. Do not reuse training or test labels as calibration labels.

## Python API

```python
from selective_prediction_lab import audit_report

report = audit_report(
    calibration_logits=temperature_logits,
    calibration_labels=temperature_labels,
    conformal_logits=conformal_logits,
    conformal_labels=conformal_labels,
    test_logits=test_logits,
    test_labels=test_labels,
    alpha=0.1,
    confidence_threshold=0.7,
)
```

Individual components are also available: `TemperatureScaler`, `SplitConformalClassifier`, `classification_metrics`, `expected_calibration_error`, and `selective_metrics`.

## Interpretation and limitations

- Split-conformal coverage is a marginal, finite-sample statement under exchangeability between the conformal-calibration and future examples. It is not a conditional or per-group guarantee; distribution shift can break the assumption.
- The package reports empirical test-sample metrics. A single ECE value depends on binning and sample size; inspect reliability plots and uncertainty intervals before operational use.
- Temperature scaling adjusts one global scalar. It cannot repair class-conditional or subgroup miscalibration by itself.
- The confidence threshold is supplied by the user, not optimized for a business objective. Selective risk can be undefined when no rows are accepted.
- This repository does not train models, make predictions from raw inputs, provide a production serving API, certify safety, or include a real dataset or model benchmark.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
uv run prediction-audit examples/sample.json
```

See [`docs/methods.md`](docs/methods.md) for formulas and split-handling notes.

