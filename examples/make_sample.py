"""Create a deterministic, synthetic three-class example without downloading a model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
output: dict[str, dict[str, list]] = {}
for name, count, scale in [
    ("temperature_fit", 240, 2.0),
    ("conformal_calibration", 240, 2.0),
    ("test", 300, 2.0),
]:
    labels = rng.integers(0, 3, size=count)
    logits = rng.normal(0.0, 0.7, size=(count, 3))
    logits[np.arange(count), labels] += scale
    output[name] = {"logits": logits.round(5).tolist(), "labels": labels.tolist()}

Path(__file__).with_name("sample.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)
print("Wrote examples/sample.json (synthetic logits and labels).")
