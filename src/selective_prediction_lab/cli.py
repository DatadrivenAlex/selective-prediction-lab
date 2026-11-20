from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report import audit_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit calibration and selective predictions from logits."
    )
    parser.add_argument(
        "input", type=Path, help="JSON file with disjoint calibration, conformal, and test splits"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="target conformal miscoverage, default 0.1"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="confidence threshold for abstention"
    )
    parser.add_argument("--bins", type=int, default=10, help="number of ECE bins")
    parser.add_argument(
        "--output", type=Path, help="write JSON report to this path instead of stdout"
    )
    args = parser.parse_args()
    data: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    required = {"temperature_fit", "conformal_calibration", "test"}
    if not required.issubset(data):
        parser.error(f"input must contain these splits: {', '.join(sorted(required))}")

    report = audit_report(
        calibration_logits=data["temperature_fit"]["logits"],
        calibration_labels=data["temperature_fit"]["labels"],
        conformal_logits=data["conformal_calibration"]["logits"],
        conformal_labels=data["conformal_calibration"]["labels"],
        test_logits=data["test"]["logits"],
        test_labels=data["test"]["labels"],
        alpha=args.alpha,
        confidence_threshold=args.threshold,
        bins=args.bins,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
