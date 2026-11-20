"""Model-agnostic tools for calibrated and selective classification."""

from .calibration import TemperatureScaler
from .conformal import SplitConformalClassifier
from .metrics import classification_metrics, expected_calibration_error
from .report import audit_report

__all__ = [
    "SplitConformalClassifier",
    "TemperatureScaler",
    "audit_report",
    "classification_metrics",
    "expected_calibration_error",
]
