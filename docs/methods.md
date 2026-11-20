# Method notes

## Temperature scaling

For logits `z`, temperature `T > 0` produces `softmax(z / T)`. `TemperatureScaler` minimizes multiclass NLL on the caller-provided fit split over a bounded log-temperature interval. A higher temperature generally softens overconfident outputs; lower can sharpen underconfident outputs. The optimizer is deterministic and does not use test labels.

## Expected calibration error

For each confidence bin `B_m`, ECE is `sum_m (|B_m| / n) * |accuracy(B_m) - mean_confidence(B_m)|`. This implementation uses equal-width bins on `[0, 1]`, includes confidence 1 in the final bin, and returns zero if all predictions are correctly confident at 1. ECE is descriptive and bin-dependent, not a proper scoring rule.

## Selective prediction

Predictions are accepted when their maximum class probability is at least the chosen threshold. Coverage is the accepted fraction. Selective risk is the error fraction among accepted predictions. The threshold is policy input; this library does not choose an operational risk target.

## Split conformal sets

The score for a calibration example `(x_i, y_i)` is `s_i = 1 - p(y_i | x_i)`. For target miscoverage `alpha`, the code selects the `ceil((n + 1) * (1 - alpha))`-th order statistic, capped at `n`; a test set contains each class `k` for which `1 - p(k | x) <= q`. This is a simple non-randomized score, not adaptive prediction sets. It may return an empty set or a large set.

The predictor assumes the conformal-calibration examples and future examples are exchangeable and that the probability function was fixed independently of conformal calibration labels. The report therefore fits temperature on a separate split before using the conformal split. Empirical coverage on a finite test set can differ from nominal coverage.
