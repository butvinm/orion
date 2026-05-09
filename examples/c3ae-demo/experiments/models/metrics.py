"""Shared FPR/FNR/Accuracy metric helper for the C3AE experiments.

Centralizes the binary-classification metric computation used by
``models/train.py`` (per-epoch validation) and ``models/eval.py`` (final
cleartext CSV). Keeping the formula in one place prevents drift — the two
call sites historically had identical inline implementations.

Decision rule: ``probs >= 0.5`` -> predicted adult.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute FPR / FNR / accuracy from sigmoid probabilities and binary targets.

    Decision rule: ``probs >= 0.5`` -> predicted adult.

    Empty-input convention: when ``targets`` has length 0 (e.g. an empty
    boundary band), returns zeros for all metrics. We pick zero rather than
    NaN so the CSV downstream is plain numeric and easy to consume; the
    accompanying ``n`` field disambiguates "0 because perfect" vs "0 because
    empty". The same convention is applied per-class: a scope with no minors
    reports ``fpr=0.0``, a scope with no adults reports ``fnr=0.0``.

    Args:
        probs:   1-D array of sigmoid outputs in ``[0, 1]``.
        targets: 1-D array of ground-truth labels in ``{0.0, 1.0}``.

    Returns:
        ``{"n": int, "fpr": float, "fnr": float, "accuracy": float}``. ``n``
        is an integer count; the other values are floats in ``[0, 1]``.
    """
    probs = np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    n = int(targets.shape[0])

    if n == 0:
        return {"n": 0, "fpr": 0.0, "fnr": 0.0, "accuracy": 0.0}

    pred_adult = probs >= 0.5
    true_adult = targets >= 0.5
    minors_mask = ~true_adult

    fpr = float(pred_adult[minors_mask].mean()) if minors_mask.sum() > 0 else 0.0
    fnr = float((~pred_adult[true_adult]).mean()) if true_adult.sum() > 0 else 0.0
    accuracy = float((pred_adult == true_adult).mean())

    return {"n": n, "fpr": fpr, "fnr": fnr, "accuracy": accuracy}
