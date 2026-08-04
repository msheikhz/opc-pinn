"""Metrics and fold aggregation."""
from __future__ import annotations
import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def all_metrics(y, p):
    return {"rmse": rmse(y, p), "mae": mae(y, p), "r2": r2(y, p)}


def aggregate_fold_metrics(folds):
    """Unweighted mean +/- std across folds (supplementary summary)."""
    out = {}
    for k in ("rmse", "mae", "r2"):
        vals = np.array([f[k] for f in folds], float)
        out[f"{k}_mean"], out[f"{k}_std"] = float(vals.mean()), float(vals.std(ddof=0))
    return out
