"""Paired cluster bootstrap over cement sources (Section III-J)."""
from __future__ import annotations
import numpy as np
from .metrics import rmse


def paired_cluster_bootstrap(y_true, pred_a, pred_b, source_ids,
                             n_boot=10000, seed=42, alpha=0.05):
    """Resample whole sources with replacement; recompute RMSE(A) - RMSE(B).

    Negative favours A. Returns dict with mean diff, CI, and fraction of
    resamples in which A had the lower RMSE.

    Caution: with only 11 clusters this interval is fragile and tends to be
    anti-conservative. Report the cluster count alongside the interval.
    """
    y_true, pred_a, pred_b = map(lambda v: np.asarray(v, float), (y_true, pred_a, pred_b))
    source_ids = np.asarray(source_ids)
    sources = np.unique(source_ids)
    rows_by_source = {s: np.where(source_ids == s)[0] for s in sources}
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(sources, size=len(sources), replace=True)
        idx = np.concatenate([rows_by_source[s] for s in drawn])
        diffs[b] = rmse(y_true[idx], pred_a[idx]) - rmse(y_true[idx], pred_b[idx])

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff_mean": float(diffs.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "frac_favoring_a": float(np.mean(diffs < 0)),
            "n_clusters": int(len(sources)), "n_boot": int(n_boot),
            "point_diff": float(rmse(y_true, pred_a) - rmse(y_true, pred_b))}
