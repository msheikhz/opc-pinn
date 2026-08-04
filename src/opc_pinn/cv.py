"""Cross-validation splitters (Section III-G)."""
from __future__ import annotations
import numpy as np


def source_stratified_kfold(source_ids, n_folds=5, seed=42):
    """Distribute each source's rows round-robin across folds.

    Off-the-shelf splitters do not fit groups of 2-3 rows: StratifiedKFold
    stratifies on a label not a group, and GroupKFold holds an entire group out.
    The fold-label order is re-randomised independently per source.
    """
    source_ids = np.asarray(source_ids)
    rng = np.random.default_rng(seed)
    fold_of_row = np.empty(len(source_ids), dtype=int)
    for s in np.unique(source_ids):
        idx = np.where(source_ids == s)[0]
        idx = rng.permutation(idx)
        labels = np.arange(len(idx)) % n_folds
        offset = rng.integers(0, n_folds)
        fold_of_row[idx] = (labels + offset) % n_folds
    for f in range(n_folds):
        test = np.where(fold_of_row == f)[0]
        train = np.where(fold_of_row != f)[0]
        yield train, test


def loso_splits(source_ids):
    """Leave-one-source-out: train on 10 sources, test on the held-out one."""
    source_ids = np.asarray(source_ids)
    for s in np.unique(source_ids):
        test = np.where(source_ids == s)[0]
        train = np.where(source_ids != s)[0]
        yield train, test, s
