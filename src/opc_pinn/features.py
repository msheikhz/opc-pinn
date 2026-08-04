"""Feature engineering (Section III-C). Standardisation uses training-fold
statistics only, to prevent leakage."""
from __future__ import annotations
import numpy as np
from . import physics as P

PINN_FEATURES = ["wc", "C3S_100", "C2S_100", "log_age", "X", "p_cap", "fc_phys_f0"]
MLP_FEATURES = PINN_FEATURES[:-1]   # 6 raw features, no physics prior


def build_features(df, hydration="eq5", f0=P.F0_LIT, n=P.N_LIT,
                   tau=P.TAU_INIT, beta=P.BETA_INIT, which="pinn"):
    out = P.chain(df["CaO"].to_numpy(float), df["SiO2"].to_numpy(float),
                  df["Al2O3"].to_numpy(float), df["Fe2O3"].to_numpy(float),
                  df["SO3"].to_numpy(float), df["wc"].to_numpy(float),
                  df["age"].to_numpy(float), np,
                  hydration=hydration, f0=f0, n=n, tau=tau, beta=beta)
    cols = {
        "wc": df["wc"].to_numpy(float),
        "C3S_100": out["C3S"] / 100.0,
        "C2S_100": out["C2S"] / 100.0,
        "log_age": np.log1p(df["age"].to_numpy(float)) / np.log(91.0),
        "X": out["X"],
        "p_cap": out["p_cap"],
        "fc_phys_f0": out["fc_phys"] / f0,
    }
    names = PINN_FEATURES if which == "pinn" else MLP_FEATURES
    return np.column_stack([cols[c] for c in names]), names


class Standardiser:
    """Fit on the training fold only; apply to both folds."""

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0, ddof=0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd
