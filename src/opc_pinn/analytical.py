"""Analytical-only baselines, with and without fitted chain parameters.

This is Experiment C of the revision package: how much of the hybrid model's
accuracy comes from four fitted scalars, and how much from the neural network?
It requires no neural network, so it runs without PyTorch.

It also decomposes, at the analytical level, the two changes that the
manuscript's frozen -> adapted axis bundles together:

    config            hydration model     f0, n        isolates
    ---------------------------------------------------------------------
    frozen_eq5        Eq. (5)             fixed        the manuscript's "Powers' model"
    fitted_eq5        Eq. (5)             fitted       parameter fitting alone
    frozen_sf         Eqs. (18)-(19)      fixed        hydration model alone
    fitted_sf         Eqs. (18)-(19)      fitted       both (the adapted chain)

Comparing fitted_eq5 - frozen_eq5 against frozen_sf - frozen_eq5 separates the
two effects in a setting where no network can absorb either.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import physics as P

OXIDES = ("CaO", "SiO2", "Al2O3", "Fe2O3", "SO3")


def _predict(df, hydration, f0, n, tau, beta, unhydrated_basis):
    out = P.chain(
        df["CaO"].to_numpy(float), df["SiO2"].to_numpy(float),
        df["Al2O3"].to_numpy(float), df["Fe2O3"].to_numpy(float),
        df["SO3"].to_numpy(float), df["wc"].to_numpy(float),
        df["age"].to_numpy(float), xp=np,
        hydration=hydration, f0=f0, n=n, tau=tau, beta=beta,
        unhydrated_basis=unhydrated_basis)
    return out["fc_phys"]


# Parameters are fitted in log space so they stay strictly positive, which is
# the same role the softplus reparameterisation plays in the torch model.
_LOG_BOUNDS = {
    "f0":   (np.log(10.0),   np.log(2000.0)),
    "n":    (np.log(0.5),    np.log(10.0)),
    "tau":  (np.log(1e-3),   np.log(500.0)),
    "beta": (np.log(0.05),   np.log(5.0)),
}


def fit_analytical(train_df, hydration="sf", fit_params=("f0", "n", "tau", "beta"),
                   unhydrated_basis="reacted_silicates",
                   init=None, verbose=False):
    """Least-squares fit of chain parameters on a training fold.

    Returns a dict of fitted parameters. Parameters not in ``fit_params`` are
    held at their literature / initial values.
    """
    _d = dict(f0=P.F0_LIT, n=P.N_LIT, tau=P.TAU_INIT, beta=P.BETA_INIT)
    _d.update(init or {})
    init = _d
    fit_params = tuple(fit_params)
    y = train_df["fc"].to_numpy(float)

    def unpack(theta_log):
        vals = dict(init)
        for name, v in zip(fit_params, theta_log):
            vals[name] = float(np.exp(v))
        return vals

    def resid(theta_log):
        v = unpack(theta_log)
        pred = _predict(train_df, hydration, v["f0"], v["n"], v["tau"], v["beta"],
                        unhydrated_basis)
        pred = np.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6)
        return pred - y

    x0 = np.array([np.log(init[p]) for p in fit_params])
    lo = np.array([_LOG_BOUNDS[p][0] for p in fit_params])
    hi = np.array([_LOG_BOUNDS[p][1] for p in fit_params])

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        max_nfev=20000, xtol=1e-12, ftol=1e-12)
    fitted = unpack(sol.x)
    fitted["_success"] = bool(sol.success)
    fitted["_cost"] = float(sol.cost)
    if verbose:
        print("  fitted:", {k: round(v, 4) for k, v in fitted.items()
                            if not k.startswith("_")})
    return fitted


def predict_analytical(df, params, hydration="sf",
                       unhydrated_basis="reacted_silicates"):
    return _predict(df, hydration, params["f0"], params["n"],
                    params["tau"], params["beta"], unhydrated_basis)


CONFIGS = {
    "frozen_eq5": dict(hydration="eq5", fit_params=()),
    "fitted_eq5": dict(hydration="eq5", fit_params=("f0", "n")),
    "frozen_sf":  dict(hydration="sf",  fit_params=()),
    "fitted_sf":  dict(hydration="sf",  fit_params=("f0", "n", "tau", "beta")),
}
