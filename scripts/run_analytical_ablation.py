"""Analytical-only ablation (Experiment C + analytical decomposition).

Runs without PyTorch. Answers two questions the manuscript currently leaves open:

  1. How much of the hybrid model's accuracy comes from four fitted scalars,
     with no neural network at all? (Experiment C of the revision package.)
  2. Within the frozen -> adapted axis, how much is the hydration model change
     and how much is parameter fitting? (Analytical version of Experiment A.)

All parameters are fitted on training folds only; test folds are never touched
during fitting, so these numbers are honest out-of-sample estimates.

Usage:  python3 scripts/run_analytical_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- make imports work regardless of working directory or layout ---
import sys as _sys
from pathlib import Path as _Path

def _add_pkg_to_path():
    """Locate the opc_pinn package and put its parent on sys.path.

    Handles: running from the repo root, from tests/, from scripts/, or with all
    files flattened into one folder alongside an opc_pinn/ directory.
    """
    here = _Path(__file__).resolve().parent
    for base in (here, *here.parents):
        for cand in (base / "src", base, base / "opc_pinn" / "..",):
            cand = cand.resolve()
            if (cand / "opc_pinn" / "physics.py").is_file():
                if str(cand) not in _sys.path:
                    _sys.path.insert(0, str(cand))
                return cand
    raise ImportError(
        "Could not find the opc_pinn package.\n"
        "Expected one of:\n"
        "    <root>/src/opc_pinn/physics.py   (repo layout)\n"
        "    <root>/opc_pinn/physics.py       (flat layout)\n"
        f"Searched upward from: {here}\n"
        "Fix: keep the folder structure from the zip, and run from the project "
        "root, e.g.\n"
        "    cd D:\\KFUPM\\Adil\\Self-Verifying\n"
        "    python tests\\test_torch_parity.py"
    )

_add_pkg_to_path()
# --- end path fix ---


from opc_pinn.analytical import CONFIGS, fit_analytical, predict_analytical
from opc_pinn.bootstrap import paired_cluster_bootstrap
from opc_pinn.cv import loso_splits, source_stratified_kfold
from opc_pinn.data import load_dataset
from opc_pinn.metrics import aggregate_fold_metrics, all_metrics

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def run_config(df, name, cfg, seed=42):
    y = df["fc"].to_numpy(float)
    src = df["source_id"].to_numpy()

    # ---- LOSO ----
    loso_pred = np.full(len(df), np.nan)
    loso_folds, fitted_per_fold = [], []
    for train_idx, test_idx, s in loso_splits(src):
        params = fit_analytical(df.iloc[train_idx], hydration=cfg["hydration"],
                                fit_params=cfg["fit_params"])
        p = predict_analytical(df.iloc[test_idx], params, hydration=cfg["hydration"])
        loso_pred[test_idx] = p
        loso_folds.append(all_metrics(y[test_idx], p))
        fitted_per_fold.append({k: v for k, v in params.items()
                                if not k.startswith("_")})

    # ---- source-stratified 5-fold ----
    kf_pred = np.full(len(df), np.nan)
    kf_folds = []
    for train_idx, test_idx in source_stratified_kfold(src, n_folds=5, seed=seed):
        params = fit_analytical(df.iloc[train_idx], hydration=cfg["hydration"],
                                fit_params=cfg["fit_params"])
        p = predict_analytical(df.iloc[test_idx], params, hydration=cfg["hydration"])
        kf_pred[test_idx] = p
        kf_folds.append(all_metrics(y[test_idx], p))

    return {
        "name": name,
        "loso_pooled": all_metrics(y, loso_pred),
        "loso_unweighted": aggregate_fold_metrics(loso_folds),
        "kfold_unweighted": aggregate_fold_metrics(kf_folds),
        "kfold_pooled": all_metrics(y, kf_pred),
        "fitted_params": fitted_per_fold,
        "loso_pred": loso_pred,
        "per_source_loso": [
            {"source": int(s), "n": int((src == s).sum()),
             **all_metrics(y[src == s], loso_pred[src == s])}
            for s in np.unique(src)],
    }


def main():
    df = load_dataset()
    y = df["fc"].to_numpy(float)
    src = df["source_id"].to_numpy()
    print(f"dataset: {len(df)} rows, {df.source_id.nunique()} sources, "
          f"sd(y) = {y.std(ddof=0):.2f} MPa\n")

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"running {name} ...")
        results[name] = run_config(df, name, cfg)

    # ---------------- pooled LOSO table ----------------
    print("\n" + "=" * 78)
    print("ANALYTICAL-ONLY ABLATION — pooled LOSO (no neural network anywhere)")
    print("=" * 78)
    print(f"{'configuration':<34}{'RMSE':>9}{'MAE':>9}{'R2':>9}   what it isolates")
    notes = {
        "frozen_eq5": "manuscript 'Powers model'",
        "fitted_eq5": "+ fitted f0, n",
        "frozen_sf":  "+ new hydration model only",
        "fitted_sf":  "+ both (adapted chain)",
    }
    for k in CONFIGS:
        m = results[k]["loso_pooled"]
        print(f"{k:<34}{m['rmse']:9.2f}{m['mae']:9.2f}{m['r2']:9.3f}   {notes[k]}")

    print("\n" + "-" * 78)
    print("5-fold CV (unweighted mean +/- std across folds)")
    print("-" * 78)
    for k in CONFIGS:
        m = results[k]["kfold_unweighted"]
        print(f"{k:<34}{m['rmse_mean']:9.2f} +/- {m['rmse_std']:.2f}"
              f"   R2 {m['r2_mean']:.3f} +/- {m['r2_std']:.3f}")

    # ---------------- decomposition ----------------
    r = {k: results[k]["loso_pooled"]["rmse"] for k in CONFIGS}
    print("\n" + "=" * 78)
    print("DECOMPOSITION of the frozen -> adapted change (pooled LOSO RMSE, MPa)")
    print("=" * 78)
    print(f"  baseline (frozen chain, Eq. 5)          {r['frozen_eq5']:8.2f}")
    print(f"  effect of fitting f0, n alone           {r['fitted_eq5'] - r['frozen_eq5']:+8.2f}")
    print(f"  effect of new hydration model alone     {r['frozen_sf'] - r['frozen_eq5']:+8.2f}")
    print(f"  effect of both together                 {r['fitted_sf'] - r['frozen_eq5']:+8.2f}")
    print(f"  adapted chain, analytical only          {r['fitted_sf']:8.2f}")
    print("  (negative = improvement)")

    # ---------------- fitted parameters ----------------
    print("\n" + "=" * 78)
    print("FITTED PARAMETERS across the 11 LOSO folds (Experiment B)")
    print("=" * 78)
    for k in ("fitted_eq5", "fitted_sf"):
        print(f"\n{k}:")
        fp = pd.DataFrame(results[k]["fitted_params"])
        for p in CONFIGS[k]["fit_params"]:
            v = fp[p].to_numpy(float)
            init = {"f0": 234.0, "n": 3.0, "tau": 1.0, "beta": 0.7}[p]
            print(f"  {p:<6} init {init:>7.2f}   fitted {v.mean():9.3f} +/- {v.std(ddof=0):7.3f}"
                  f"   range [{v.min():.3f}, {v.max():.3f}]")

    # ---------------- bootstrap ----------------
    print("\n" + "=" * 78)
    print("PAIRED CLUSTER BOOTSTRAP (10,000 resamples of whole sources, seed 42)")
    print("=" * 78)
    comparisons = [
        ("fitted_sf", "frozen_eq5"), ("fitted_sf", "fitted_eq5"),
        ("fitted_sf", "frozen_sf"), ("fitted_eq5", "frozen_eq5"),
        ("frozen_sf", "frozen_eq5"),
    ]
    boot_out = {}
    for a, b in comparisons:
        bs = paired_cluster_bootstrap(y, results[a]["loso_pred"], results[b]["loso_pred"],
                                      src, n_boot=10000, seed=42)
        boot_out[f"{a}_vs_{b}"] = bs
        verdict = ("A better" if bs["ci_high"] < 0 else
                   "B better" if bs["ci_low"] > 0 else "indistinguishable")
        print(f"  {a} vs {b}")
        print(f"    diff {bs['point_diff']:+7.3f}   95% CI [{bs['ci_low']:+.3f}, "
              f"{bs['ci_high']:+.3f}]   frac favouring A {bs['frac_favoring_a']:.3f}   -> {verdict}")

    # ---------------- per-source LOSO ----------------
    print("\n" + "=" * 78)
    print("PER-SOURCE LOSO RMSE (Experiment F)")
    print("=" * 78)
    hdr = f"{'source':>7}{'n':>5}" + "".join(f"{k:>13}" for k in CONFIGS)
    print(hdr)
    for i, s in enumerate(np.unique(src)):
        row = f"{int(s):>7}{int((src == s).sum()):>5}"
        for k in CONFIGS:
            row += f"{results[k]['per_source_loso'][i]['rmse']:13.2f}"
        print(row)

    # ---------------- persist ----------------
    serialisable = {
        k: {kk: vv for kk, vv in v.items() if kk != "loso_pred"}
        for k, v in results.items()}
    serialisable["_bootstrap"] = boot_out
    (OUT / "analytical_ablation.json").write_text(json.dumps(serialisable, indent=2, default=float))
    np.savez(OUT / "analytical_loso_predictions.npz",
             y=y, source_id=src, **{k: results[k]["loso_pred"] for k in CONFIGS})
    print(f"\nwrote {OUT/'analytical_ablation.json'}")


if __name__ == "__main__":
    main()
