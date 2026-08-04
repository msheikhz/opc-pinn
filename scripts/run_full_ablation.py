"""Full ablation grid (Experiments A, B, D, E, F). Requires PyTorch.

NOT EXECUTED in the authoring environment. Run the gate first:
    python3 tests/test_torch_parity.py
then:
    python3 scripts/run_full_ablation.py --seeds 42 43 44 45 46

The grid decomposes the manuscript's frozen -> adapted axis, which currently
bundles three simultaneous changes. Cells:

  id                       chain params  hydration  lambda1  correction
  ------------------------------------------------------------------------
  pure_mlp                 --            --         --       none (6 features)
  mlp7                     --            --         --       none (7 features)   <- isolates the feature
  add_frozen               fixed         Eq.(5)     0.05     additive   (original PINN)
  add_frozen_nolam         fixed         Eq.(5)     0.00     additive   <- isolates losing lambda1
  add_frozen_sf            fixed         Eq.(18)    0.05     additive   <- isolates hydration model
  add_calib_lam            learnable     Eq.(18)    0.05     additive   <- isolates learnable params
  add_calib                learnable     Eq.(18)    0.00     additive   (manuscript's winner)
  coup_frozen              fixed         Eq.(5)     --       coupled
  coup_calib               learnable     Eq.(18)    --       coupled
  coup_calib_col           learnable     Eq.(18)    --       coupled + collocation
  add_calib_col            learnable     Eq.(18)    0.00     additive + collocation  <- collocation where it can bind

Reading it: add_frozen_sf - add_frozen isolates the hydration model;
add_frozen_nolam - add_frozen isolates the loss change; add_calib_lam -
add_frozen_sf isolates the learnable parameters. Their sum should approximately
equal add_calib - add_frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

ROOT = _Path(__file__).resolve().parents[1]

from opc_pinn.bootstrap import paired_cluster_bootstrap
from opc_pinn.cv import loso_splits, source_stratified_kfold
from opc_pinn.data import load_dataset
from opc_pinn.features import Standardiser, build_features
from opc_pinn.losses import (bounds_loss, correction_magnitude_loss, data_loss,
                             monotonicity_collocation_loss,
                             physics_consistency_loss)
from opc_pinn.metrics import aggregate_fold_metrics, all_metrics
from opc_pinn.models import AdditiveModel, CoupledModel, PureMLP

DTYPE = torch.float64

GRID = {
    "pure_mlp":        dict(kind="mlp", n_feat=6),
    "mlp7":            dict(kind="mlp", n_feat=7),
    "add_frozen":      dict(kind="add", hydration="eq5", calibrated=False, lam1=0.05),
    "add_frozen_nolam": dict(kind="add", hydration="eq5", calibrated=False, lam1=0.0),
    "add_frozen_sf":   dict(kind="add", hydration="sf", calibrated=False, lam1=0.05),
    "add_calib_lam":   dict(kind="add", hydration="sf", calibrated=True, lam1=0.05),
    "add_calib":       dict(kind="add", hydration="sf", calibrated=True, lam1=0.0),
    "add_calib_col":   dict(kind="add", hydration="sf", calibrated=True, lam1=0.0, colloc=True),
    "coup_frozen":     dict(kind="coup", hydration="eq5", calibrated=False),
    "coup_calib":      dict(kind="coup", hydration="sf", calibrated=True),
    "coup_calib_col":  dict(kind="coup", hydration="sf", calibrated=True, colloc=True),
}


def make_model(cfg):
    if cfg["kind"] == "mlp":
        return PureMLP(cfg["n_feat"])
    if cfg["kind"] == "add":
        return AdditiveModel(7, hydration=cfg["hydration"], calibrated=cfg["calibrated"])
    return CoupledModel(7, hydration=cfg["hydration"], calibrated=cfg["calibrated"],
                        d_alpha_scale=cfg.get("d_alpha_scale", 0.10),
                        d_X_scale=cfg.get("d_X_scale", 0.05),
                        soft_clip=cfg.get("soft_clip", False))


def sample_collocation(df, n=2000, seed=0):
    """Latin hypercube over the dataset's observed ranges (Eq. 23)."""
    from scipy.stats import qmc
    ox_cols = ["CaO", "SiO2", "Al2O3", "Fe2O3", "SO3"]
    lo = np.r_[df[ox_cols].min().to_numpy(), df["wc"].min(), df["age"].min()]
    hi = np.r_[df[ox_cols].max().to_numpy(), df["wc"].max(), df["age"].max()]
    s = qmc.LatinHypercube(d=7, seed=seed).random(n)
    v = qmc.scale(s, lo, hi)
    return {"ox": torch.tensor(v[:, :5], dtype=DTYPE, requires_grad=True),
            "wc": torch.tensor(v[:, 5], dtype=DTYPE, requires_grad=True),
            "age": torch.tensor(v[:, 6], dtype=DTYPE, requires_grad=True)}


def train_one(model, cfg, tr, te, df, colloc, epochs=1500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    which = "mlp" if cfg.get("n_feat") == 6 else "pinn"
    X_all, _ = build_features(df, hydration=cfg.get("hydration", "eq5"), which=which)
    std = Standardiser().fit(X_all[tr])
    Xt = torch.tensor(std.transform(X_all), dtype=DTYPE)

    ox = torch.tensor(df[["CaO", "SiO2", "Al2O3", "Fe2O3", "SO3"]].to_numpy(float), dtype=DTYPE)
    wc = torch.tensor(df["wc"].to_numpy(float), dtype=DTYPE)
    age = torch.tensor(df["age"].to_numpy(float), dtype=DTYPE)
    y = torch.tensor(df["fc"].to_numpy(float), dtype=DTYPE)

    model = model.to(DTYPE)
    # Two parameter groups. The chain scalars (f0 ~ 234, and n, tau, beta ~ 1)
    # need a much larger step than the network weights: f0 enters as fc = f0*X^n,
    # so its gradient is X^n ~ 0.08, and at the shared lr=1e-3 it moves only ~1
    # unit over the whole run (234.0 -> 233.2 observed) — i.e. it never
    # calibrates, and the network is forced to absorb the scale error, which it
    # can do on training sources but not on a held-out one. A dedicated higher lr
    # lets the chain reach the same optimum the analytical least-squares fit
    # finds (f0 ~ 98, n ~ 1.06 under eq5).
    chain_params, net_params = [], []
    for name, p in model.named_parameters():
        (chain_params if "chain.params" in name else net_params).append(p)
    groups = [{"params": net_params, "lr": 1e-3}]
    if chain_params:
        groups.append({"params": chain_params, "lr": 5e-2})
    opt = torch.optim.Adam(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    def feature_fn(o, w, a):
        # collocation points need features rebuilt inside the graph so that
        # d(pred)/d(age) and d(pred)/d(wc) are non-zero through the features too
        from opc_pinn import physics as P
        f0, n_, tau, beta = model.chain.params.values()
        out = P.chain(o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], w, a, torch,
                      hydration=cfg.get("hydration", "eq5"), f0=f0, n=n_, tau=tau, beta=beta)
        cols = [w, out["C3S"] / 100, out["C2S"] / 100,
                torch.log1p(a) / np.log(91.0), out["X"], out["p_cap"],
                out["fc_phys"] / f0]
        Z = torch.stack(cols, dim=1)
        mu = torch.tensor(std.mu, dtype=DTYPE)
        sd = torch.tensor(std.sd, dtype=DTYPE)
        return (Z - mu) / sd

    best, best_state = np.inf, None
    tr_t = torch.tensor(tr, dtype=torch.long)
    te_t = torch.tensor(te, dtype=torch.long)

    # Checkpoint on a validation signal, not training loss. Selecting the
    # lowest *training* MSE rewards memorising the training sources, which is
    # exactly the LOSO failure mode. Hold out the rows of one training source as
    # an internal validation fold; if the training set has only one source (it
    # never does here, but guard anyway) fall back to training loss.
    src_arr = df["source_id"].to_numpy()[tr]
    uniq = np.unique(src_arr)
    if len(uniq) > 1:
        rng = np.random.default_rng(seed)
        val_src = uniq[rng.integers(len(uniq))]
        val_mask = src_arr == val_src
        fit_t = torch.tensor(tr[~val_mask], dtype=torch.long)
        val_t = torch.tensor(tr[val_mask], dtype=torch.long)
    else:
        fit_t = tr_t
        val_t = tr_t

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred, aux = model(Xt[fit_t], ox[fit_t], wc[fit_t], age[fit_t])
        loss = data_loss(pred, y[fit_t])

        if cfg["kind"] == "add":
            if cfg.get("lam1", 0.0) > 0:
                loss = loss + cfg["lam1"] * physics_consistency_loss(pred, aux["fc_phys"])
            loss = loss + 0.01 * bounds_loss(pred)
        elif cfg["kind"] == "coup":
            loss = loss + correction_magnitude_loss(
                aux["d_alpha"], aux["d_X"], cfg.get("lam_a", 0.05), cfg.get("lam_X", 0.05))

        if cfg.get("colloc"):
            loss = loss + cfg.get("lam_mono", 0.1) * monotonicity_collocation_loss(
                model, colloc, feature_fn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vp, _ = model(Xt[val_t], ox[val_t], wc[val_t], age[val_t])
                v = float(torch.mean((vp - y[val_t]) ** 2))
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p, _ = model(Xt[te_t], ox[te_t], wc[te_t], age[te_t])

    fitted = {}
    if hasattr(model, "chain") and model.chain.params.calibrated:
        f0, n_, tau, beta = model.chain.params.values()
        fitted = dict(f0=float(f0), n=float(n_), tau=float(tau), beta=float(beta))
    return p.numpy(), fitted


def run_cell(name, cfg, df, seeds, epochs):
    y, src = df["fc"].to_numpy(float), df["source_id"].to_numpy()
    colloc = sample_collocation(df) if cfg.get("colloc") else None
    per_seed = []
    for seed in seeds:
        loso_pred = np.full(len(df), np.nan)
        fitted = []
        for tr, te, s in loso_splits(src):
            p, fp = train_one(make_model(cfg), cfg, tr, te, df, colloc, epochs, seed)
            loso_pred[te] = p
            if fp:
                fitted.append(fp)
        kf_pred = np.full(len(df), np.nan)
        for tr, te in source_stratified_kfold(src, 5, seed=seed):
            p, _ = train_one(make_model(cfg), cfg, tr, te, df, colloc, epochs, seed)
            kf_pred[te] = p
        per_seed.append({"seed": seed, "loso_pred": loso_pred, "kf_pred": kf_pred,
                         "fitted": fitted})
        print(f"    seed {seed}: LOSO RMSE {all_metrics(y, loso_pred)['rmse']:.3f}")

    loso_rmse = [all_metrics(y, s["loso_pred"])["rmse"] for s in per_seed]
    return {"name": name, "per_seed": per_seed,
            "loso_pooled_mean": float(np.mean(loso_rmse)),
            "loso_pooled_std": float(np.std(loso_rmse, ddof=0)),
            "loso_pooled_seed0": all_metrics(y, per_seed[0]["loso_pred"]),
            "kfold_pooled_seed0": all_metrics(y, per_seed[0]["kf_pred"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--cells", nargs="+", default=list(GRID))
    args = ap.parse_args()

    df = load_dataset()
    y, src = df["fc"].to_numpy(float), df["source_id"].to_numpy()
    out = {}
    for name in args.cells:
        print(f"\n=== {name} ===")
        out[name] = run_cell(name, GRID[name], df, args.seeds, args.epochs)

    print("\n" + "=" * 74)
    print(f"{'cell':<20}{'LOSO RMSE (mean +/- std over seeds)':>40}")
    print("=" * 74)
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["loso_pooled_mean"]):
        print(f"{k:<20}{v['loso_pooled_mean']:>28.3f} +/- {v['loso_pooled_std']:.3f}")

    if {"add_frozen", "add_frozen_sf", "add_frozen_nolam", "add_calib_lam",
        "add_calib"} <= set(out):
        b = out["add_frozen"]["loso_pooled_mean"]
        print("\nDECOMPOSITION of frozen -> calibrated (MPa, negative = better):")
        print(f"  hydration model alone   {out['add_frozen_sf']['loso_pooled_mean'] - b:+7.3f}")
        print(f"  losing lambda1 alone    {out['add_frozen_nolam']['loso_pooled_mean'] - b:+7.3f}")
        print(f"  learnable params alone  {out['add_calib_lam']['loso_pooled_mean'] - out['add_frozen_sf']['loso_pooled_mean']:+7.3f}")
        print(f"  all three together      {out['add_calib']['loso_pooled_mean'] - b:+7.3f}")

    for k, v in out.items():
        fitted = [f for s in v["per_seed"] for f in s["fitted"]]
        if fitted:
            fp = pd.DataFrame(fitted)
            print(f"\nfitted parameters, {k}:")
            for c in fp.columns:
                print(f"  {c:<6} {fp[c].mean():10.3f} +/- {fp[c].std(ddof=0):8.3f}"
                      f"   range [{fp[c].min():.3f}, {fp[c].max():.3f}]")

    if {"add_calib", "pure_mlp"} <= set(out):
        print("\nPAIRED CLUSTER BOOTSTRAP (seed-0 predictions):")
        for a, b in [("add_calib", "pure_mlp"), ("add_calib", "add_frozen"),
                     ("coup_calib_col", "pure_mlp")]:
            if a in out and b in out:
                bs = paired_cluster_bootstrap(y, out[a]["per_seed"][0]["loso_pred"],
                                              out[b]["per_seed"][0]["loso_pred"],
                                              src, 10000, 42)
                print(f"  {a} vs {b}: diff {bs['point_diff']:+.3f} "
                      f"CI [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}] "
                      f"frac {bs['frac_favoring_a']:.3f}")

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in out.items()}
    (res / "full_ablation.json").write_text(json.dumps(slim, indent=2, default=float))
    np.savez(res / "full_ablation_predictions.npz", y=y, source_id=src,
             **{f"{k}_seed{s['seed']}": s["loso_pred"]
                for k, v in out.items() for s in v["per_seed"]})
    print(f"\nwrote {res/'full_ablation.json'}")


if __name__ == "__main__":
    main()
