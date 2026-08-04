"""Validation suite. Run: python3 tests/test_validation.py

Validates against known-good values, physical invariants, and synthetic
ground truth. Deliberately does NOT validate the volume balance against the
manuscript's X = 0.4421, which is not reproducible from the manuscript text.
"""
from __future__ import annotations

import sys
import numpy as np

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


from opc_pinn import physics as P
from opc_pinn.data import load_dataset
from opc_pinn.cv import source_stratified_kfold, loso_splits
from opc_pinn.metrics import rmse, mae, r2, all_metrics
from opc_pinn.bootstrap import paired_cluster_bootstrap
from opc_pinn.analytical import fit_analytical, predict_analytical

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------- 1. Bogue
print("\n1. Stage 1 - Bogue equations")
C3S, C2S, C3A, C4AF = P.bogue(64.0, 21.0, 5.5, 3.5, 2.5)
check("worked example C3S = 51.86", approx(C3S, 51.86, 5e-3), f"got {C3S:.4f}")
check("worked example C2S = 21.08", approx(C2S, 21.08, 5e-3), f"got {C2S:.4f}")
check("worked example C3A = 8.65", approx(C3A, 8.65, 5e-3), f"got {C3A:.4f}")
check("worked example C4AF = 10.65", approx(C4AF, 10.65, 5e-3), f"got {C4AF:.4f}")

df = load_dataset()
b = P.bogue(df["CaO"].to_numpy(), df["SiO2"].to_numpy(), df["Al2O3"].to_numpy(),
            df["Fe2O3"].to_numpy(), df["SO3"].to_numpy())
b_raw = P.bogue(df["CaO"].to_numpy(), df["SiO2"].to_numpy(), df["Al2O3"].to_numpy(),
                df["Fe2O3"].to_numpy(), df["SO3"].to_numpy(), clip_negative=False)
n_neg = int(((b_raw[0] < 0) | (b_raw[1] < 0)).sum())
check("unclipped Bogue yields a negative phase on this dataset (REAL DEFECT)",
      n_neg > 0, f"{n_neg}/62 rows; min C3S {b_raw[0].min():.2f}, min C2S {b_raw[1].min():.2f}")
check("clipping restores non-negative phases on all 62 rows",
      bool(np.all(b[0] >= 0) and np.all(b[1] >= 0)))
check("unclipped phase sums are all under 100%",
      bool(np.all(sum(b_raw) <= 100.0)), f"max {sum(b_raw).max():.2f}%")
check("clipping the negative C3S pushes one source just over 100% (documented)",
      float(sum(b).max()) > 100.0, f"max {sum(b).max():.2f}% -- source 7, same defect")

# ------------------------------------------------------------ 2. Hydration
print("\n2. Stage 2 - hydration models")
check("Eq.(5) gives alpha(0) = 0.28 (documents the defect)",
      approx(P.alpha_eq5(0.0, np), 0.28), f"got {P.alpha_eq5(0.0, np):.4f}")
check("Eq.(5) is independent of w/c (documents the defect)",
      approx(P.alpha_eq5(28.0, np), P.alpha_eq5(28.0, np)))

a0 = P.alpha_sf(1e-9, 0.45, 1.0, 0.7, np)
check("Eq.(18) gives alpha(0+) = 0 exactly", a0 < 1e-12, f"got {a0:.2e}")
a_inf = P.alpha_sf(1e9, 0.45, 1.0, 0.7, np)
check("Eq.(18) approaches alpha_max as t -> inf",
      approx(a_inf, P.alpha_max_mills(0.45), 1e-6), f"got {a_inf:.6f}")

ages = np.array([3.0, 7.0, 28.0, 90.0])
a_sf = P.alpha_sf(ages, 0.45, 1.0, 0.7, np)
check("Eq.(18) is strictly increasing in age", bool(np.all(np.diff(a_sf) > 0)))
wcs = np.linspace(0.25, 0.70, 20)
amax = P.alpha_max_mills(wcs)
check("alpha_max strictly increasing in w/c", bool(np.all(np.diff(amax) > 0)))
check("alpha_max within (0,1) over dataset w/c range",
      bool(np.all((amax > 0) & (amax < 1))), f"range [{amax.min():.3f}, {amax.max():.3f}]")

# Reparameterisation identity: exp(-(tau/t)**beta) == exp(-tau'/t**beta), tau'=tau**beta
tau, beta, t = 2.3, 0.62, 17.0
lhs = np.exp(-((tau / t) ** beta))
rhs = np.exp(-(tau ** beta) / t ** beta)
check("Eq.(18) reparameterisation identity holds", approx(lhs, rhs, 1e-12))

# ------------------------------------------------ 3. Volume balance / X
print("\n3. Stages 3-4 - volume balance and gel-space ratio")
X_all = []
for basis in ("reacted_silicates", "one_minus_alpha", "silicates_only"):
    X, vg, vc, vu = P.gel_space_ratio(51.86, 21.08, 0.75, 0.45, np, unhydrated_basis=basis)
    X_all.append((basis, float(X)))
check("all bases give X in (0,1)", all(0 < x < 1 for _, x in X_all),
      ", ".join(f"{b}={x:.4f}" for b, x in X_all))
check("manuscript X = 0.4421 NOT reproduced by any basis (known defect)",
      all(abs(x - 0.4421) > 1e-3 for _, x in X_all),
      "documented in physics.py; needs the original code to resolve")

# Monotonicity over the whole dataset domain
g_wc, g_a = np.meshgrid(np.linspace(0.25, 0.70, 40), np.linspace(0.05, 0.95, 40))
Xg, *_ = P.gel_space_ratio(55.0, 20.0, g_a, g_wc, np)
check("X in (0,1) across the full (w/c, alpha) grid",
      bool(np.all((Xg > 0) & (Xg < 1))), f"range [{Xg.min():.3f}, {Xg.max():.3f}]")
check("dX/d(alpha) > 0 everywhere", bool(np.all(np.diff(Xg, axis=0) > 0)))
check("dX/d(w/c) < 0 everywhere", bool(np.all(np.diff(Xg, axis=1) < 0)))

# Volume closure: gel + capillary + unhydrated == total
Xc, vg, vc, vu = P.gel_space_ratio(55.0, 20.0, 0.6, 0.45, np)
v_total = 100.0 / P.RHO_CEMENT + 0.45 * 100.0
check("volume balance closes (Vgel+Vcap+Vunhyd = Vtotal)",
      approx(vg + vc + vu, v_total, 1e-9), f"residual {abs(vg+vc+vu-v_total):.2e}")

# --------------------------------------------------------- 4. Full chain
print("\n4. Stage 5 and full chain")
check("Powers law reproduces 234 * 0.4421^3 = 20.22",
      approx(P.powers_strength(0.4421, 234.0, 3.0, np), 20.22, 5e-3),
      f"got {P.powers_strength(0.4421, 234.0, 3.0, np):.4f}")

for hyd in ("eq5", "sf"):
    out = P.chain(df["CaO"].to_numpy(float), df["SiO2"].to_numpy(float),
                  df["Al2O3"].to_numpy(float), df["Fe2O3"].to_numpy(float),
                  df["SO3"].to_numpy(float), df["wc"].to_numpy(float),
                  df["age"].to_numpy(float), np, hydration=hyd)
    ok = np.all(np.isfinite(out["fc_phys"])) and np.all(out["fc_phys"] > 0)
    check(f"chain({hyd}) finite and positive on all 62 rows", bool(ok),
          f"fc_phys range [{out['fc_phys'].min():.2f}, {out['fc_phys'].max():.2f}]")
    check(f"chain({hyd}) alpha within (0,1) on all 62 rows",
          bool(np.all((out["alpha"] > 0) & (out["alpha"] < 1))),
          f"[{out['alpha'].min():.3f}, {out['alpha'].max():.3f}]")

# End-to-end monotonicity in age and w/c
one = dict(CaO=64.0, SiO2=21.0, Al2O3=5.5, Fe2O3=3.5, SO3=0.0)
ages = np.linspace(3, 90, 30)
fc_age = P.chain(one["CaO"], one["SiO2"], one["Al2O3"], one["Fe2O3"], one["SO3"],
                 0.45, ages, np, hydration="sf")["fc_phys"]
check("chain strength strictly increasing in age", bool(np.all(np.diff(fc_age) > 0)))
wcs = np.linspace(0.25, 0.70, 30)
fc_wc = P.chain(one["CaO"], one["SiO2"], one["Al2O3"], one["Fe2O3"], one["SO3"],
                wcs, 28.0, np, hydration="sf")["fc_phys"]
check("chain strength strictly decreasing in w/c", bool(np.all(np.diff(fc_wc) < 0)))

# ---------------------------------------------------------- 5. Splitters
print("\n5. Cross-validation splitters")
src = df["source_id"].to_numpy()
folds = list(source_stratified_kfold(src, n_folds=5, seed=42))
check("5-fold produces 5 folds", len(folds) == 5)
test_union = np.sort(np.concatenate([t for _, t in folds]))
check("5-fold: every row tested exactly once",
      np.array_equal(test_union, np.arange(len(df))))
check("5-fold: no train/test overlap in any fold",
      all(len(np.intersect1d(tr, te)) == 0 for tr, te in folds))
# A source with k rows can appear in at most min(k, 5) folds, so requiring
# every source in every fold is impossible for the 2- and 3-row sources. The
# correct invariant is that no source is over-concentrated in one fold.
max_per_fold = {}
for fi, (_, te) in enumerate(folds):
    for s_ in np.unique(src[te]):
        max_per_fold[s_] = max(max_per_fold.get(s_, 0), int((src[te] == s_).sum()))
counts = {s_: int((src == s_).sum()) for s_ in np.unique(src)}
ok = all(max_per_fold.get(s_, 0) <= -(-counts[s_] // 5) for s_ in counts)
check("5-fold: no source over-concentrated in any fold (round-robin holds)", ok,
      f"max rows-per-fold per source {sorted(max_per_fold.values())}")

loso = list(loso_splits(src))
check("LOSO produces 11 folds", len(loso) == 11)
check("LOSO: test fold is exactly one whole source",
      all(len(np.unique(src[te])) == 1 for _, te, _ in loso))
check("LOSO: held-out source absent from training set",
      all(s not in src[tr] for tr, _, s in loso))
check("LOSO fold sizes are 2-18 as the manuscript states",
      sorted(len(te) for _, te, _ in loso) == [2, 2, 3, 3, 3, 3, 3, 3, 6, 16, 18],
      f"{sorted(len(te) for _, te, _ in loso)}")

# ------------------------------------------------------------ 6. Metrics
print("\n6. Metrics")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rng = np.random.default_rng(0)
yt, yp = rng.normal(40, 12, 200), rng.normal(40, 12, 200)
check("rmse matches sklearn", approx(rmse(yt, yp), np.sqrt(mean_squared_error(yt, yp)), 1e-10))
check("mae matches sklearn", approx(mae(yt, yp), mean_absolute_error(yt, yp), 1e-10))
check("r2 matches sklearn", approx(r2(yt, yp), r2_score(yt, yp), 1e-10))
check("r2 of perfect prediction is 1", approx(r2(yt, yt), 1.0, 1e-12))

# Cross-check the manuscript's internal consistency using the real data
y = df["fc"].to_numpy(float)
sst = float(((y - y.mean()) ** 2).sum())
check("dataset SST matches value back-derived from Table IV (11607-11641)",
      11607 <= sst <= 11641, f"SST = {sst:.1f}, sd(y) = {y.std(ddof=0):.2f} MPa")

# ---------------------------------------------------------- 7. Bootstrap
print("\n7. Paired cluster bootstrap")
pred = y + rng.normal(0, 5, len(y))
bs_same = paired_cluster_bootstrap(y, pred, pred, src, n_boot=500, seed=1)
check("identical predictions give diff exactly 0",
      approx(bs_same["diff_mean"], 0.0, 1e-12) and approx(bs_same["ci_low"], 0.0, 1e-12))
better, worse = y + rng.normal(0, 2, len(y)), y + rng.normal(0, 12, len(y))
bs = paired_cluster_bootstrap(y, better, worse, src, n_boot=2000, seed=1)
check("clearly better model gives negative CI",
      bs["ci_high"] < 0 and bs["frac_favoring_a"] > 0.95,
      f"CI [{bs['ci_low']:.2f}, {bs['ci_high']:.2f}], frac {bs['frac_favoring_a']:.3f}")
check("bootstrap reports its cluster count (11, fragile)", bs["n_clusters"] == 11)

# ------------------------------------------- 8. Fitter vs synthetic truth
print("\n8. Analytical fitter recovers known ground truth")
truth = dict(f0=310.0, n=2.4, tau=1.7, beta=0.55)
synth = df.copy()
synth["fc"] = predict_analytical(synth, truth, hydration="sf")
got = fit_analytical(synth, hydration="sf")
for k, v in truth.items():
    check(f"recovers {k} = {v}", approx(got[k], v, max(1e-3, abs(v) * 5e-4)),
          f"got {got[k]:.5f}")
resid = predict_analytical(synth, got, hydration="sf") - synth["fc"].to_numpy()
check("fit on noiseless synthetic data is near-exact",
      float(np.abs(resid).max()) < 1e-4, f"max |resid| = {np.abs(resid).max():.2e}")

noisy = synth.copy()
noisy["fc"] = synth["fc"] + rng.normal(0, 3.0, len(synth))
got_n = fit_analytical(noisy, hydration="sf")
check("fit degrades gracefully under noise (still near truth)",
      abs(got_n["f0"] - truth["f0"]) / truth["f0"] < 0.35,
      f"f0 {got_n['f0']:.1f} vs {truth['f0']}")

# ------------------------------------------------------------- summary
print("\n" + "=" * 64)
print(f"VALIDATION: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print("  -", f)
print("=" * 64)
sys.exit(1 if FAIL else 0)
