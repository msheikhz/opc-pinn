"""Per-fold LOSO diagnostic for one configuration.

Answers: is a high pooled RMSE caused by a couple of exploding folds (an
out-of-distribution chemistry problem) or by uniformly mediocre training (a
real training problem)? Run:

    python scripts/diagnose_folds.py add_calib
    python scripts/diagnose_folds.py add_frozen
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from opc_pinn.data import load_dataset
from opc_pinn.cv import loso_splits
from opc_pinn.metrics import rmse
from run_full_ablation import GRID, make_model, sample_collocation, train_one

cell = sys.argv[1] if len(sys.argv) > 1 else "add_calib"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
cfg = GRID[cell]

df = load_dataset()
y = df["fc"].to_numpy(float)
src = df["source_id"].to_numpy()
colloc = sample_collocation(df) if cfg.get("colloc") else None

print(f"\n=== {cell}, seed {seed} — per-fold LOSO ===")
print(f"{'source':>7}{'n':>4}{'RMSE':>9}{'fitted (f0, n, tau, beta)':>40}")
pred = np.full(len(df), np.nan)
per_fold = []
for tr, te, s in loso_splits(src):
    p, fp = train_one(make_model(cfg), cfg, tr, te, df, colloc, epochs=1500, seed=seed)
    pred[te] = p
    r = rmse(y[te], p)
    per_fold.append((int(s), len(te), r))
    fps = (f"f0={fp['f0']:.1f} n={fp['n']:.2f} tau={fp['tau']:.2f} beta={fp['beta']:.2f}"
           if fp else "(frozen)")
    print(f"{int(s):>7}{len(te):>4}{r:>9.2f}   {fps}")

pooled = rmse(y, pred)
unweighted = np.mean([r for _, _, r in per_fold])
print(f"\npooled RMSE      {pooled:8.3f}")
print(f"unweighted mean  {unweighted:8.3f}")
worst = sorted(per_fold, key=lambda t: -t[2])[:2]
print(f"worst 2 folds    sources {[w[0] for w in worst]} at RMSE {[round(w[2],1) for w in worst]}")
keep = [t for t in per_fold if t[0] not in {w[0] for w in worst}]
idx = np.concatenate([np.where(src == s)[0] for s, _, _ in keep])
print(f"pooled excl. worst-2  {rmse(y[idx], pred[idx]):8.3f}")
