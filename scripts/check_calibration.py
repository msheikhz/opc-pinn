"""Fast check that the chain now actually calibrates. Trains ONE LOSO fold and
prints the fitted parameters. Before the fix, f0 stayed ~233; after, it should
move substantially toward the analytical optimum (~98 under eq5-style fitting,
though with the network attached and SF hydration the exact value differs).

    python scripts/check_calibration.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from opc_pinn.data import load_dataset
from opc_pinn.cv import loso_splits
from opc_pinn.metrics import rmse
from run_full_ablation import GRID, make_model, train_one

df = load_dataset(); src = df["source_id"].to_numpy(); y = df["fc"].to_numpy(float)
tr, te, s = next(loso_splits(src))
print("Training add_calib on one LOSO fold (held-out source", int(s), ")...")
p, fp = train_one(make_model(GRID["add_calib"]), GRID["add_calib"], tr, te, df, None,
                  epochs=1500, seed=42)
print("fitted:", {k: round(v, 3) for k, v in fp.items()})
print("test RMSE this fold:", round(rmse(y[te], p), 3))
print()
print("BEFORE fix: f0~233.2, n~2.71  (chain barely moved)")
print("AFTER fix : f0 should have moved substantially; if it is still ~233,")
print("            the two-LR change did not take effect (stale file?).")
