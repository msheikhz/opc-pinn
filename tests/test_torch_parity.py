"""Parity + gradient checks for the torch layer. Requires torch.

Asserts the torch chain reproduces the validated NumPy chain, that gradients
reach every calibratable parameter, and that the collocation loss is nonzero on
a deliberately non-monotone function (i.e. the penalty can actually fire).
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


try:
    import torch
except ImportError:
    print("SKIP: torch not installed in this environment.")
    print("Run this file wherever you train; it is the gate on the torch layer.")
    sys.exit(0)

from opc_pinn import physics as P
from opc_pinn.data import load_dataset
from opc_pinn.models import PhysicsChain, AdditiveModel, CoupledModel
from opc_pinn.losses import monotonicity_collocation_loss

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

df = load_dataset()
ox_np = df[["CaO", "SiO2", "Al2O3", "Fe2O3", "SO3"]].to_numpy(float)
wc_np, age_np = df["wc"].to_numpy(float), df["age"].to_numpy(float)
ox = torch.tensor(ox_np, dtype=torch.float64)
wc = torch.tensor(wc_np, dtype=torch.float64)
age = torch.tensor(age_np, dtype=torch.float64)

print("\n1. torch vs numpy chain parity")
for hyd in ("eq5", "sf"):
    ref = P.chain(*[ox_np[:, i] for i in range(5)], wc_np, age_np, np, hydration=hyd)
    got = PhysicsChain(hydration=hyd, calibrated=False)(ox, wc, age)
    for key in ("alpha", "X", "fc_phys"):
        d = float(np.max(np.abs(got[key].detach().numpy() - ref[key])))
        check(f"{hyd}: {key} matches numpy", d < 1e-6, f"max|diff| = {d:.2e}")

print("\n2. gradients reach every calibratable parameter")
m = AdditiveModel(7, hydration="sf", calibrated=True)
x = torch.randn(len(df), 7, dtype=torch.float64)
pred, _ = m(x, ox, wc, age)
pred.sum().backward()
for name in ("f0", "n", "tau", "beta"):
    g = getattr(m.chain.params, f"raw_{name}").grad
    check(f"grad flows to {name}", g is not None and torch.isfinite(g).all()
          and float(g.abs()) > 0, f"grad = {float(g):.3e}" if g is not None else "None")

print("\n3. frozen chain has no chain gradients but network still trains")
mf = AdditiveModel(7, hydration="eq5", calibrated=False)
pred, _ = mf(x, ox, wc, age); pred.sum().backward()
check("frozen chain params are buffers, not parameters",
      not any("raw_" in n for n, _ in mf.chain.params.named_parameters()))
check("network weights still receive gradients",
      all(p.grad is not None for p in mf.net.parameters()))

print("\n4. softplus init lands exactly on literature values")
f0, n, tau, beta = PhysicsChain("sf", calibrated=True).params.values()
for nm, got_v, want in (("f0", f0, 234.0), ("n", n, 3.0),
                        ("tau", tau, 1.0), ("beta", beta, 0.7)):
    check(f"{nm} initialises to {want}", abs(float(got_v) - want) < 1e-6,
          f"got {float(got_v):.8f}")

print("\n5. coupled architecture respects its bounds")
mc = CoupledModel(7, hydration="sf", calibrated=True)
with torch.no_grad():
    _, aux = mc(x * 10, ox, wc, age)
a_max = P.alpha_max_mills(wc)
check("alpha_hat within [0, alpha_max]",
      bool(((aux["alpha"] >= 0) & (aux["alpha"] <= a_max + 1e-9)).all()))
check("X_hat within [0, 1]", bool(((aux["X"] >= 0) & (aux["X"] <= 1 + 1e-9)).all()))
check("|d_alpha| <= scale", float(aux["d_alpha"].abs().max()) <= 0.10 + 1e-9)
check("|d_X| <= scale", float(aux["d_X"].abs().max()) <= 0.05 + 1e-9)

print("\n6. soft clip restores gradient outside the bound")
mh = CoupledModel(7, calibrated=True, soft_clip=False)
ms = CoupledModel(7, calibrated=True, soft_clip=True)
grads = {}
for tag, mm in (("hard", mh), ("soft", ms)):
    xx = torch.randn(len(df), 7, dtype=torch.float64) * 20
    p, _ = mm(xx, ox, wc, age); p.sum().backward()
    grads[tag] = sum(float(q.grad.abs().sum()) for q in mm.net.parameters()
                     if q.grad is not None)
check("soft clip yields nonzero network gradient at saturation",
      grads["soft"] > 0, f"hard {grads['hard']:.3e}, soft {grads['soft']:.3e}")

print("\n7. collocation loss fires on a non-monotone model")
class Bad(torch.nn.Module):
    def forward(self, x, ox, wc, age):
        # decreasing in age AND increasing in wc: violates both rules, and
        # depends on both inputs so autograd has a path to each
        return -age + 5.0 * wc, {}
colloc = {"ox": ox.clone().requires_grad_(True),
          "wc": wc.clone().requires_grad_(True),
          "age": age.clone().requires_grad_(True)}
L = monotonicity_collocation_loss(Bad(), colloc, lambda o, w, a: torch.zeros(len(o), 7, dtype=torch.float64))
check("penalty is strictly positive for a decreasing-in-age model",
      float(L) > 0, f"L_colloc = {float(L):.4f}")

class Good(torch.nn.Module):
    def forward(self, x, ox, wc, age):
        return age.clone() - 10.0 * wc, {}
L2 = monotonicity_collocation_loss(Good(), colloc, lambda o, w, a: torch.zeros(len(o), 7, dtype=torch.float64))
check("penalty is zero for a correctly-monotone model", float(L2) < 1e-12,
      f"L_colloc = {float(L2):.2e}")

print("\n" + "=" * 64)
print(f"TORCH PARITY: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  -", f)
print("=" * 64)
sys.exit(1 if FAIL else 0)
