"""PyTorch models. NOT EXECUTED in the authoring environment (no torch, no network).

Everything torch-independent — the physics chain, splitters, metrics, bootstrap,
analytical fitter — is validated in tests/test_validation.py and runs today.
This module is a thin torch layer over the same physics functions, so the maths
is already checked; what remains unverified is the torch plumbing.

Before trusting any number from this file, run:
    python3 tests/test_torch_parity.py
which asserts the torch chain reproduces the validated NumPy chain to 1e-6 and
that gradients flow to every calibratable parameter.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import physics as P


def _inverse_softplus(x: float) -> float:
    """theta such that softplus(theta) == x, so training starts at exactly x."""
    return float(np.log(np.expm1(x)))


class ResidualBlock(nn.Module):
    """Eq. (13): h <- tanh(h + W2 tanh(W1 h + b1) + b2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        for m in (self.fc1, self.fc2):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h):
        return torch.tanh(h + self.fc2(torch.tanh(self.fc1(h))))


class ResidualNet(nn.Module):
    """Input projection -> 3 residual blocks -> dropout -> linear head."""

    def __init__(self, n_in: int, hidden: int = 64, n_blocks: int = 3,
                 dropout: float = 0.10, n_out: int = 1):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        for b in self.blocks:
            h = b(h)
        return self.head(self.drop(h))


class ChainParameters(nn.Module):
    """f0, n, tau, beta, held via softplus so they stay strictly positive.

    ``calibrated=False`` registers them as buffers (fixed); ``True`` registers
    them as parameters updated by the same optimizer as the network weights.
    Initialised by inverse-softplus so training begins at exactly the literature
    values regardless of which mode is used — the frozen and calibrated arms
    therefore start from an identical chain.
    """

    def __init__(self, calibrated: bool, f0=P.F0_LIT, n=P.N_LIT,
                 tau=P.TAU_INIT, beta=P.BETA_INIT):
        super().__init__()
        self.calibrated = calibrated
        for name, val in (("f0", f0), ("n", n), ("tau", tau), ("beta", beta)):
            t = torch.tensor(_inverse_softplus(val), dtype=torch.float64)
            if calibrated:
                self.register_parameter(f"raw_{name}", nn.Parameter(t))
            else:
                self.register_buffer(f"raw_{name}", t)

    def values(self):
        sp = torch.nn.functional.softplus
        return (sp(self.raw_f0), sp(self.raw_n), sp(self.raw_tau), sp(self.raw_beta))


def _soft_clip(x, lo, hi, sharpness=50.0):
    """Differentiable clip. Hard clipping has zero gradient outside the bound,
    which discards the learning signal exactly when a large correction is wanted
    — one of the two hypotheses the manuscript offers for why the coupled
    architecture failed. This variant attenuates rather than discards, so the
    coupled arm can be retested without that confound."""
    return lo + (hi - lo) * torch.sigmoid(sharpness * (x - lo) / (hi - lo) - sharpness / 2)


class PhysicsChain(nn.Module):
    """Differentiable five-stage chain. Calls the same functions as the NumPy
    version, with xp=torch."""

    def __init__(self, hydration="eq5", calibrated=False,
                 unhydrated_basis="reacted_silicates"):
        super().__init__()
        self.hydration = hydration
        self.unhydrated_basis = unhydrated_basis
        self.params = ChainParameters(calibrated)

    def forward(self, ox, wc, age, alpha_override=None, X_override=None):
        """ox: (N,5) tensor of CaO, SiO2, Al2O3, Fe2O3, SO3."""
        f0, n, tau, beta = self.params.values()
        CaO, SiO2, Al2O3, Fe2O3, SO3 = (ox[:, i] for i in range(5))
        C3S, C2S, C3A, C4AF = P.bogue(CaO, SiO2, Al2O3, Fe2O3, SO3, torch)

        if alpha_override is not None:
            alpha = alpha_override
        elif self.hydration == "eq5":
            alpha = P.alpha_eq5(age, torch)
        else:
            alpha = P.alpha_sf(age, wc, tau, beta, torch)

        X, v_gel, v_cap, v_unhyd = P.gel_space_ratio(
            C3S, C2S, alpha, wc, torch, unhydrated_basis=self.unhydrated_basis)
        if X_override is not None:
            X = X_override
        fc_phys = P.powers_strength(X, f0, n, torch)
        return dict(alpha=alpha, X=X, fc_phys=fc_phys, C3S=C3S, C2S=C2S,
                    p_cap=P.capillary_porosity(wc, alpha))


class AdditiveModel(nn.Module):
    """f_hat = fc_phys + Delta_NN(x)   —  Eq. (12)."""

    def __init__(self, n_features, hydration="eq5", calibrated=False,
                 dtype=torch.float64, **kw):
        super().__init__()
        self.chain = PhysicsChain(hydration, calibrated, **kw)
        self.net = ResidualNet(n_features, n_out=1)
        # nn.Linear initialises weights as float32; the physics chain runs in
        # float64. Cast the whole module to a single dtype (double by default)
        # so linear layers and chain math agree regardless of how the caller
        # constructs the model. Without this, a float64 input hits a float32
        # weight and torch raises "mat1 and mat2 must have the same dtype".
        self.to(dtype)

    def forward(self, x, ox, wc, age):
        out = self.chain(ox, wc, age)
        delta = self.net(x).squeeze(-1)
        return out["fc_phys"] + delta, {**out, "delta": delta}


class CoupledModel(nn.Module):
    """Bounded nudges to alpha and X, injected mid-chain — Eq. (21).

    ``soft_clip=True`` replaces the manuscript's hard clip with a differentiable
    one, so the "zero gradient outside the bound" hypothesis can be tested.
    ``d_alpha_scale`` / ``d_X_scale`` are exposed for the sweep the manuscript
    identifies as missing.
    """

    def __init__(self, n_features, hydration="sf", calibrated=True,
                 d_alpha_scale=0.10, d_X_scale=0.05, soft_clip=False,
                 dtype=torch.float64, **kw):
        super().__init__()
        self.chain = PhysicsChain(hydration, calibrated, **kw)
        self.net = ResidualNet(n_features, n_out=2)
        self.d_alpha_scale, self.d_X_scale = d_alpha_scale, d_X_scale
        self.soft_clip = soft_clip
        self.to(dtype)   # see AdditiveModel: unify linear-layer and chain dtype

    def forward(self, x, ox, wc, age):
        f0, n, tau, beta = self.chain.params.values()
        raw = self.net(x)
        d_alpha = self.d_alpha_scale * torch.tanh(raw[:, 0])
        d_X = self.d_X_scale * torch.tanh(raw[:, 1])

        nominal = self.chain(ox, wc, age)
        a_max = P.alpha_max_mills(wc)
        clip = _soft_clip if self.soft_clip else (
            lambda v, lo, hi: torch.clamp(v, min=lo, max=hi)
            if np.isscalar(hi) else torch.minimum(torch.clamp(v, min=lo), hi))
        alpha_hat = clip(nominal["alpha"] + d_alpha, 0.0, a_max)

        corrected = self.chain(ox, wc, age, alpha_override=alpha_hat)
        X_hat = clip(corrected["X"] + d_X, 0.0, 1.0)
        fc = P.powers_strength(X_hat, f0, n, torch)
        return fc, {**corrected, "X": X_hat, "alpha": alpha_hat,
                    "d_alpha": d_alpha, "d_X": d_X}


class PureMLP(nn.Module):
    """Baseline: identical topology, 6 raw features, no physics path."""

    def __init__(self, n_features, dtype=torch.float64):
        super().__init__()
        self.net = ResidualNet(n_features, n_out=1)
        self.to(dtype)   # see AdditiveModel: unify linear-layer and chain dtype

    def forward(self, x, ox=None, wc=None, age=None):
        return self.net(x).squeeze(-1), {}
