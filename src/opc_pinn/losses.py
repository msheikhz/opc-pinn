"""Loss terms (Eqs. 14-17, 22-23)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def data_loss(pred, target):
    return F.mse_loss(pred, target)


def physics_consistency_loss(pred, fc_phys):
    """Eq. (16). NOTE: for the additive architecture pred - fc_phys == Delta_NN
    identically, so this term is algebraically lambda*||Delta_NN||^2 — output
    shrinkage toward the baseline, not a governing-equation residual. Kept to
    reproduce the manuscript's frozen configurations; documented as such."""
    return F.mse_loss(pred, fc_phys)


def bounds_loss(pred):
    """Eq. (17): penalise physically inadmissible negative strength."""
    return torch.mean(torch.clamp(-pred, min=0.0) ** 2)


def correction_magnitude_loss(d_alpha, d_X, lam_a, lam_X):
    """The two penalty terms of Eq. (22)."""
    return lam_a * torch.mean(d_alpha ** 2) + lam_X * torch.mean(d_X ** 2)


def monotonicity_collocation_loss(model, colloc, feature_fn):
    """Eq. (23): penalise d(fc)/d(age) < 0 and d(fc)/d(wc) > 0 at unlabeled
    collocation points.

    ``colloc`` holds requires_grad tensors (ox, wc, age); ``feature_fn`` maps
    them to the standardised feature matrix so the graph stays connected from
    wc and age through to the prediction.
    """
    ox, wc, age = colloc["ox"], colloc["wc"], colloc["age"]
    x = feature_fn(ox, wc, age)
    pred, _ = model(x, ox, wc, age)
    # allow_unused=True: if the prediction happens not to depend on age or wc
    # (e.g. a degenerate model, or a fitted chain whose exponent zeroes a path),
    # the corresponding gradient is None rather than an error. Treat None as a
    # zero-gradient field, which is the correct reading — a term the output does
    # not depend on cannot violate monotonicity. The real AdditiveModel and
    # CoupledModel both depend on age and wc through the physics chain, so this
    # guard only ever fires for degenerate/stub models.
    g_age, g_wc = torch.autograd.grad(pred.sum(), [age, wc],
                                      create_graph=True, allow_unused=True)
    zero = torch.zeros_like(pred)
    g_age = zero if g_age is None else g_age
    g_wc = zero if g_wc is None else g_wc
    return torch.mean(torch.clamp(-g_age, min=0.0) ** 2
                      + torch.clamp(g_wc, min=0.0) ** 2)
