"""Five-stage analytical physics chain, written backend-agnostically.

Every function takes an ``xp`` argument that is either ``numpy`` or ``torch``.
Only operations present in both namespaces with identical signatures are used
(``exp``, ``clip``, ``where``, arithmetic), so the same code path serves the
NumPy validation suite and the differentiable PyTorch model.

Stage numbering follows the manuscript (Section III-B).

IMPORTANT — volume-balance provenance
-------------------------------------
The manuscript's worked example reports X = 0.4421 for
(CaO 64, SiO2 21, Al2O3 5.5, Fe2O3 3.5, SO3 2.5, w/c 0.45, alpha 0.75).
That value is NOT reproducible from the manuscript text: eight plausible
readings of "V_gel" and "V_unhyd" span X = 0.378 to 0.874, with the two nearest
natural readings giving 0.4386 and 0.4483. Reproducing 0.4421 exactly requires an
unhydrated mass of 19.99 g per 100 g cement, which corresponds to no natural
definition.

This module therefore implements an explicit, documented definition and exposes
the alternatives via ``unhydrated_basis``. Validation is against physical
invariants (bounds, monotonicity, mass balance), not against the unreproducible
0.4421. Set ``unhydrated_basis`` to whichever matches your original code once you
have checked it.
"""
from __future__ import annotations

# Molar masses, g/mol
M_C3S, M_C2S, M_CSH, M_CH, M_H2O = 228.32, 172.24, 227.0, 74.09, 18.015

# Densities, g/cm3
RHO_CEMENT, RHO_WATER, RHO_CSH, RHO_CH = 3.15, 1.00, 2.60, 2.24

# Stoichiometric coefficients of Eqs. (6)-(7)
H_PER_C3S, CH_PER_C3S = 5.3, 1.3
H_PER_C2S, CH_PER_C2S = 4.3, 0.3

# Literature starting values (Section III-B, III-J)
F0_LIT, N_LIT = 234.0, 3.0
TAU_INIT, BETA_INIT = 1.0, 0.7


# ---------------------------------------------------------------- Stage 1
def bogue(CaO, SiO2, Al2O3, Fe2O3, SO3, xp=None, clip_negative=True):
    """Bogue clinker phases, Eqs. (1)-(4). Returns (C3S, C2S, C3A, C4AF) in %.

    ``clip_negative`` floors each phase at zero. This is NOT cosmetic on this
    dataset: source 1 (16 of 62 rows) has a lime saturation high enough that
    Eq. (2) returns C2S = -10.82% at SO3 = 0, and still returns a negative value
    for every SO3 assumption up to 3%. Left unclipped, a negative C2S propagates
    a negative reacted mass through Eqs. (6)-(7), subtracting phantom volume from
    V_gel and corrupting X for 26% of the dataset. Clipping to zero is the
    standard treatment of a Bogue result that falls outside its domain of
    validity; the underlying issue is that Bogue is being applied to a cement it
    does not describe well, which belongs in the manuscript as a limitation.
    """
    C3S = 4.071 * CaO - 7.600 * SiO2 - 6.718 * Al2O3 - 1.430 * Fe2O3 - 2.852 * SO3
    C2S = 2.867 * SiO2 - 0.7544 * C3S
    C3A = 2.650 * Al2O3 - 1.692 * Fe2O3
    C4AF = 3.043 * Fe2O3
    if clip_negative:
        xp = xp or _infer_xp(C3S)
        C3S, C2S, C3A, C4AF = (xp.clip(v, 0.0, None) for v in (C3S, C2S, C3A, C4AF))
    return C3S, C2S, C3A, C4AF


def _infer_xp(v):
    """Return the array namespace of ``v`` so bogue() can be called without xp."""
    mod = type(v).__module__.split(".")[0]
    if mod == "torch":
        import torch
        return torch
    import numpy
    return numpy


# ---------------------------------------------------------------- Stage 2
def alpha_eq5(age, xp):
    """Original age-only hydration, Eq. (5).

    Retained to reproduce the frozen-chain configurations. Physically incorrect
    in two ways, both documented in the manuscript revision: alpha(0) = 0.28,
    and there is no dependence on w/c, so no water-limited ceiling.
    """
    return 0.28 + 0.67 * (1.0 - xp.exp(-0.05 * age))


def alpha_max_mills(wc, xp=None):
    """Ultimate degree of hydration, Eq. (19).

    Schindler & Folliard's relation with fly-ash and slag terms zeroed; the
    underlying empirical finding is due to Mills (1966). Their general form caps
    at 1.0, which is never active over this dataset's w/c range.
    """
    return 1.031 * wc / (0.194 + wc)


def alpha_sf(age, wc, tau, beta, xp):
    """Schindler-Folliard exponential hydration, Eq. (18).

    alpha(t) = alpha_max(w/c) * exp(-(tau/t)**beta)

    Written in the standard form. Note exp(-(tau/t)**beta) and exp(-tau/t**beta)
    are the same family under tau' = tau**beta, so a reparameterised
    implementation gives identical fits; this form is used because it matches the
    source. alpha(0) = 0 exactly, by construction.
    """
    return alpha_max_mills(wc, xp) * xp.exp(-((tau / age) ** beta))


# ------------------------------------------------------- Stages 2b, 3, 4
def hydration_products(C3S, C2S, alpha):
    """Masses of C-S-H and CH per 100 g cement, from Eqs. (6)-(7)."""
    r3 = alpha * C3S            # reacted C3S, g per 100 g cement
    r2 = alpha * C2S            # reacted C2S
    n3 = r3 / M_C3S             # mol
    n2 = r2 / M_C2S
    m_csh = (n3 + n2) * M_CSH
    m_ch = (n3 * CH_PER_C3S + n2 * CH_PER_C2S) * M_CH
    m_bound_water = (n3 * H_PER_C3S + n2 * H_PER_C2S) * M_H2O
    return m_csh, m_ch, m_bound_water, r3, r2


def gel_space_ratio(C3S, C2S, alpha, wc, xp, unhydrated_basis="reacted_silicates"):
    """Stages 3-4: volume balance (Eq. 8) and gel-space ratio (Eq. 9).

    Basis: 100 g cement + 100*w/c g water.

    ``unhydrated_basis`` selects what counts as unhydrated solid:
      "reacted_silicates" : 100 - (reacted C3S + reacted C2S)   [default]
      "one_minus_alpha"   : (1 - alpha) * 100
      "silicates_only"    : (1 - alpha) * (C3S + C2S)
      "all_phases"        : (1 - alpha) * (C3S + C2S + C3A + C4AF)  [needs C3A, C4AF]

    Capillary volume absorbs both free water and chemical shrinkage, which is why
    bound water is not subtracted a second time.
    """
    m_csh, m_ch, _, r3, r2 = hydration_products(C3S, C2S, alpha)

    v_gel = m_csh / RHO_CSH + m_ch / RHO_CH
    v_total = 100.0 / RHO_CEMENT + wc * 100.0 / RHO_WATER

    if unhydrated_basis == "reacted_silicates":
        m_unhyd = 100.0 - r3 - r2
    elif unhydrated_basis == "one_minus_alpha":
        m_unhyd = (1.0 - alpha) * 100.0
    elif unhydrated_basis == "silicates_only":
        m_unhyd = (1.0 - alpha) * (C3S + C2S)
    else:
        raise ValueError(f"unknown unhydrated_basis: {unhydrated_basis}")

    v_unhyd = m_unhyd / RHO_CEMENT
    v_cap = v_total - v_gel - v_unhyd
    # Guard: capillary volume cannot be negative. Clipping keeps X in (0, 1]
    # and keeps the expression differentiable everywhere it is used.
    v_cap = xp.clip(v_cap, 0.0, None) if hasattr(xp, "clip") else v_cap
    X = v_gel / (v_gel + v_cap)
    return X, v_gel, v_cap, v_unhyd


def capillary_porosity(wc, alpha):
    """Powers-Brownyard capillary porosity, Eq. (10). NN input only."""
    return (wc - 0.36 * alpha) / (wc + 0.32)


# ---------------------------------------------------------------- Stage 5
def powers_strength(X, f0, n, xp):
    """Powers gel-space ratio strength model, Eq. (11)."""
    return f0 * X ** n


# ------------------------------------------------------------- full chain
def chain(CaO, SiO2, Al2O3, Fe2O3, SO3, wc, age, xp,
          hydration="eq5", f0=F0_LIT, n=N_LIT, tau=TAU_INIT, beta=BETA_INIT,
          unhydrated_basis="reacted_silicates", alpha_override=None,
          clip_negative_phases=True):
    """Run the full five-stage chain. Returns a dict of every intermediate.

    ``hydration`` is "eq5" (age-only, frozen configs) or "sf" (Eqs. 18-19).
    ``alpha_override`` bypasses Stage 2, used by the worked-example check and by
    the coupled architecture, which injects a corrected alpha.
    """
    C3S, C2S, C3A, C4AF = bogue(CaO, SiO2, Al2O3, Fe2O3, SO3, xp,
                                clip_negative=clip_negative_phases)

    if alpha_override is not None:
        alpha = alpha_override
    elif hydration == "eq5":
        alpha = alpha_eq5(age, xp)
    elif hydration == "sf":
        alpha = alpha_sf(age, wc, tau, beta, xp)
    else:
        raise ValueError(f"unknown hydration model: {hydration}")

    X, v_gel, v_cap, v_unhyd = gel_space_ratio(
        C3S, C2S, alpha, wc, xp, unhydrated_basis=unhydrated_basis)
    p_cap = capillary_porosity(wc, alpha)
    fc_phys = powers_strength(X, f0, n, xp)

    return dict(C3S=C3S, C2S=C2S, C3A=C3A, C4AF=C4AF, alpha=alpha, X=X,
                v_gel=v_gel, v_cap=v_cap, v_unhyd=v_unhyd,
                p_cap=p_cap, fc_phys=fc_phys)
