# Summary of results

All numbers below are the final results reported in the associated paper. Two
evaluation regimes are used on the 62-specimen, 11-source dataset: source-
stratified 5-fold and leave-one-source-out (LOSO). Neural configurations are run
over five random seeds (42–46) and reported as mean ± s.d. of the pooled LOSO
RMSE. Analytical results are deterministic.

Reproduce with:
    python scripts/run_analytical_ablation.py                    # analytical (no torch)
    python scripts/run_full_ablation.py --seeds 42 43 44 45 46   # neural (needs torch)

## Model-free baselines (pooled LOSO)

| baseline | RMSE (MPa) |
|---|---|
| predict training-set mean | 14.27 |
| linear regression (w/c, log-age) | 7.60 |
| analytical chain, 4 params fitted, no network | **6.71** |
| analytical chain, frozen at literature values | 19.74 |

## Neural configurations (5-seed pooled LOSO, mean ± s.d.)

| configuration | LOSO RMSE | best seed | 5-fold |
|---|---|---|---|
| additive, calibrated + phys. loss | 8.51 ± 0.95 | 7.24 | 5.50 |
| coupled, calibrated + collocation | 8.52 ± 0.42 | 8.47 | 5.83 |
| additive, calibrated | 8.65 ± 0.85 | 7.29 | 5.35 |
| coupled, calibrated | 8.99 ± 0.66 | 9.79 | 5.70 |
| additive, calibrated + collocation | 9.41 ± 0.88 | 8.02 | 5.05 |
| MLP + physical feature (7 feat.) | 11.59 ± 1.67 | 10.15 | 7.90 |
| additive, frozen (literature-parameter hybrid) | 11.95 ± 1.97 | 10.51 | 7.19 |
| coupled, frozen | 12.38 ± 0.88 | 11.77 | 11.74 |
| plain neural network (6 feat.) | 12.42 ± 2.66 | 12.68 | 8.95 |
| additive, frozen, no physics loss | 12.60 ± 1.90 | 10.73 | 7.28 |

## Key findings

1. Calibration cleanly separates a calibrated tier (8.5–9.4 MPa) from an
   uncalibrated tier (11.6–12.6 MPa), with no overlap.
2. Paired across seeds, the additive calibrated hybrid beats the frozen-chain
   hybrid by 3.3 ± 1.6 MPa and the plain network by 3.8 ± 2.3 MPa, consistent in
   sign across all five seeds; calibration also roughly halves seed variance.
3. Coupling the correction into the chain and the collocation penalty do not
   improve on the simple additive calibrated model.
4. No neural configuration surpasses the four-parameter analytical baseline
   (6.71 MPa) under LOSO.
5. Two of the eleven cements (sources S1, S7) lie outside the Bogue domain,
   returning inadmissible phase fractions; S1 supplies a quarter of the data.

## Notes on the physics chain

- The worked-example gel–space ratio X = 0.4421 is sensitive to the definition
  of "unhydrated cement" in the volume balance; the definition used here (100
  minus reacted silicate mass) is in `src/opc_pinn/physics.py`. Validation checks
  physical invariants, not the exact literature constant.
- Bogue returns a negative phase for the two out-of-domain cements; negative
  fractions are floored at zero (`bogue(..., clip_negative=True)`).
