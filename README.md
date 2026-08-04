# opc_pinn — physics-informed OPC paste strength prediction


## Status

| component | validated? | how |
|---|---|---|
| physics chain (Bogue, hydration, volume balance, Powers) | **yes** | 51/51 checks, `tests/test_validation.py` |
| CV splitters (5-fold stratified, LOSO) | **yes** | partition invariants, fold sizes |
| metrics | **yes** | cross-checked against sklearn |
| paired cluster bootstrap | **yes** | degenerate + separated cases |
| analytical fitter | **yes** | recovers synthetic ground truth to 5 dp |
| analytical ablation results | **yes** | executed, see `results/` |
| torch models + training | reproducible | run `tests/test_torch_parity.py` (gate) then `scripts/run_full_ablation.py` |

## Run

Run from the project root. Imports and the data file are both located
automatically, so any working directory works, but the folder layout must be
preserved (or everything flattened into one folder with `opc_pinn/` inside it).

```bash
python3 tests/test_validation.py            # 51 checks, no torch needed
python3 scripts/run_analytical_ablation.py  # real results, no torch needed

pip install torch
python3 tests/test_torch_parity.py          # GATE: run before trusting any torch number
python3 scripts/run_full_ablation.py --seeds 42 43 44 45 46
```

## Data

Uses `ew-points_finalN.txt` (62 rows, 11 sources) which includes the measured
SO3 column required by Eq. (1). 
"Available on Request"

## Known defects carried from the manuscript

1. **The worked example's X = 0.4421 is not reproducible.** Eight readings of the
   Stage 3 volume balance span X = 0.378–0.874; the two nearest give 0.4386 and
   0.4483. `physics.gel_space_ratio` exposes `unhydrated_basis` so you can select
   whichever matches the original code. Validation asserts physical invariants,
   not the unreproducible constant.
2. **Bogue returns a negative phase for 3 of 62 rows** (source 7, C3S = −10.1%).
   `bogue(..., clip_negative=True)` floors at zero; the underlying issue is that
   two sources sit outside Bogue's domain of validity.
