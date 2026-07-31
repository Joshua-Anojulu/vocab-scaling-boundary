# Stage B.3 — zero-GPU re-analysis: results

Run against `reference/exp_data.csv` (1200 rows, 6 families × 10 vocabularies × 20 budgets).
No GPU-hours spent.

## PASS — the ruler is exactly validated

`tests/test_reference_parity.py`: **12/12 pass.**

- `func_flops` reproduces **every one of the 1200 published FLOPs values** to < 1e-12
  relative error. This simultaneously validates the fertility polynomial, the width
  lookup and the FLOP convention.
- The accounting identity `N_nv = nominal_total − 2·16384·d` reproduces **all six**
  published `Non_vocab_parameters` values exactly. The identity predicted
  `682M → 631,668,352` *before* that family was known to be in the data.
- `N_nv` is paired with the full vocabulary grid at every family ⇒ it cannot be a
  function of `V` ⇒ it excludes **both** embedding tables. `total(untied) = N_nv + 2·V·d`.
- Two families (110M, 176M) **share `d = 768`** with different `N_nv`. The reference
  family is therefore not a function of width, which is direct evidence that no sub-50M
  width rule can be recovered from it — ours is an admitted new extrapolation.
- `L_u < 0` on all 1200 rows, confirming the sign convention (more negative is better).

## PASS — design numbers reproduce the approved plan exactly

| target | d | d_ffn | n_head | n_layer | realized N_nv | V_pred | V_run | grid |
|---|---|---|---|---|---|---|---|---|
| 2M | 192 | 512 | 3 | 5 | 2,213,952 | 1714.78 | 1664 | 384, 896, 1664, 3456, 6912 |
| 4M | 256 | 704 | 4 | 5 | 4,016,896 | 2115.47 | 2176 | 512, 1024, 2176, 4224, 8448 |
| 8M | 320 | 832 | 5 | 7 | 8,463,040 | 3154.00 | 3200 | 768, 1536, 3200, 6272, 12672 |
| 16M | 384 | 1024 | 6 | 9 | 15,932,544 | 4458.75 | 4480 | 1152, 2176, 4480, 8960, 17792 |

Budget: `n=3` → 91 runs, `1.26015e18`; **`n=5` (the floor) → 139 runs, `1.86858e18`**
= 51.9 / 34.6 / 20.8 ideal hours at 10 / 15 / 25 TFLOP/s. With an all-scale boundary
extension: 159 runs, `2.11794e18`, 39.2 h @ 15. All match the approved plan.

## ✅ PASS — the analysis pipeline is validated to machine precision

**Resolved.** `src/isoflop.py` is a faithful port of `approach1_isoflops.py`, and it
recovers the three constants the reference itself ships in `optimal_Nv_predict.py`, from
`exp_data.csv` alone:

| constant | recovered | published | \|Δ\| |
|---|---|---|---|
| `K1` (N_nv) | −2.484651016385 | −2.484651016163 | 2.2e-10 |
| `K2` (N_v) | −1.589031299165 | −1.589031299256 | 9.1e-11 |
| `alpha2` | 0.416362263412 | 0.416362263414 | 1.1e-12 |

Frontier: 126 points. Relative MSE: N_nv 3.5e-4, N_v 6.7e-4, H 2.8e-4. Implied
`N_v ∝ N_nv^0.832725`, matching published A1 exactly. Locked in by
`tests/test_isoflop_port.py`; the tolerance (1e-7) is set by L-BFGS-B convergence, not
chosen to fit.

### Why the first attempt was 8.3% off — a structural difference, not a bug

The earlier in-house estimator fitted per-family loss bowls at interpolated budgets, took
each bowl's argmin, and regressed `ln N_v` on `ln N_nv`. That recovered α = 0.7450, or
0.7664 after applying their own 40% outlier removal iterated to convergence — so
outliers were never the explanation. The bias was systematic: recovered optima sat below
the prediction at every family, with the gap widening at scale, flattening the slope.

Their estimator is a different object:

1. **Pooled, not per-family.** Original and interpolated points are concatenated and
   sorted by FLOPs; families are not fitted separately.
2. **A running-minimum Pareto frontier, not a per-budget argmin.** With `length_bin = 1`,
   a point is kept only if its loss is strictly below every point kept so far.
3. **Huber loss (δ=0.001) in log space**, minimised by L-BFGS-B over a 400-point grid of
   initial guesses, selected by relative MSE.
4. **The N_nv and H exponents are held FIXED at 0.5** (following Chinchilla); only `N_v`'s
   exponent is free.
5. **Interpolation is along a diagonal, not a mesh.** `griddata` cubic over `(N_nv, H)` is
   evaluated at `zip(new_Nnv, new_H)` — 50 points along the compute-optimal ray, since
   both scale as √C. A second pass interpolates quadratically across `V` using only each
   family's maximum-H point.

Any one of these would move an exponent at the third decimal place. Together they account
for the whole 8.3%.

**Lesson recorded for Stage D:** "a reasonable estimator" is not a substitute for *their*
estimator when the claim under test is about their published number. The plan's insistence
on matching Tao wherever the claim depends on it was load-bearing, and this is the first
place it paid.

### Scope note: this does not invalidate the curvature estimate below

The discrepancy was in the *exponent estimator*. The bowl fitter — quadratic in `ln V` at
fixed budget — is a separate object, and is the correct one for curvature, which is what
the power calculation needs. A1's frontier does not produce bowls at all. The curvature
figures below stand.

## Curvature at the 33M anchor (conservative cross-check only)

All 8 sampled budgets give convex bowls (10 vocabulary points each).

- quadratic coefficient in `ln V`: median **0.1147**, range 0.0435 – 0.1864
- `V*` median **4753**
- bowl-fit RMSE median **0.0425** in `L_u` units

Per the plan this feeds the power calculation only as a conservative cross-check, never
as a plug-in for below-range scales. Note one budget (`C = 1.377e17`) produced
`V* = 604`, a clear outlier from a very flat bowl — a concrete reminder that a
quadratic minimum is poorly determined when curvature is low, which is precisely the
risk the seed floor and the lack-of-fit gate exist to catch.

Note also that at the anchor, `|ln(4753/6179)| = 0.262`, **inside** the M1 equivalence
margin of `ln 1.5 = 0.4055`. On their own data at their own anchor, the prediction would
pass M1 — encouraging for the method, but it is their data, not ours.
