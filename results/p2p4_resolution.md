# Proposed resolution of P2 (unigram smoothing) and P4 (EOS packing)

**Status: proposal.** `PLAN.md` is approved-final and is not edited. If this survives
review it is appended to `AMENDMENTS.md`, dated, before any confirmatory run.

Both items were recorded as "must be reconciled against Tao's implementation, or fixed
here with rationale, before any confirmatory run." This resolves them by measurement
rather than by argument.

---

## 1. A correction to the record: the reference *does* expose `L_u`

`AMENDMENTS.md` P2 states "their released code does not expose the unigram construction."
That is half wrong and the wrong half matters. `reference/tinyllama_pretrain.py:406-435`
contains `validate_pplu`, the exact routine that produced the `Lossu` column:

```python
probabilities   = lookup_probabilities[targets].unsqueeze(2)          # [B,S,1]
normalized      = torch.nn.functional.softmax(logits, dim=-1) / probabilities
loss            = torch.nn.functional.nll_loss(torch.log(normalized), targets)
```

`nll_loss` selects the target column, so this is exactly

```
loss = −log p_model(target) + log p_unigram(target)  =  CE_model − H_unigram
```

**This confirms the definition and the sign convention in `src/metrics.py` are correct**,
including that `L_u` is negative when the model beats unigram. That was previously asserted
from their published row values (−1.35 / −2.10 / −2.50) and is now confirmed from source.

One convention detail to match: they accumulate a per-batch mean and then average the
batch means (`losses.append(loss.item())`, then `losses.mean()`). This equals the
token-weighted mean that `src/metrics.py` computes **only when every batch has the same
token count**. Our eval must therefore either use uniform full blocks or the token-weighted
mean; it must not mix a short final batch into an unweighted mean of means.

### What it still does not expose, and cannot

The lookup table is loaded from an external file that is not in the release:

```python
with open(path_to_tokenid_probabilities, 'rb') as f:
    tokenid_probabilities = json.load(f)
max_key = max(tokenid_probabilities.keys())
lookup_probabilities = torch.empty(max_key + 1).to(fabric.device)
```

Two defects make the estimator unrecoverable from the code even in principle:

1. `json.load` returns **string** keys. `max(...)` is therefore a lexicographic max over
   strings and `max_key + 1` raises `TypeError: can only concatenate str (not "int") to
   str`. Verified directly. **The released script cannot run as published**, so it is not
   byte-for-byte the script that produced the results, and the probabilities file it
   expects has no recoverable schema.
2. `torch.empty`, not `torch.zeros`. Any id absent from the dict keeps **uninitialised
   memory** as its unigram probability. So the code has no representable zero-frequency
   convention at all: an unseen token would contribute an arbitrary value.

The honest conclusion is that P2's zero-frequency question **cannot** be answered by
matching their implementation. It has to be fixed here with rationale. The remaining
question is whether the choice can change any conclusion.

---

## 2. Measurement: how much can the convention move a decision?

Smoothing enters `L_u` only through `H_unigram`, which depends on token counts and not on
any trained model. The entire sensitivity is therefore computable on the corpus already
tokenized, with no GPU. Both M1 and M2 compare `L_u` **across** vocabularies, so a shift
common to all V is harmless and only the **spread across V** can move a decision.

`scripts/p2_smoothing_sensitivity.py`, over all 20 vocabularies, unigram fitted on the full
`train` array and evaluated on `selection_val` (17,749 documents, identical text for every
V):

| convention | spread of shift across V (nats/token) |
|---|---|
| add-0.5 | 2.27e−06 |
| add-0.1 | 4.34e−06 |
| add-0.01 | 5.15e−06 |
| add-1e−4 | 7.50e−06 |
| add-1e−6 | 1.10e−05 |
| **drop EOS (P4, metric side)** | **1.17e−03** |

Six orders of magnitude of smoothing strength agree to within 1.1e−5 nats.

### Converting nats into the decision's own units

M1 is a test on `θ = ln(N_v*/N_v_pred)` with `N_v*` from `argmin_V L_u`, so a perturbation
matters only through how far it moves an argmin. For `L_u` locally quadratic with curvature
`a`, a perturbation `ε` displaces the argmin to where `a·(x − x*) + ε′(x) = 0`, so what
matters is the **slope** `ε′`, against the curvature `a`.

**Two corrections to an earlier version of this analysis**, both of which moved the answer
by an order of magnitude and are recorded because the first version was wrong:

1. *Spread is not slope.* Converting a convention's cross-V spread `S` into a monotone tilt
   `S / range(ln V)` is **not** a worst case — a perturbation with bounded range can have
   arbitrarily large derivative. The measured shift is now used directly and its largest
   slope between adjacent grid vocabularies taken. That is up to **39× larger** than the
   tilt heuristic gave.
2. *The curvature floor depends on how it is fitted.* A quadratic fitted globally over
   Tao's grid [4096, 96256] is not a local model of the minimum. Curvature is now computed
   under three windows (global, 7-point, 5-point about the minimum) and the smallest value
   taken, which is the conservative direction — weaker curvature means a given slope moves
   the argmin further.

`scripts/p2p4_decision_relevance.py` rebuilds IsoFLOP slices exactly as their approach2
does (quadratic `interp1d` of `Lossu` against FLOPs, interpolation only, never
extrapolation) across all six of their `N_nv` families:

| window | n | min curvature | median | max |
|---|---|---|---|---|
| global | 30 | 0.04732 | 0.13613 | 0.33729 |
| local-7 | 29 | 0.01583 | 0.14031 | 0.19355 |
| local-5 | 28 | **0.00965** | 0.11381 | 0.29580 |

Twenty-one slices fit a minimum outside the measured grid (one global slice returns
`V* = 788`; some local ones return `V* = 3`), which shows the quadratic is not always a
trustworthy model. Excluding those does **not** rescue the floor — the local-5 minimum of
0.00965 has its `V*` inside the grid and survives the filter. It is used as-is.

Taking `a = 0.00965`, the slope needed to displace the argmin by the M1 margin
`ln 1.5 = 0.4055` is **0.003913 nats per unit `ln V`**:

| convention | spread (nats) | max local slope | factor below the M1 margin |
|---|---|---|---|
| add-0.5 | 2.27e−06 | 2.28e−05 | 171× |
| add-0.1 | 4.34e−06 | 4.37e−05 | 90× |
| add-0.01 | 5.15e−06 | 5.19e−05 | 75× |
| add-1e−4 | 7.50e−06 | 6.05e−05 | 65× |
| add-1e−6 | 1.10e−05 | 6.87e−05 | **57×** |
| **drop EOS** | 1.17e−03 | 4.48e−04 | **9×** |

So P2 clears the margin by at least 57× under the most pessimistic curvature available,
while **P4 clears it by only 9×**. That gap is the reason the two are resolved differently
below.

### Zero-frequency events are real but negligible

They do occur, so an unsmoothed MLE would be undefined and *some* convention is required:

| V | eval ids unseen in train | eval instances | share of eval tokens |
|---|---|---|---|
| 384 | 144, 145 | 36 | 7.6e−07 |
| 3456 | 1868 | 1 | 3.8e−08 |
| 6912 | 1770, 1868 | 4 | 1.7e−07 |

Forty token instances out of ~97M. Every other vocabulary has none. This is why the
smoothing strength is nearly irrelevant: 46–69 ids per vocabulary are unseen in training,
but they are essentially unseen in evaluation too.

---

## 3. Proposed resolutions

### P2 — ADOPT add-1 (Laplace) over the full vocabulary, fitted on the training split

Rationale, in the order that carries the weight:

1. Matching Tao is **impossible**, not merely unattempted — the released loader cannot run
   and has no representable zero-frequency convention (§1).
2. A convention is **required**, because zero-frequency eval tokens occur in 3 of 20
   vocabularies.
3. The choice is **not decision-relevant**: every smoothing strength from add-1 to add-1e−6
   perturbs cross-vocabulary `L_u` differences by ≤1.1e−5 nats, whose largest local slope
   is at least 57× below what would move `θ` by the M1 margin — and that is under the most
   pessimistic curvature recoverable from Tao's data, not a favourable one.

This is already what `src/metrics.py` implements, so nothing changes in code; what changes
is that it stops being an open item and becomes a dated, justified, quantified choice. The
`OPEN ITEM` docstring should be replaced with a pointer to this record.

**Reported as a limitation:** `L_u` here is not guaranteed numerically comparable to their
published `Lossu` values, because the unigram estimator differs by an unknown amount. This
does not affect the study, which tests the *location of the optimum over V*, not absolute
loss — but it forbids any direct comparison of our `L_u` numbers against their table.

### P4 — KEEP the EOS separator, with a pilot-conditional gate on M2

The metric-side effect is bounded above: 1.17e−03 nats of cross-V spread, whose largest
local slope sits **9×** below the M1 margin. That is the whole margin, and it is thin —
one order of magnitude, against a curvature floor that itself moved 5× when the fitting
window changed. P4 is therefore **not** declared settled the way P2 is.

**The bound also does not cover M2, and does not cover training.** Two further gaps, stated
plainly rather than waved through:

- **M2 has no equivalence margin.** It is a sign test on `D = L_u(V_run, 1.1C) − L_u(V*, C)`.
  A 1.17e−03 nat perturbation flips the sign of `D` whenever `|D|` is that small. Nothing
  measured so far bounds `|D|`.
- **The measurement above is metric-side only.** Removing EOS would change the training
  token stream, hence the trained model, hence `CE_model`. That effect is not measurable
  without running models and is *not* bounded by anything in this document.

Proposed: keep EOS-separated packing (it is the standard choice for packed pretraining, it
is implemented, and removing it would require retokenizing 5.1B tokens), and add an
explicit gate to Stage B.7:

> **Gate.** The power pilot reports `|D̂_pilot|`. If `|D̂_pilot| < 1.2e−02` nats — ten times
> the measured P4 metric-side spread — the packing convention is capable of flipping the M2
> sign test and must be resolved before any confirmatory run. If `|D̂_pilot| ≥ 1.2e−02`, P4
> is settled for M2 as well and is reported as a limitation only.

This gate is preregistered *before* the pilot runs, so it cannot be a post-hoc reaction to
its result.

---

## 4. Residual risk

The curvature bound is measured on Tao's families at `N_nv ≥ 33M`. This study runs at
2M–16M, where curvature is unmeasured — that is, in part, what the study is for. **The
bound is therefore not fully non-circular**: it assumes `L_u` below 33M is not dramatically
flatter in `V` than anything observed above it. If it is, the margins shrink in proportion.

That risk falls almost entirely on P4. A 9× margin is erased by a 9× flattening, which is
not obviously implausible. The P2 margin of 57× has enough headroom that no flattening
consistent with the study being worth running would reach it.

**Mitigation.** The pilot measures curvature in our own regime directly. Its measured
`d²L_u/d(lnV)²` is to be recorded and compared against the 0.00965 floor used here. If the
pilot's curvature is below 0.00965, this entire conversion is invalid in our regime and
both P2 and P4 must be re-derived against the pilot's own value before any confirmatory
run. That check is preregistered here, before the pilot runs.
