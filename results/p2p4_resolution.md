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
matters only through how far it moves an argmin — which depends on the curvature of `L_u`
in `ln V`. `scripts/p2p4_decision_relevance.py` measures that curvature from Tao's own
released points, rebuilding IsoFLOP slices exactly as their approach2 does (quadratic
`interp1d` of `Lossu` against FLOPs, interpolation only, never extrapolation), then fitting
a quadratic in `ln V`. Thirty slices across all six of their `N_nv` families:

```
d²L_u/d(lnV)²  :  min 0.04732   median 0.13613   max 0.33729
```

Taking the **weakest** curvature observed, the tilt in `dL_u/d(lnV)` needed to displace the
argmin by the M1 margin `ln 1.5 = 0.4055` is **0.019185 nats per unit `ln V`**. Treating
each convention's spread as a monotone tilt over our grid's `ln V` range of 3.836:

| convention | tilt | factor below the M1 margin |
|---|---|---|
| add-0.5 | 5.93e−07 | 32,376× |
| add-1e−6 | 2.86e−06 | 6,699× |
| **drop EOS** | 3.06e−04 | **63×** |

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
   perturbs cross-vocabulary `L_u` differences by ≤1.1e−5 nats, at least 6,699× below what
   would move `θ` by the M1 margin.

This is already what `src/metrics.py` implements, so nothing changes in code; what changes
is that it stops being an open item and becomes a dated, justified, quantified choice. The
`OPEN ITEM` docstring should be replaced with a pointer to this record.

**Reported as a limitation:** `L_u` here is not guaranteed numerically comparable to their
published `Lossu` values, because the unigram estimator differs by an unknown amount. This
does not affect the study, which tests the *location of the optimum over V*, not absolute
loss — but it forbids any direct comparison of our `L_u` numbers against their table.

### P4 — KEEP the EOS separator, with a pilot-conditional gate on M2

The metric-side effect is bounded above: 1.17e−03 nats of cross-V spread, 63× below the M1
margin. Comfortable for M1.

**But that bound does not cover M2, and does not cover training.** Two gaps, stated
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
2M–16M, where curvature is unmeasured — that is, in part, what the study is for. If `L_u`
proves dramatically flatter in `V` below 33M, the 63× margin on P4 shrinks proportionally.
The pilot measures curvature in our regime directly and should be checked against the
0.04732 floor used here. The P2 margin (≥6,699×) has enough headroom that no plausible
flattening reaches it.
