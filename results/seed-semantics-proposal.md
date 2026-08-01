# Proposed: what a "seed" varies, and why it blocks Stage B.7

**Status: proposal.** `PLAN.md` is approved-final and is not edited. Found while building the
pilot runner, before any pilot run.

---

## The defect

`PLAN.md` specifies BCa bootstrap "resampling **at the seed level within scale** (seeds are
the exchangeable unit; vocabulary points are fixed design)". It never says what a seed
*varies*. The implementation answers that question by accident:

* `src/train.py:143` — `torch.manual_seed(cfg.seed)`, the only use of the seed.
* `src/train.py:105` — `TokenStream` is "sequential, non-repeating ... with no shuffling".

So **a seed varies model initialisation only. Every seed reads identical tokens in identical
order.**

## Why this is not a detail

The seed-level variance is the input to everything inferential in this study:

* BCa intervals are built by resampling seed clusters.
* The Stage B.8 power simulation converts that variance into `n`, the seed count for all
  139 confirmatory runs.
* M1-equiv is a **TOST**, which rejects when the interval is *narrow enough* to sit inside
  `±ln 1.5`.

Variance estimated from initialisation alone is a **lower bound** on run-to-run variance,
because it excludes data-order variation. Three consequences, all in the same direction:

1. BCa intervals are too narrow.
2. TOST becomes **anti-conservative** — it declares "the prediction is practically right"
   more readily than the evidence supports. The equivalence claim is the study's most
   likely positive result.
3. The computed `n` is too small, so the confirmatory runs are underpowered against the
   variance that actually obtains.

And it lands squarely on the study's central credibility claim. The strongest defensible
thing about this work is that Tao et al. ran one run per cell with no seed replication while
this study replicates. **A replication whose replicates differ only in initialisation is a
weaker instrument than that claim implies**, and a reviewer who reads `train.py` will see it.

## What the reference does, and why it does not rescue us

`reference/tinyllama_pretrain.py:438` shuffles filenames and shuffles within the
`PackedDataset` buffer — but with `random.seed(12345)` and `seed=12345+global_rank`,
**hardcoded**. Tao's data order is shuffled yet deterministic, identical across all their
runs. They varied nothing, and ran one run per cell.

So matching them on determinism is faithful. It is also irrelevant: the problem is not
fidelity to their pipeline, it is whether *our* variance estimate supports *our* inference.
It does not, and no property of their code changes that.

## Proposed resolution

**Seeds vary initialisation AND data order.** Concretely: permute the block indices of the
full token array with a per-seed RNG, then read a prefix of that permuted order.

This is chosen over the alternatives for a specific reason — it **preserves the
preregistered nesting property**. `PLAN.md` requires the ordered corpus be held fixed so
that "runs at different budgets see the same tokens in the same order, differing only in how
far they read." Permuting the whole array once per seed and reading a prefix keeps exactly
that: at a given seed, the `1.1·C` run reads the `C` run's blocks plus more, in the same
order. Matched-seed pairing for M2 is preserved, and the bootstrap block structure is
unchanged.

Rejected alternatives:

* *Different starting offset per seed* — infeasible. The token arrays were sized to budget
  and carry only ~2% headroom (e.g. V=768: 201,225,884 available against 197,280,278
  needed). An offset runs off the end.
* *Leave it, record as a limitation* — the bias is in the anti-conservative direction on
  the study's most likely positive result. A limitation note does not repair an interval
  that is too narrow.
* *Shuffle within a buffer, as the reference does* — reproduces their partial shuffling
  without the property we need, since their seed is fixed anyway.

**Cost: zero GPU-hours.** Same tokens, same count, different order. It is a change to
`TokenStream`, not to the budget.

## Why this blocks Stage B.7 specifically

The pilot's entire purpose is to estimate the variance that sets `n`. Running it under
initialisation-only seeds would produce a variance estimate that the confirmatory runs then
inherit — and the confirmatory runs would be sized by a number known to be too small. The
pilot would have to be re-run. **This must be settled before B.7, not before the
confirmatory runs.**

## What must be recorded either way

Whichever way this is decided, the pilot must report the two variance components separately
where possible, so the eventual paper can state what its replication does and does not
cover. If the proposal is adopted, that is initialisation + data order; if not, the
limitation must be stated in the direction of harm, not neutrally.
