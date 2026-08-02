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

**The precise claim, corrected.** An earlier draft said initialisation-only variance is a
lower bound on run-to-run variance and that every consequence runs the same way. That is
overstated and is withdrawn. A fixed-order conditional variance is not mathematically
guaranteed to be below the order-averaged variance for every statistic, and order effects
can cancel or amplify in paired contrasts such as `D`.

What is defensible, and sufficient: **the pilot would estimate the wrong variance
component.** The estimand the design calls for is joint over initialisation and data order;
initialisation-only estimates something else, and the BCa intervals, TOST, and the power
calculation are all built on it. In the usual case this narrows intervals and makes
equivalence TOST anti-conservative — the equivalence claim being the study's most likely
positive result — but that direction is the expectation, not a proof.

And it lands squarely on the study's central credibility claim. The strongest defensible
thing about this work is that Tao et al. ran one run per cell with no seed replication while
this study replicates. **A replication whose replicates differ only in initialisation is a
weaker instrument than that claim implies**, and a reviewer who reads `train.py` will see it.

## What the reference does, and why it does not rescue us

An earlier draft of this section said the reference shuffles within the `PackedDataset`
buffer. **It does not, on the path actually used.** `create_dataloaders` calls
`create_dataloader(..., shuffle=False, ...)` at `reference/tinyllama_pretrain.py:488-496`,
with the comment "shuffle=False for the data-constraint experiments," so the buffer shuffle
is inactive. What remains is the unconditional filename shuffle at `:446-448` under
`random.seed(12345)` — **hardcoded**.

So Tao's data order is a deterministic permutation of files, read sequentially within them,
identical across all their runs. They varied nothing, and ran one run per cell.

Matching them on determinism is therefore faithful. It is also irrelevant: the problem is
not fidelity to their pipeline, it is whether *our* variance estimate supports *our*
inference. It does not, and no property of their code changes that.

## Proposed resolution

**Seeds vary initialisation AND data order.** Concretely: permute the SEQUENCE order of the
full token array with a per-seed RNG, then read a prefix of that permuted order.

**Sequences must be self-contained, and an earlier draft of this proposal got that wrong.**
It said "permute the block indices," which with the then-current `TokenStream` would have
concatenated permuted blocks and formed shifted targets across the joins — inventing one
false next-token pair per block, about 0.049% of targets at B=2048. That is the same order
as the EOS effects amendment A4 already treats as decision-relevant, so it would have been a
real artefact rather than a rounding detail.

The implemented form takes sequence `k` as `tokens[k*B : k*B + B + 1]`: `B` inputs plus the
one lookahead token supplying its final target. Each sequence carries its own targets, so
reordering can never manufacture a pair absent from the corpus. This is exactly what the
reference does — `create_dataloaders` requests `effective_block_size = block_size + 1` for
this reason — so the fix moves *toward* the reference rather than away from it.

**Implementation contract, which the guarantees depend on:**

* One permutation per `(vocabulary, seed)`, drawn over the whole array, reused for every
  budget and both arms — in particular `V_run@C` and `V_run@1.1C` must share it, or M2's
  matched-seed pairing is broken.
* Bootstrap resampling keeps the complete six-configuration seed vector as one block.
* Amendment A1's wording that "every run reads the same documents in the same sequence" is
  **explicitly superseded across seeds**: it now holds within a seed, not globally. That is
  the point of the change, and it must be stated rather than quietly relaxed.

**The contract is enforced in code, not just written down.** An earlier version of this fix
left `order_seed` for the caller to pass, so a confirmatory run that simply omitted it would
have read corpus order and silently reproduced the very defect being repaired — a contract
living in prose while the code satisfied it by accident, which is the original bug exactly.
Three guards now close that:

1. `train_run` **refuses** a stream whose `order_seed` is not the run's `seed`. Reading
   corpus order requires setting `unseeded_order_ok=True`, making it a declaration rather
   than an omission. Benchmark and smoke runs set it; confirmatory runs cannot.
2. `TrainResult` records `order_seed` and `stream_sequences` — the size of the permuted
   domain. The realistic way to break nesting is a caller slicing the token array to budget
   before constructing the stream: the permutation is then drawn over a smaller domain, the
   `1.1·C` arm stops nesting its `C` arm, and **nothing in the loss curves would show it.**
   Requiring equal `stream_sequences` across a `(vocabulary, seed)` group makes it auditable
   from the artifacts alone.
3. The nesting test previously compared two whole streams built from the same array and
   seed, which is tautologically true and could not fail. It now reads two different budgets
   and asserts the smaller is a strict prefix of the larger, and a companion test exercises
   the slicing failure directly.

This is chosen over the alternatives for a specific reason — it **preserves the
preregistered nesting property**. `PLAN.md` requires the ordered corpus be held fixed so
that "runs at different budgets see the same tokens in the same order, differing only in how
far they read." Permuting the whole array once per seed and reading a prefix keeps exactly
that: at a given seed, the `1.1·C` run reads the `C` run's sequences plus more, in the same
order. Matched-seed pairing for M2 is preserved, and the seed-level bootstrap block
structure is unchanged.

Rejected alternatives:

* *Different starting offset per seed* — infeasible. The token arrays were sized to budget
  and carry only ~2% headroom (e.g. V=768: 201,225,884 available against 197,280,278
  needed). An offset runs off the end. Permutation has no such problem: it reorders the
  same sequences rather than requiring new ones.
* *Leave it, record as a limitation* — rejected because the pilot would estimate the wrong
  variance component, and a limitation note does not repair an interval built on it. The
  expected direction is anti-conservative on the study's most likely positive result, which
  raises the stakes, but the objection does not depend on that direction holding.
* *Shuffle within a prefetch buffer* — this is what `PackedDataset` supports, but it is
  inactive on the reference's own path (`shuffle=False`), so it is neither the faithful
  choice nor a useful one: a buffer shuffle is local, and their seed is fixed regardless.

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


---

## A second defect found in the same review

`TrainConfig.total_steps` ceiled the step count while its docstring claimed "the last step is
trimmed." Nothing trimmed it, and the loop consumed full batches every step, so every run
**overshot** its token budget by up to `tokens_per_step - 1`.

This matters for the same reason P4 does. IsoFLOP here is enforced on exact token counts, so
an overshoot is a direct, `V`-dependent perturbation of `C`. Across the six pilot
configurations the overshoot reaches **+0.0134%** at `micro_batch=4, grad_accum=4` (worst
cell `V=12672`), and **+0.0850%** at `8/8` — the larger effective batch the VRAM pilot may
select. Converted through the IsoFLOP slope (`dL_u/d(lnC)` about -1.08 from Tao's own
points), +0.0134% is roughly 1.4e-4 nats: an order below the P4 metric-side spread, but of
the same character, and free to remove.

**Fixed.** The budget is now the largest whole number of sequences that does not exceed
`T_target`, and the final step is genuinely trimmed — with gradients scaled by the
micro-batches actually taken, so a short final step does not silently carry less weight than
a full one. The error becomes an **undershoot of at most one sequence**: -0.0010% worst case
across the six configurations, at both batch shapes, always conservative, with
`consumed_tokens` reported exactly rather than assumed equal to target.

**Both figures are re-derivable, not quoted.** `scripts/budget_convention_error.py` emits
`results/budget_convention_error.json` with per-cell `target_tokens`, `old_tokens`,
`new_tokens` and both error columns. An earlier draft of this section said **+0.0129%** and
**-0.0011%**; neither reproduces under any rounding convention, and both were rounded
printed values quoted back as measurements. That is the fourth instance of this specific
failure in this project — after the 9.0×/8.7× and 37×/36.6× corrections — which is why the
numbers now live in an artifact that regenerates rather than in prose.

Two smaller defects travelled with it, both artifacts of the same assumption that every step
is full width. The throughput window added `cfg.tokens_per_step` per step regardless of
trimming, crediting a short final step at full width and inflating the last window — and
those windows are what the Stage B runtime projections are built from. And `TrainResult.steps`
reported the planned `total_steps` rather than the steps actually taken. Both now report what
happened rather than what was planned.
