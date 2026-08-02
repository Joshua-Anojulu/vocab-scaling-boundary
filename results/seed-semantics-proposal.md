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
2. `TrainResult` records a witness that makes nesting auditable from the artifacts alone.
   The realistic way to break nesting is a caller slicing the token array to budget before
   constructing the stream: the permutation is then drawn over a smaller domain, the `1.1·C`
   arm stops nesting its `C` arm, and **nothing in the loss curves would show it.** A first
   version recorded only `stream_sequences`, which the reviewer correctly rejected as
   insufficient — it cannot distinguish a different token array of the same length, a
   different same-length slice, or a change in the permutation algorithm itself. The
   recorded witness is now `order_seed`, `stream_sequences`, `sequences_consumed`,
   `tokens_digest` (content, not length), `order_digest` (the consumed permutation prefix),
   and `numpy_version`.

   On that last field: `np.random.default_rng(seed).permutation(n)` is **not promised stable
   across NumPy versions**, and this design must not claim it is. Nothing here needs
   cross-version reproducibility — the permutation only has to be fixed within a
   `(vocabulary, seed)` group, and a group is produced by one process — but a run repeated
   after an upgrade may read a different order, and `order_digest` is what makes that
   visible instead of silent.
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
inherit — and the confirmatory runs would be sized by a number **known to estimate the wrong
variance component**, whose direction of error is expected but not proven. The pilot would
have to be re-run. **This must be settled before B.7, not before the confirmatory runs.**

## What must be recorded either way

Whichever way this is decided, the pilot must report the two variance components separately
where possible, so the eventual paper can state what its replication does and does not
cover. If the proposal is adopted, that is initialisation + data order; if not, the
limitation must be stated in the direction of harm, not neutrally.


---

## The fix broke the unigram baseline, and that had to be fixed too

Found by the reviewer, not by the author, and it is the most consequential item here because
it lands on the primary metric.

`L_u = CE_model − H_unigram`, and amendment A3 fits `H_unigram` on **each config's consumed
train prefix** — implemented as `train_tokens[:consumed]`. That implementation was correct
only because consumption was sequential. **Permuting sequence order makes the consumed set
scattered, so a prefix fit stops describing what the model read.** The seed-order fix would
have introduced a defect into `L_u` while repairing one in the variance estimate.

Two distinct harms, and the second is the one that actually forces the fix:

1. **Bias.** The baseline is fitted on text the model never read.
2. **A deleted variance component.** The prefix fit is *identical across seeds* while the
   model's training set is not. It would have set the baseline's contribution to seed
   variance to exactly zero — in the metric whose seed variance Stage B.7 exists to
   estimate, inside the amendment that exists to stop exactly that kind of silent
   variance deletion.

**Fixed:** `unigram_logp` takes the run's consumed sequence order and fits on those
sequences' targets. `TokenStream.consumed_order` supplies it, `evaluate_run` accepts it, and
omitting it for a permuted run is the caller error the guards are there to catch.

**Measured, not assumed** — `scripts/unigram_order_sensitivity.py`, on real tokenized pilot
data, eight seeds, evaluation targets held identical so only the fit set moves:

| quantity | worst over the V grid |
|---|---|
| bias in `H_unigram`, prefix vs correct | **9.86e−05 nats** (V=3200) |
| between-seed SD of `H_unigram` | **7.47e−05 nats** |
| local slope in `ln V` | **1.26e−04 nats per ln V** |
| implied argmin displacement | **1.31e−02 in ln V** |
| M1 margin (`ln 1.5`) vs that displacement | **31×, 95% CI 19.9×–44.9×** |

**Quote the interval, not the point estimate — the point estimate is not stable.** At three
seeds this read 23.0×; at eight it reads 31×. The slope is a finite difference of
seed-*averaged* biases, so its noise is set by `seed_sd/√n` and is not small next to the
slope itself. The bootstrap resamples the seed vector jointly, matching the study's own
seed-level block bootstrap. An earlier draft of this section quoted **23.0×** as though the
last digit carried information; it did not.

The defensible reading is the one the interval supports: **the displacement sits at least
about twenty-fold inside the M1 margin.** That is comfortable, and it is still the
second-tightest margin measured in this study — looser than P2's smoothing convention at
57×, tighter than everything except P4's EOS separator at 8.7×.

**The margin is not why the fix is required**, and this is not a justification retrofitted
after a comfortable number came back — it is the argument that was made when the defect was
found. A margin argument cannot reach the second harm at all. A variance component set to
zero by construction is not a small error in an estimate; it is a different estimand, and no
margin makes it the right one. That is the same objection this amendment makes about
initialisation-only seeds, and it would have been self-defeating to fix one while
introducing the other. The bias measurement bounds how much the *first* harm would have
cost; it says nothing about the second, which is the harm that forces the change.

---

## A second defect found in the same review

`TrainConfig.total_steps` ceiled the step count while its docstring claimed "the last step is
trimmed." Nothing trimmed it, and the loop consumed full batches every step, so every run
**overshot** its token budget by up to `tokens_per_step - 1`.

This matters for the same reason P4 does. IsoFLOP here is enforced on exact token counts, so
an overshoot is a direct, `V`-dependent perturbation of `C`. Across the six pilot
configurations the overshoot reaches **+0.0129%** at `micro_batch=4, grad_accum=4` (worst
cell `V=12672`), and **+0.0845%** at `8/8` — the larger effective batch the VRAM pilot may
select. Converted through the IsoFLOP slope (`dL_u/d(lnC)` about -1.08 from Tao's own
points), +0.0129% is roughly 1.4e-4 nats: an order below the P4 metric-side spread, but of
the same character, and free to remove.

**Fixed.** The budget is now the largest whole number of sequences that does not exceed
`T_target`, and the final step is genuinely trimmed — with gradients weighted by each
micro-batch's share of the step's sequences, so a short final step neither carries less
weight than it should nor more. The error becomes an **undershoot of at most one sequence**:
-0.0011% worst case across the six configurations, at both batch shapes, always
conservative, with `consumed_tokens` reported exactly rather than assumed equal to target.

**The figures are re-derivable, and the reason matters.**
`scripts/budget_convention_error.py` emits `results/budget_convention_error.json` with
per-cell `target_tokens`, `old_tokens`, `new_tokens` and both error columns, deriving `C`
and the vocabulary grid from `src/` rather than from a transcribed literal. That is not
housekeeping: a draft of this section briefly reported **+0.0134% / -0.0010%**, computed
against `C` rounded to `1.03084e16`. The rounding is a relative change of **4.6e-06**, but
the per-cell overshoot depends on `T_target mod tokens_per_step`, so it is *chaotically*
sensitive to `C` and the worst cell moved by 4%.

The lesson is not "round more carefully." It is that **the per-cell overshoot is not a
stable quantity and should not be the thing anyone reasons from.** The stable quantity is
the bound `(tokens_per_step − 1) / T_target`, which varies smoothly and bounds every cell.
The exact figures are reported because they are cheap to regenerate, not because a reader
should rely on their last digit.

Two smaller defects travelled with it, both artifacts of the same assumption that every step
is full width. The throughput window added `cfg.tokens_per_step` per step regardless of
trimming, crediting a short final step at full width and inflating the last window — and
those windows are what the Stage B runtime projections are built from. And `TrainResult.steps`
reported the planned `total_steps` rather than the steps actually taken. Both now report what
happened rather than what was planned.
