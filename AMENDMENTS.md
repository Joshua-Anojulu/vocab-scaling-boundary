# Amendments to the preregistration

`PLAN.md` is deliberately **not edited**. It carries `status: approved-final` with
`final_body_sha256 = 970a523f…`, and rewriting the body would break the binding between
that hash and what the reviewer actually approved. Preregistrations are amended by
appending a dated record, not by revision.

Each entry states what changed, why, whether the trigger was external or discretionary,
and whether it was adopted.

---

## A1 — Corpus source (ADOPTED, forced) — 2026-07-30

**Preregistered:** `cerebras/SlimPajama-627B`, `train` split, `chunk1`, shards by
ascending index, RNG seed 1234.

**Adopted:** `DKYoon/SlimPajama-6B`, `train` split.

**Why — the preregistered corpus no longer exists.** Authenticated to the Hub as
`JoshuaAnojulu`, `HfApi().dataset_info("cerebras/SlimPajama-627B")` returns
**404 `RepositoryNotFoundError`**. The unauthenticated 401 seen earlier was the Hub
masking existence from anonymous callers, not a terms gate — with a valid token it
resolves to a plain 404. The `cerebras` org currently hosts four datasets, none of them
SlimPajama. There is nothing left to request access to.

**Why not a chunk1 mirror.** Two independently uploaded mirrors exist,
`rokset3/slim_pajama_chunk1` and `UltraRonin/SlimPajama-chunk1`. They were cross-checked
against each other over their first 200 documents:

```
rokset3    : 200 docs, 782,330 bytes, sha256 4941d27b7fc33549…
UltraRonin : 200 docs, 604,396 bytes, sha256 300f58442c379195…
matching documents: 0 / 200
```

They disagree completely, so neither can be validated against the other and neither's
fidelity to the original ordering is establishable by any check available. A nominal
claim of being "chunk1" is not evidence of being chunk1 in the preregistered order.

**Why this source.** It remains within SlimPajama, which matters because the law under
test was fitted on SlimPajama and a different corpus would make the corpus itself a
confound. It is the most-used SlimPajama derivative on the Hub by a wide margin (13,536
downloads, 64 likes, versus 342 and 419 for the mirrors), so a third party reproducing
this study can obtain the same bytes.

**Consequences to carry into reporting:**

* Corpus order is that of `DKYoon/SlimPajama-6B`, not chunk1 shard order. The plan's
  requirement that the *ordered corpus be held fixed across budgets* is preserved — every
  run reads the same documents in the same sequence — but the sequence is not the
  preregistered one.
* The subset is a sample of SlimPajama, so per-domain proportions may differ from chunk1.
  This should be stated as a limitation rather than assumed harmless.

---

## A2 — Tokenizer-fit sample size (ADOPTED, consequential) — 2026-07-30

**Preregistered:** 2 GB sampled from the tokenizer-fit split.

**Adopted:** 150 MB.

**Why.** The tokenizer-fit split is 2% of buckets by construction. Against a ~24 GB
corpus that ceiling is ~480 MB, so 2 GB is unreachable; and filling even 400 MB would
require streaming ~20 GB, essentially the whole dataset, to satisfy a cap that BPE does
not need. 150 MB requires a ~7.5 GB scan and still provides orders of magnitude more
merge candidates than a 17,792-token vocabulary can consume.

**Guard.** `src/tokenizer.py` raises `TRAINING CORPUS EXHAUSTED` if BPE cannot reach the
requested vocabulary, so an inadequate sample fails loudly rather than silently yielding
`V_tok != V_head`. This is not a theoretical guard — it fired during development on a
corpus that was genuinely too small.

---

## PROPOSED, NOT ADOPTED

Recorded so the decisions are visible and dated, and so that adopting one later is
demonstrably not a post-hoc reaction to results.

### P1 — Checkpoint reuse for the M2 arm (~13 GPU-hours)

Round 2 of review blocked intermediate-checkpoint reuse because "under cosine decay, a
checkpoint taken partway through a long run has a different learning-rate history from a
model intentionally trained to that lower compute budget."

Reading `reference/tinyllama_pretrain.py` shows Tao's schedule is **linear warmup then
constant** — `get_lr` returns the base rate unchanged after warmup, and `min_lr = 4e-5`
is defined and referenced nowhere. Under a constant rate, a checkpoint at step *k* has
exactly the learning-rate history of a run trained to step *k*, so the objection does not
apply and reuse is sound.

**Not adopted** because the preregistration specifies separate budget-specific runs and
the saving became visible only after the budget was measured. If taken, it must be
recorded here *before* any confirmatory run, noting that the trigger was reading the
reference schedule rather than the cost.

### P2 — Unigram smoothing convention for `L_u`  ·  **CLOSED, superseded by A3**

The plan requires matching Tao's exact smoothing, special-token, boundary and
zero-frequency conventions. Their released code does not expose the unigram construction.
`src/metrics.py` currently uses add-1 over the full vocabulary as an explicit documented
default. **Unresolved.** Must be reconciled, or the choice fixed here with its rationale,
before any confirmatory run.

### P3 — Shrink-rule ordering

The preregistered shrink rule sheds architectures from the middle, but 16M is 56% of the
measured 89.1 GPU-hour budget, so the rule saves little (5.6 h, then 25.6 h). Dropping or
de-prioritising 16M would save ~50 h in one step at the cost of the scale nearest the
anchor. **Not adopted** — reordering after seeing costs is exactly the post-hoc freedom
the preregistration exists to prevent.

### P4 — Document packing convention (EOS separator)  ·  **CLOSED, superseded by A4**

Flat token streams need a document boundary marker or the model learns to run one
document into the next. `src/tokenize_corpus.py` inserts `<eos>` after every document.

The plan specifies three special tokens (BOS, EOS, PAD) but not their use, and Tao's
released code does not expose its packing convention. EOS-separation is the standard
choice for packed pretraining and is what is implemented.

**Why it is not free.** It adds exactly one token per document, which changes the token
count, and therefore changes both the realized fertility and the denominator of `L_u`.
At the measured document sizes (~4.4 KB mean) the effect is small — order 0.05% of tokens
— but it is not zero, and it interacts with P2: the unigram distribution that `L_u`
normalises against will contain that EOS mass.

**Status: implemented, unreconciled.** Like P2, this is a convention that must be either
matched to theirs or fixed here with its rationale before any confirmatory run. Recorded
now so the choice is dated and visible rather than implicit in the code.

---

## A3 — Unigram smoothing convention for `L_u` (ADOPTED, discretionary) — 2026-08-01

**Supersedes P2**, which is closed.

**Preregistered:** match Tao's exact smoothing / special-token / boundary / zero-frequency
conventions.

**Adopted:** add-1 (Laplace) over the full vocabulary, unigram fitted on the **training
split only** — specifically each vocabulary's own tokenized train array, which is the prefix
that configuration consumes. This is what `src/metrics.py` already implements; what changes
is that it is now a dated, justified choice rather than an open item.

**Why matching is impossible, not merely unattempted.** `reference/tinyllama_pretrain.py`
does expose the `L_u` computation — `validate_pplu` at lines 406–435 — which **confirms the
definition and sign convention** used here (`L_u = CE_model − H_unigram`, negative when the
model beats unigram). The earlier record in P2 that "their released code does not expose the
unigram construction" was half wrong and is corrected.

What the release does *not* determine is the estimator. The lookup table is loaded from an
external file that is not published, via a loader with two defects:

* `json.load` returns string keys, so `max(keys()) + 1` raises
  `TypeError: can only concatenate str (not "int") to str`. Verified directly. The released
  script cannot run on an ordinary JSON object without patching keys to integers.
* `torch.empty`, not `torch.zeros` — any id absent from the dict retains uninitialised
  memory as its unigram probability, so no zero-frequency convention is representable.

So the exact smoothing and zero-frequency convention is **not recoverable from the release**.
This is a claim about what the release determines, not about what the authors ran.

**Why a convention is required.** Zero-frequency evaluation tokens occur in 3 of 20
vocabularies (V=384: ids 144/145, 36 instances; V=3456: id 1868, 1 instance; V=6912: ids
1770/1868, 4 instances). Unsmoothed MLE would be undefined.

**Why the choice cannot change a conclusion.** Smoothing enters `L_u` only through
`H_unigram`, so its entire effect is computable without training anything. Both M1 and M2
compare `L_u` across vocabularies, so what matters is the local slope of the induced shift
in `ln V`, against the curvature of `L_u` about its minimum. Measured in
`results/p2_smoothing_sensitivity.json` and `results/p2p4_decision_relevance.json`:

* Every smoothing strength from add-1 to add-1e−6 agrees to within **1.1e−5 nats** of
  cross-vocabulary spread; max local slope 6.87e−05.
* The slope that would displace `θ` by the M1 margin `ln 1.5` is **3.913e−03**, using the
  most pessimistic curvature recoverable from Tao's data (0.00965, the minimum across three
  fitting windows over all six of their `N_nv` families).
* Margin: **≥57×**, degrading to **36.6×** when every vocabulary is refitted on a common
  33M-token prefix, the smallest `T_target` in the grid.

**Conditional, and it is not decorative.** That curvature is measured at `N_nv ≥ 33M`; this
study runs at 2M–16M, which is in part what the study is for. Stage B.7 must record its own
measured `d²L_u/d(lnV)²`. **If the pilot's curvature falls below 0.00965, this conversion is
invalid in our regime and both A3 and A4 must be re-derived against the pilot's value before
any confirmatory run.**

**Reported as a limitation.** `L_u` here is not guaranteed numerically comparable to their
published `Lossu` column, because the unigram estimator differs by an unknown amount. This
does not affect the test, which is about the *location of the optimum over V*, but it
forbids direct comparison of absolute `L_u` values against their table.

---

## A4 — Document packing convention: EOS separator (ADOPTED with a gate) — 2026-08-01

**Supersedes P4**, which is closed.

**Adopted:** keep the `<eos>` separator inserted after every document, as
`src/tokenize_corpus.py` implements. Removing it would require retokenizing 5.1B tokens.

**Metric-side effect, measured.** Dropping EOS from both the fitted unigram and the
evaluation stream shifts `H_unigram` by 2.8e−04 to 1.5e−03 nats depending on `V`, a
cross-vocabulary spread of **1.17e−03** with max local slope 4.48e−04 — a margin of only
**8.7×** against the M1 margin, versus ≥57× for A3. On the 33M prefix it is **9.4×** —
essentially unchanged. An earlier draft attributed this to the document rate being fixed by
a prefix; that is wrong and was withdrawn in review round 3 (33M tokens covers 13,137
documents at V=384 and 30,185 at V=17792). The insensitivity is empirical. **What it
establishes is that the margin does not improve when the fit set changes, so more data will
not fix it.**

**Gate on M2, preregistered before the pilot runs.** M2 has no equivalence margin — it is a
sign test on `D` — so a 1.17e−03 nat perturbation flips it whenever `|D|` is that small.
Stage B.7 runs 3 seeds with the `1.1·C` configuration matched-seed inside the same bootstrap
block, so a **paired** interval for `D_pilot` is available by construction. Let
`B = 1.2e−02` nats, ten times the measured metric-side spread.

> **Pass** only if the paired 95% interval for `D_pilot` lies **wholly outside** `[−B, +B]`.
> Otherwise the packing convention can flip the M2 sign test and **must be resolved before
> any confirmatory run.**

A point estimate is explicitly insufficient: with 3 seeds, `|D̂_pilot|` above `B` can arise
from a true `D` inside the band, passing a point-estimate gate while leaving the sign
undetermined.

**What passing establishes — and what it does not.** It bounds the metric side only.
Removing EOS would also change the training token stream and hence `CE_model`, and that
effect is measured nowhere in this study. On a pass the correct statement is the weaker one:
*metric-side P4 is not capable of flipping the M2 sign test at the observed effect size; EOS
remains a preregistered convention and a reported limitation, and its training-side effect
is unmeasured.* Establishing more would require a paired no-EOS training arm, which is not
budgeted. **Declining to run it is a scope decision, recorded here rather than left
implicit.**

---

## Review provenance for A3 and A4

Both were submitted for adversarial cross-model review before adoption, because they fix the
definition of the primary metric and touch the headline novelty claim. Three rounds; full
record in `PLAN-REVIEW-LOG.md`. Round 1 returned 3 blocking and 4 advisory findings, all
accepted and applied, including a correction of fact from the reviewer: `N_v = V·d` is Tao's
analytical proxy justified by output-layer FLOP dominance, **not** a definition of the output
head. Round 2 returned 1 blocking (already fixed in-tree when it landed; the review clone
predated the fix) and 1 advisory. The author independently found and corrected two
order-of-magnitude errors in the conversion before the reviewer returned, and added two
robustness checks the reviewer did not request.

`PLAN.md` was not edited at any point.

---

## A5 — Checkpoint reuse: NOT ADOPTED, final — 2026-08-01

**Closes P1**, and tightens P1's own deadline, which was wrong.

**Decision: not adopted.** Every run is a separate, budget-specific training run. The ~13
GPU-hour saving is declined.

**P1's stated deadline was too weak, and this is the substantive part of A5.** P1 said the
choice "must be recorded before any confirmatory run." That is insufficient. Stage B.7 is
itself a paired experiment: it runs the five vocabulary points at `C` **and** `V_run = 3200`
at `1.1·C`, and `D̂_pilot = L_u(V_run, 1.1C) − L_u(V*, C)` is the difference between them.
Under checkpoint reuse the `C` observation could be an intermediate checkpoint of the
`1.1·C` run rather than an independent run. So the choice changes **what `D̂_pilot`
measures**, and `D̂_pilot` parameterises the M2 power targets that set `n` for all 139
confirmatory runs.

Deciding after the pilot would therefore produce confirmatory runs whose `D` is generated by
a different procedure from the pilot `D` that sized them. **The deadline is before Stage
B.7, not before the confirmatory runs.** Recorded now, before the pilot has been run.

**Why not adopted.** The preregistration specifies separate budget-specific runs. P1
established that the round-2 objection to reuse was wrong on the facts — Tao's schedule is
linear warmup then constant (`get_lr` returns the base rate unchanged after warmup;
`min_lr = 4e-5` is defined and never referenced), so under a constant rate a checkpoint at
step *k* does have the learning-rate history of a run trained to step *k*. Reuse is sound.

It is declined anyway, for two reasons that are about accuracy rather than cost:

1. The saving became visible only **after** the budget was measured. Adopting a
   procedure change whose trigger is a cost measurement is exactly the post-hoc freedom the
   preregistration exists to remove, even when the change is defensible on its merits.
2. Reuse introduces a dependence between the `C` and `1.1·C` observations that the M2
   bootstrap treats as paired-but-distinct. Independent runs keep the estimator's assumed
   structure and the actual data-generating process aligned. Soundness of the learning-rate
   argument does not by itself establish that the induced correlation is harmless.

**Trigger, recorded explicitly:** the reuse question was reopened by *reading the reference
schedule* in `reference/tinyllama_pretrain.py`, not by the measured budget. The budget
measurement determined only the size of the forgone saving.

**Consequence carried into reporting:** the study spends ~13 GPU-hours it did not have to.
That is stated as a deliberate design choice, not an oversight.

---

## A6 — what a seed varies (2026-08-01)

**Status: ADOPTED.** Supersedes A1 in part. Proposal and measurements:
`results/seed-semantics-proposal.md`. Review record: `PLAN-REVIEW-LOG.md`, seed-semantics
track, rounds 1–2.

### The gap in the preregistration

`PLAN.md` requires BCa resampling "at the seed level within scale (seeds are the
exchangeable unit; vocabulary points are fixed design)" but **never states what a seed
varies.** The implementation answered by accident: `torch.manual_seed(cfg.seed)` was the
seed's only use and the token stream was strictly sequential, so seeds varied model
initialisation while every seed read identical tokens in identical order.

### The amendment

**For seed-replicated pilot and confirmatory runs, a seed varies model initialisation AND
data order.** Concretely:

* The ordered corpus is fixed **within a `(vocabulary, seed)`**. One permutation of sequence
  order is drawn per `(vocabulary, seed)` over the WHOLE token array, and every budget reads
  a prefix of it. `PLAN.md`'s requirement that runs at different budgets "see the same tokens
  in the same order, differing only in how far they read" therefore holds within a seed.
* **Across seeds, data order and the consumed sequence subset vary by design.** This is the
  point of the amendment.
* The M2 `C` and `1.1·C` arms **must share the same `(vocabulary, seed)` permutation**, or
  matched-seed pairing is broken.
* The bootstrap resamples **the complete six-configuration seed vector as one block.**
* Sequences are self-contained — sequence `k` spans `tokens[k*B : k*B + B + 1]`, `B` inputs
  plus the one lookahead token supplying its final target — so reordering cannot manufacture
  a next-token pair absent from the corpus. This matches the reference, which requests
  `effective_block_size = block_size + 1` for the same reason.
* **`H_unigram` is fitted on the sequences the run actually consumed**, not on a contiguous
  prefix. A3's phrase "the consumed train prefix" described sequential consumption; under
  permutation the consumed set is scattered and a prefix fit would both bias the baseline
  and delete a real component of its seed variance.

### What A1 said, and what survives

A1 states that "every run reads the same documents in the same sequence." That is
**superseded in part**: it holds within a `(vocabulary, seed)`, not globally across seeds.
A1 is otherwise unchanged — the corpus, the split boundaries, and the tokenized arrays are
untouched by this amendment. Nothing is re-tokenized and no GPU time is added: same tokens,
same count, different order.

### What this amendment does NOT claim

Recorded explicitly, because each was either asserted and withdrawn during review or is a
claim the design does not need:

1. **Not** that initialisation-only variance is a lower bound on run-to-run variance, nor
   that every consequence runs in one direction. That claim was made in an early draft and
   is **withdrawn**. A fixed-order conditional variance is not mathematically guaranteed to
   sit below the order-averaged variance for every statistic, and order effects can cancel
   or amplify in paired contrasts such as `D`. The defensible claim, which is what justifies
   the amendment, is that initialisation-only seeds **estimate the wrong variance
   component**: the estimand the design requires is joint over initialisation and data
   order. Anti-conservative TOST is the *expected* direction, reported as an expectation and
   not as a proof.
2. **Not** that all runs read the same documents in the same sequence. See above.
3. **Not** that `np.random.default_rng(seed).permutation(n)` is stable across NumPy
   versions. It is not promised to be. The design does not require cross-version
   reproducibility — a `(vocabulary, seed)` group is produced by one process — but
   `numpy_version` and `order_digest` are recorded so a repeat after an upgrade is visible
   rather than silent.
4. **Not** that the unigram baseline is a contiguous consumed prefix. It is the consumed
   sequence set; see above.

### Enforcement, because a contract in prose is what failed here

The original defect was a requirement that lived in the preregistration while the code
satisfied it by accident. The same failure mode is closed structurally:

* `train_run` **refuses** a stream whose `order_seed` is not the run's `seed`; corpus order
  requires an explicit `unseeded_order_ok=True`, which no confirmatory run sets.
* `TrainResult` records `order_seed`, `stream_sequences`, `sequences_consumed`,
  `tokens_digest`, `order_digest`, `consumed_order_digest` and `numpy_version`, so nesting
  and matched-seed pairing are auditable from the artifacts without trusting the runner.

**The audit rule, stated so it is checkable rather than merely asserted:** two runs sharing
a `(vocabulary, seed)` are correctly nested **iff** they agree on `order_seed`,
`stream_sequences`, `tokens_digest` and `order_digest`, with `sequences_consumed` recording
how far each read. `tokens_digest` is a full content digest of the token array and
`order_digest` covers the **whole** permutation, not the consumed prefix — a `C` arm and a
`1.1·C` arm consume different amounts by construction, so prefix digests differ even when
nesting is perfect and could not decide the question. `consumed_order_digest` is a
descriptive record of what each run read, not the nesting witness.

### Reporting obligation

The pilot must report initialisation and data-order variance jointly, and state that it does
so. `PLAN.md`'s "what must be recorded" asks for the two components separately where
possible; the joint estimate is what the inference uses, and any separate reporting is
descriptive only.

---

## A7 — the effective batch size (2026-08-02)

**Status: PROPOSED**, pending adversarial review. Not adopted; no run has used it yet.

### The gap

`PLAN.md` specifies "the shared Tao training recipe" and "Tao's schedule" but **never fixes
the number of sequences per optimizer step.** Neither does A1–A6. It was found while
choosing a batch shape for the Stage B.7 runner, where it presented itself as a throughput
question and is not one.

The batch size is not free, because the released recipe PAIRS it with a learning rate. The
reference sets `global_batch_size = 512` sequences and `learning_rate = 4e-4` together
(`reference/tinyllama_pretrain.py:37,36`), and derives the micro-batch split from it:
`gradient_accumulation_steps = batch_size // micro_batch_size` (`:125`). Choosing a
different effective batch while keeping `4e-4` would be a different recipe wearing the same
name.

### The amendment

**The effective batch is 512 sequences per optimizer step, matching the reference.**
`micro_batch` is a memory-partitioning knob only, and `grad_accum` is derived as
`512 // micro_batch`; a micro-batch that does not divide 512 is refused rather than allowed
to land near it.

### Why this rather than the alternative, decided on measurement

The obvious objection is that at these budgets a 512-sequence batch yields few optimizer
updates: **131–190** across the six pilot configurations. That was the author's initial
concern, and it is answered by Tao's own released data rather than by argument.

At their smallest fitted scale (33M non-vocabulary parameters, 200 released runs) their runs
span **57 to 1,144 optimizer steps, median 601** (nominal; see the artifact caveat below).
This pilot's **131–190 updates** sit in the LOWER TAIL of that distribution — with only about 10–15% of their evaluations falling below
this study's range, and both endpoints below their 25th-percentile value; 0.21–0.30× their
median — above their minimum but not typical of their
grid. Recomputed in `tests/test_pilot.py` against the committed
`results/reference_step_stats.json`, so a future change that moves the study out of the low
tail fails a test.

**"Inside their range" overstates it, and the precise position is this:** their range spans
20×, so landing inside it is a weak test. Our step counts sit at the lower tail of their
smallest-scale evaluations: **about 10% of them fall below this study's shortest run and
about 15% below its longest**, and both endpoints sit below the 25th-percentile value of
their grid.

That is expected rather than alarming, and the reason is the study's own premise. Our pilot
budget is `C = 1.031e16` against their smallest-scale ladder of `1.272e16` to `5.940e17`:
**0.81× their smallest budget.** This study exists to probe *below* the region they fitted,
so running at fewer optimizer steps than most of their runs is the direct consequence of the
question being asked, not a defect in the recipe. The claim A7 rests on is the narrow one —
these step counts are not outside the regime the law was fitted in — and not the broader one
that they are typical of it.

The alternative — a small effective batch of 16 sequences, which is what a
throughput-first choice would have produced — gives ~6,000 optimizer steps, **10× Tao's
median at the comparable scale**, at a learning rate the released recipe pairs with a batch
32× larger. That is the larger departure from the recipe, not this. The initial framing of
the decision had it exactly backwards.

### What this amendment does NOT claim

1. **Not** that 512 is optimal for these budgets. It is faithful, which is the requirement
   here; the study tests Tao's law under Tao's recipe, and optimizing the recipe would
   confound the test.
2. **Not** that the resulting models are well converged in an absolute sense. They are
   trained under the reference's CONSTANTS. They are not trained under its trajectory -- see the
   quantified learning-rate-exposure limitation below, where the difference reaches 1.69x.
   "Same regime" is therefore not a phrase this study may use unqualified.
3. **Not** that micro-batch is inert. It changes throughput and peak memory, and it is
   chosen by measurement (`scripts/pilot_batch_probe.py`,
   `results/pilot_batch_probe.json`) — but it cannot change the effective batch, which is
   what the enforcement above guarantees.

### Reporting obligation

The paper must state the effective batch, the derived micro-batch split, and the optimizer
step counts alongside Tao's at the comparable scale. A reviewer's first question about a
study run at 1/130th of the reference's token budget will be whether the models were trained
comparably; the step-count comparison is the answer and belongs in the text rather than in a
repository.

---

### A7 — correction and expansion, before adoption (2026-08-02)

Round 1 of review returned five blocking findings. Two changed the amendment's substance and
one changed a training constant. A7 remains **PROPOSED**; nothing below has been run.

**The LR-coupling claim is withdrawn.** A7 argued the batch is not free because
`learning_rate = 4e-4` is "tuned to" a 512-sequence batch. The reference pairs the two but
gives **no evidence how `4e-4` was selected**, and the reviewer was right that this was
asserted rather than shown. The defensible claim is narrower and sufficient: **fidelity to
the released recipe.** The two values were used together in the runs the law was fitted on,
so changing one while keeping the other departs from that pairing — whether or not the
pairing was arrived at by tuning.

This is the fifth time in this project that a mechanism was asserted ahead of the evidence
for it. The others: the predicted fertility flattening, the tilt-is-worst-case argument, the
document-rate mechanism, and the transcribed `C`. The numbers have held up; the explanations
offered for them have not.

**Warmup was 8% and should be 10%, and this is not a rounding matter.** `WARMUP_FRACTION` was
derived from `tinyllama.py`'s module defaults (`warmup_steps=2000, max_step=25000`). The
script that actually produced the released IsoFLOP data,
`experiments/light_train/scripts/run.sh`, passes `warmup_steps=5480, max_step=54800` —
exactly **10%**. Verified against the upstream repository directly, not inferred from the
vendored copy. The defaults were never the experiment. `src/train.py` now uses `5480/54800`.

**A discovery that outgrew this amendment: Tao's IsoFLOP points are not separate runs.**
`compute_eval_steps(max_steps, evals_per_interval=20)` produces 20 linearly spaced in-training
evaluations, and `reference/exp_data.csv` contains **exactly 20 rows per (vocabulary, scale)**
at steps `57, 114, 172, … 1144` for the 33M family. Their IsoFLOP curve across compute is
therefore built from **checkpoints of a single run per configuration**, not from
budget-specific runs.

`PLAN.md` prescribes "separate budget-specific runs", which is a different procedure. The
consequence is recorded here rather than acted on, because it bears on an already-adopted
amendment and is not this amendment's to settle.

**It does, however, refute A5's stated reasoning.** A5 declined checkpoint reuse while
accepting that "under a constant rate, a checkpoint at step *k* does have the learning-rate
history of a run trained to step *k*." That equivalence **is false whenever warmup scales with
run length**, which it does: a dedicated run to step *k* warms up over `0.1k` steps, whereas a
checkpoint at step *k* of a 1144-step run warmed up over 114. A5's conclusion survives on its
*second*, independent ground — that reuse induces a dependence between the `C` and `1.1·C`
observations which the M2 bootstrap treats as paired-but-distinct — but its first ground is
withdrawn. A5's text is left intact and annotated here rather than rewritten, following this
project's convention for superseded reasoning.

**Consequence to carry into reporting, not to fix silently.** This study trains a dedicated
run per budget with warmup scaled to that run; Tao read intermediate checkpoints of a longer
run whose warmup was scaled to the longer run. At the low-compute end the learning-rate
trajectories therefore differ, and that is a procedural departure from the reference which
must be stated in the paper rather than left for a reader to discover in the code.

**Remaining round-1 corrections, applied.**

*What A7 is.* It is a **prospective clarification of a hyperparameter the preregistration
omitted**, not a restatement of something `PLAN.md` already fixed. `PLAN.md` says "the shared
Tao training recipe" and stops there. The effective batch and the warmup fraction are
therefore **unpreregistered researcher choices**, made before any outcome was observed and
recorded here for that reason. Nothing about the framing should suggest the plan settled
them; it did not, and the honest description is that this study is fixing them now, in
public, in advance.

*Step counts, corrected twice.* The pilot figure "130–189" was **floored whole batches, not
optimizer updates**. With the trimmed final step the actual update counts are **131–190**;
`tests/test_pilot.py` now computes them through `TrainConfig.total_steps` so the amendment
cannot drift from the code.

*Nominal versus artifact.* The reference figures "57 to 1,144, median 601" are **nominal**,
derived here as `num_characters · f(V) / (512 · 2048)`. The released checkpoint filenames for
the 33M family run `step-000060` to `step-001200`, median 630 — so the derivation and Tao's
own emitted artifacts do not agree exactly. Both are recorded in
`results/reference_step_stats.json`. The comparison uses the nominal values because they are
what this repository can recompute from released data; the artifact values are recorded so the
discrepancy is visible rather than buried. **The conclusion is unchanged under either**: the
pilot sits in the lower tail on both.

*Position, stated precisely.* The claim is **lower-tail but above the minimum** — roughly the
lower tail — about 10–15% of their evaluations fall below this study's range, both endpoints
sit below their 25th-percentile value, and the range is 0.21–0.30× the median. It is **not**
that the pilot is typical of
their grid, and range-inclusion is explicitly disclaimed as too weak to carry the decision:
their range spans 20×.

*Provenance of the step statistics.* `reference/exp_data.csv` is gitignored as regenerable,
which meant the test guarding these numbers **silently skipped in any clone lacking it**,
including the reviewer's. A test that guards a claim and does not run reads as coverage it
does not provide. The statistics now live in a committed artifact,
`results/reference_step_stats.json`, carrying the CSV's sha256 so the summary can be checked
against the source wherever the source is present.

*Micro-batch, measured.* `results/pilot_batch_probe.json` was regenerated against derived
`grad_accum`; the earlier file compared effective batches of 16 and 64, which this amendment
rejects and the pilot will never run at. Across micro-batches 2/4/8/16 at the matching
`grad_accum` of 256/128/64/32, **micro_batch=4 is fastest and second-cheapest in memory**:
9.48 h projected for 18 runs at 1.89 GB peak, against 10.26/10.56/10.96 h for the others.
Measuring at the real `grad_accum` changed the projection materially — the stale file
projected 13.04 h — because the optimizer step amortises very differently at `ga=128`.

### Decision: separate budget-specific runs are retained (2026-08-02)

**Decided by the author of the study**, after the discovery that Tao's IsoFLOP points are
intermediate checkpoints of one run per configuration rather than budget-specific runs.

**`PLAN.md` as written stands: one dedicated run per budget.** The alternative — reproducing
their procedure by training a longer run and reading checkpoints — would be closer to what
they did and would also be cheaper, and it is declined anyway, for three reasons:

1. `PLAN.md` prescribes separate runs. Changing a preregistered *procedure* partway through,
   on the basis of a discovery made while building the runner, is the precise freedom that
   preregistration exists to remove. The discovery is real and is recorded; acting on it is
   a different thing from recording it.
2. A5's second ground is **weaker than first stated and is demoted, not relied upon.** It
   held that reuse induces a dependence between the `C` and `1.1·C` observations which the
   M2 bootstrap treats as paired-but-distinct. Review pointed out that a seed-level paired
   bootstrap resampling the complete seed vector can preserve within-seed covariance, so
   reuse changes the data-generating procedure without automatically invalidating the
   bootstrap — and if it were a problem, it would likely be a fixable one. It is recorded as
   a consideration, not as a proof, and the decision does not rest on it.
3. The cheaper option being also the more faithful one is exactly the configuration in which
   a mid-study procedure change is least trustworthy, not most.

**The departure is therefore real and is carried into reporting rather than resolved, and it
is larger than "the histories differ" suggests.** This study trains each budget with warmup
scaled to that run's length; Tao read checkpoints of a longer run whose warmup was scaled to
the longer run. Quantified as cumulative base-learning-rate exposure — the sum of the LR
multiplier over all updates, which is what a linear warmup actually changes:

| updates | this study (warmup 10% of its own run) | checkpoint of a 1144-step run (warmup 114) | ratio |
|---|---|---|---|
| 131 | 124.0 | 73.5 | **1.69×** |
| 190 | 180.0 | 132.5 | **1.36×** |

So at the pilot's shortest configuration a model here receives **about 69% more cumulative
learning rate** than the reference procedure would have delivered at the same token budget.
That is not a rounding difference, and it means the phrase "the same regime as the reference"
must not be used without this qualification. The models are trained under the reference's
*constants*; they are not trained under the reference's *trajectory*.
This must appear in the paper's methods as a stated difference from the reference, not as a
detail left in the repository. It is a limitation of the comparison, and pretending the
procedures match would be the worse error.

**Round-2 corrections, applied.**

*Provenance of the 10% warmup, overclaimed and now stated honestly.* An earlier version said
`run.sh` "actually produced the released IsoFLOP data." **The repository does not show that:**
`exp_data.csv` predates the script in git history and the script loops over `vocab=4096`
only. So neither candidate is proven to be what generated the released data — 8% is a module
default that may never have been passed to anything, and 10% is the only warmup ratio the
project is on record as actually passing. **10% is chosen as the better-evidenced of two weak
options, and this study cannot claim to have matched Tao's warmup — only to have matched the
one value they published a script for.** That is the sixth time in this project a claim ran
ahead of its evidence.

*Withdrawn LR language, purged from the code as well as the prose.* Round 1 withdrew the
"tuned to" claim and it was corrected in `AMENDMENTS.md` only, while `src/train.py`,
`src/pilot.py`, `scripts/pilot_batch_probe.py` and `tests/test_pilot.py` kept asserting it.
The same failure as the stale docstring one track earlier: the argument was fixed in one
place and left standing in four. All now say the released recipe *pairs* the two.

*`lr_at`'s docstring corrected.* It still said an intermediate checkpoint has "exactly the
learning-rate history of a run trained to that step." That is false under 10% warmup scaling
with separate budget-specific runs, and is the same claim A5 was annotated for.

*Stale figures fixed in the PRIMARY text, not only in a later note.* `130–189` and "inside
that range" survived in A7's main body and in `src/pilot.py`'s user-visible docstring after
being corrected further down. A reader hits the wrong number first; appending a correction is
not correcting.

*A5's second ground, demoted.* See the decision record below — a seed-level paired bootstrap
can preserve within-seed covariance, so checkpoint reuse changes the data-generating
procedure without automatically invalidating the bootstrap. It is recorded as a
consideration, not a proof, and the retain-separate-runs decision does not rest on it.

**Warmup stability — the first check was invalid and has been withdrawn (2026-08-02).**

Round 2 asked whether 10% of a 131–190 step run — **13–19 warmup steps**, against Tao's ~114
— creates a problem of its own. The question is empirical and `scripts/warmup_stability_check.py`
was written to answer it.

**The first version did not test what its output was cited for.** It set
`warmup_fraction = real_warmup / real_total`, but `warmup_steps` is
`round(total_steps × warmup_fraction)` and `total_steps` there is the probe's 30 — not the
real 131–190. The actual warmup exercised was **3 steps**, while the amendment cited the
result as evidence about 13–19. The numbers were also written to `runs/diagnostics/`, which
`.gitignore` excludes, so the table cited evidence no reader could open — the same failure as
the test that silently skipped without `exp_data.csv`, two rounds after that one was fixed.

Both are corrected: the probe now takes the fraction against its own step count and
**asserts** `cfg.warmup_steps == real_warmup` before training, the pass criterion requires
loss *sustained* below chance rather than a transient dip, and output is committed to
`results/warmup_stability.json`. The numeric claim is withheld until that corrected run
completes; **no stability claim is made here on the strength of the invalid check.**
