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
