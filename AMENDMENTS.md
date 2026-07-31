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

### P2 — Unigram smoothing convention for `L_u`

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

### P4 — Document packing convention (EOS separator)

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
