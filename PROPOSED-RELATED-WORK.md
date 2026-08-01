# Proposed: Over-Tokenized Transformer related-work addition

**Status: PROPOSAL. `PLAN.md` is `approved-final` (7 rounds) and is not edited here.**
Target: Stage F related work, plus one qualifier on the novelty claim.

---

## (a) Draft paragraph

> **Relation to Over-Tokenized Transformer (arXiv:2501.16975).** Huang et al. report that
> enlarging the **input** vocabulary improves training loss log-linearly and, they argue,
> "generally" — explicitly framing this against the intuition that small models want small
> vocabularies. Read quickly, that appears to pre-refute the present hypothesis. It does
> not, because the two results are about different parameters.
>
> Over-Tokenized **decouples** the input vocabulary from the output vocabulary and locates
> its gain on the **input** side: an *n*-gram input embedding much larger than the output
> head, with the output head left small. Tao et al.'s law is a statement about
> `N_v = V·d` — **one** `V × d` table, not `2·V·d`. Their paper notes that vocabulary
> parameters would normally include both the embedding and the output layer, and adopts
> `V·d` as an analytical proxy on the grounds that output-layer FLOPs dominate; their
> released accounting is consistent with this, holding `N_nv` fixed as `V` varies across
> all six published families. The quantity the law governs is therefore an
> **output-weighted proxy for vocabulary allocation**, not the input-side capacity that
> Over-Tokenized manipulates. The present study holds `V_tok = V_head = V` by construction — one vocabulary
> serving both sides, untied weights, sizes divisible by the padding quantum so no rows go
> unused. It therefore tests the **vocabulary-allocation law under coupling**, and makes
> no claim about the decoupled input-side regime Over-Tokenized studies.
>
> The two results are in fact concordant on the output side. Over-Tokenized's own §3.1
> synthetic-CFG experiments, at ~2.4M and 85M parameters, find that a larger **output**
> vocabulary *harms* the smaller model — the same direction this study tests below 33M.
> Its headline "bigger is better" claim is an input-side finding, and the paper's own
> output-side evidence points the other way.

**Why this framing is defensible and not a dodge.** Three independent checks:

1. *Their own stated convention.* The paper says vocabulary parameters would typically
   include both embedding and output layer, and uses `V·d` for analytical simplicity
   because output-layer FLOPs dominate. So the output weighting is theirs, not our reading.
2. *Data.* `Non_vocab_parameters` is **invariant across all ten vocabulary sizes** at fixed
   `embed_dim`, in every one of their six families — so `N_nv` excludes both embedding
   tables, and `N_v = V·d` counts exactly one `V × d` table, not both. Verified directly.
3. *FLOP accounting.* Their budget `C = 6·(N_nv + V·d)·H·f(V)` charges `V·d` the full
   matmul cost, and only the output projection performs a matmul against the vocabulary;
   the input embedding is a row lookup. This is the same argument their footnote makes, and
   it holds independently of any code reading.

**An earlier draft of this document over-claimed here**, saying their code "defines `N_v` as
the output head only." It does not define it that way — `V·d` is a deliberate proxy whose
justification is FLOP dominance. The distinction matters, because a proxy can be a poor one
precisely where the study operates: at small `d` the input embedding is a larger share of
parameters, so `V·d` under-counts actual vocabulary parameters by a factor approaching two.
That is inherited deliberately — following their convention exactly is what makes the
comparison valid — but it must be reported, not assumed harmless.

---

## (b) Does this change H₀, M1/M2, or the architecture family?

**Confirmed: no design change.** Point by point:

| element | affected? | why |
|---|---|---|
| **H₀** | no | Stated over `N_v = V·d` — Tao's output-weighted vocabulary-allocation proxy — at A1-implied compute-optimal `C`. Over-Tokenized makes no claim about that quantity under coupling. |
| **M1** (`θ = ln(N_v*/N_v_pred)`) | no | Both terms are the same `V·d` proxy, so the proxy's known bias cancels in the ratio. The margin `ln 1.5` is calibrated to Tao's own grid spacing, which is defined in the same units. |
| **M2** (`D = L_u(V_run,1.1C) − L_u(V*,C)`) | no | A loss-regret statistic. Independent of how vocabulary is allocated between sides. |
| **Architecture family** | no | `V_tok = V_head = V` was already a construction constraint, stated in the accounting section. Over-Tokenized does not bear on the depth/width/FFN rule. |
| **Grids, budget, decision procedure** | no | Untouched. |

**One push-back, and it is not a design flaw.** The result — whichever way it goes — is a
statement about the **coupled** regime only, and the current novelty claim does not say so.
A reviewer who knows Over-Tokenized will ask "would decoupling rescue the law?", and the
honest answer is that this study cannot say. That is a **scope qualifier on the claim**,
not a change to the test:

> *the first controlled English BPE test of Tao et al.'s published vocabulary-parameter
> prediction below 33M `N_nv`, **under coupled input/output vocabulary size**, at their own
> compute-optimal budgets and under their own metric.*

Five added words. "Under coupled input/output vocabulary size" is preferred over the
earlier "with input and output vocabulary held equal": it names the regime rather than
describing an implementation detail, and it is the term a reader of Over-Tokenized will
recognise. It narrows the claim, which makes it more defensible, and it pre-empts the
obvious objection rather than waiting for a reviewer to raise it.

**Process note.** The paragraph itself is additive Stage-F guidance and would not normally
warrant re-review. The claim qualifier does touch the headline sentence, which is the one
thing a reviewer reads first. Given the plan is `approved-final`, the qualifier should go
through one Codex round rather than being edited in quietly.

---

## (c) Discussion sketch, if the law fails below 33M

Two mechanisms would then be available, and they are **complementary rather than
competing** — one explains why the output side underperforms the law, the other explains
what the input side would have preferred.

**Godey et al. (arXiv:2404.07647) — output-side rank bottleneck.** The vocabulary
projection has rank bounded by `d`. At 14–31M they show small hidden dimensions induce
saturation in exactly this projection. Under coupling, `d` is small precisely where the law
predicts a large `V`, so the output head cannot express the distribution the extra
vocabulary buys. Prediction: the **empirical optimum sits below the predicted `N_v`** — the
law over-allocates to a head that cannot use it.

**Over-Tokenized (arXiv:2501.16975) — input-side benefit.** Larger input vocabulary helps
log-linearly and independently of model size. Under coupling, the input side is *held back*
by whatever the output side can tolerate.

**The joint reading.** If our test rejects with `θ < 0` (optimum below prediction), the two
mechanisms compose into a single account: at small scale the output head is rank-limited
while the input side still wants more vocabulary, so **coupling — not vocabulary size — is
the binding constraint**, and the correct small-model design is a large input vocabulary
with a small output head. That reframes a negative result about Tao's law into a positive
design recommendation, and it is directly testable as follow-up.

**The instrument already exists.** Stage E's mechanism ablation varies `d` at matched
`N_nv` — precisely the Godey lever. If the failure magnitude tracks `d` rather than `V`,
that is direct evidence for the rank account over a generic "small models are different"
story.

**Honest caveat that must be stated, not buried.** This composition only works for
`θ < 0`. If the empirical optimum sits **above** the prediction (`θ > 0`), Godey's mechanism
predicts the wrong sign and the two-mechanism story collapses; that outcome would need a
different explanation and should not be retrofitted. The discussion must be written to
survive either direction, and the direction is not known until the runs are done. Committing
to the narrative before seeing `θ` would be exactly the post-hoc storytelling the
preregistration exists to prevent.

---

## (d) Framing — a partial push-back

**Agreed on the substance.** `exp_data.csv` shows Tao et al. ran **one run per cell, no seed
replication** across all 1,200 rows. Their bowl minima therefore carry unquantified noise,
and every downstream extrapolation inherits it. This study runs `n ≥ 5` seeds with BCa
intervals and a lack-of-fit gate. That asymmetry is real and it is the most defensible thing
about the work.

It is also sharper than "we replicated": **M1's margin (`ln 1.5`) is calibrated against
their published grid spacing, so the comparison is between a replicated estimate and an
unreplicated one at a granularity finer than their own noise.** That is the sentence worth
writing.

**Where I would push back: leading with it.** A paper whose opening move is "they did not
replicate, we did" reads as a replication study, and replication is a weaker novelty claim
than the one actually available — validating a widely-cited law outside its fitted range.
Leading with rigour also invites the reviewer to score the work as methodological rather
than substantive.

**Proposed ordering instead:**

1. **The question** — the law is extrapolated 1–2 orders of magnitude below its data, and
   recommends ~1,715–4,459-token vocabularies at 2–16M where practitioners inherit 32k.
   Nobody has checked.
2. **Why the check is credible** — their minima are single-run; ours are replicated at
   `n ≥ 5` with equivalence testing against an a-priori margin. Not a footnote, the second
   paragraph.
3. **What it cannot say** — coupled regime only (the (b) qualifier).

Replication is the **credibility** claim, not the **novelty** claim. Making it carry both
weakens the first without strengthening the second. Stated second and stated plainly, it
does the work you want it to do.

---

## Recommended action

1. Add the (a) paragraph to Stage F related work — additive, low risk.
2. Add the scope qualifier "under coupled input/output vocabulary size" from (b) to the
   novelty claim. **Codex round completed 2026-08-01**: approved as accurate and necessary,
   with the wording change from "with input and output vocabulary held equal" adopted, and
   the over-claim about `N_v` being "the output head only" corrected.
3. Hold (c) as write-up guidance, explicitly conditional on the sign of `θ`.
4. Adopt the (d) ordering: question first, replication second.

None of this blocks Stage B.7. The pilot's design is unaffected.
