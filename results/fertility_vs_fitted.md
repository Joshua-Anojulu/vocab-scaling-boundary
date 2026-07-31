# Realized fertility vs Tao's fitted f(V)

Twenty byte-level BPE tokenizers trained on the 150 MB tokenizer-fit split
(36,576 documents, `DKYoon/SlimPajama-6B` per amendment A1). Every one satisfies the
preregistered invariants: realized size exactly `V`, lossless round-trip, `unk = 0`.

`f(V) = 0.00639222·ln(V)² − 0.15811069·ln(V) + 1.20470122` is Tao's fitted
tokens-per-character term, used inside their FLOP convention. It was fitted over
**V ∈ [4096, 96256]**.

| V | measured tok/char | f(V) | ratio | region |
|---|---|---|---|---|
| 384 | 0.5970 | 0.4902 | **1.218** | below fitted range |
| 512 | 0.5212 | 0.4671 | 1.116 | below |
| 768 | 0.4578 | 0.4364 | 1.049 | below |
| 896 | 0.4398 | 0.4253 | 1.034 | below |
| 1024 | 0.4254 | 0.4159 | 1.023 | below |
| 1152 | 0.4133 | 0.4078 | 1.014 | below |
| 1536 | 0.3879 | 0.3888 | 0.998 | below |
| 1664 | 0.3818 | 0.3836 | 0.995 | below |
| 2176 | 0.3609 | 0.3671 | 0.983 | below |
| 3200 | 0.3350 | 0.3450 | 0.971 | below |
| 3456 | 0.3295 | 0.3408 | 0.967 | below |
| 4224 | 0.3176 | 0.3302 | 0.962 | **inside** |
| 4480 | 0.3143 | 0.3272 | 0.960 | inside |
| 6144 | 0.2972 | 0.3119 | 0.953 | inside |
| 6272 | 0.2960 | 0.3109 | 0.952 | inside |
| 6912 | 0.2908 | 0.3065 | 0.949 | inside |
| 8448 | 0.2810 | — | — | inside |
| 8960 | 0.2784 | — | — | inside |
| 12672 | 0.2646 | — | — | inside |
| 17792 | 0.2537 | — | — | inside |

## A hypothesis that the control refuted

The first eleven points suggested downward-extrapolation drift: the ratio rises steeply as
V falls below Tao's fitted range, reaching 1.218 at V=384 (10.7× below their minimum).
The prediction that follows is that the ratio should flatten toward 1.0 **inside**
[4096, 96256], where the polynomial is interpolating rather than extrapolating.

**It does not.** Inside the fitted range the ratio is 0.949–0.962 — a persistent ~5%
offset — and it continues declining monotonically with **no discontinuity at the V=4096
boundary**. The curve is smooth through the point where the explanation predicted a break.

So this is not "accurate inside its range, drifting outside." Measured fertility and
`f(V)` are two smooth curves of different shape that happen to cross near V≈1550. The
apparent extrapolation story was an artifact of only looking below the boundary.

**The correct reading** is mundane: `f(V)` is a property of *a corpus and a tokenizer
pipeline*. Both differ here — `SlimPajama-6B` rather than chunk1 (amendment A1, forced by
the preregistered corpus being deleted), and this study's byte-level BPE configuration
rather than whatever theirs was. A shape difference is the expected outcome, not a finding.

## Consequences

**Strengthens a plan decision.** Enforcing IsoFLOP on exact token counts — adopted in
review round 4 after the reviewer noticed `f(V)` is *fitted* rather than realized
fertility — is not merely prudent here, it is required. Consuming `H = T_target/f(V)`
characters would have mis-budgeted **every run in the study**, by 5% at the top of the
grid and 22% at the bottom, with no error surfaced anywhere. Enforcing on tokens makes `C`
exact by construction and converts the mismatch into a measured quantity.

**Narrows a claim.** This must NOT be reported as evidence that the fertility polynomial
degrades under extrapolation. The inside-range control refutes that. It is a
corpus-plus-pipeline difference and says nothing about the vocabulary law under test.

**Corpus sizing re-verified against measurement.** The ingestion caps were computed with
`f(V)`. Recomputed with measured fertility, the worst single run is the anchor at V=6144
needing **2.29 GB** of train text (2.51 GB for its M2 arm at 1.1·C), against **6.00 GB**
ingested — 2.39× headroom. Sub-33M runs need 0.12–1.05 GB each.

## Operational note

Two tokenizers took wildly longer than their neighbours — V=4480 at **3890 s** against
68–78 s, and a smaller excursion at V=12672 — while the runs immediately after took 31 s
and 35 s, faster than anything before. This is machine contention, not a property of BPE
training, which is deterministic and passed all invariants in both cases. It cost
wall-clock only. Same lesson as the contaminated 2M throughput benchmark: this host is
not a quiet lab, and timings taken on it need corroboration before they are trusted.
