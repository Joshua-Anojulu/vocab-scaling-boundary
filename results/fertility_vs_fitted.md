# Realized fertility vs Tao's fitted f(V)

Twenty byte-level BPE tokenizers, `DKYoon/SlimPajama-6B` per amendment A1. Every one
satisfies the preregistered invariants: realized size exactly `V`, lossless round-trip,
`unk = 0`.

`f(V) = 0.00639222·ln(V)² − 0.15811069·ln(V) + 1.20470122` is Tao's fitted
tokens-per-character term, used inside their FLOP convention. It was fitted over
**V ∈ [4096, 96256]**.

## Measured on population counts, on identical text

This table supersedes an earlier one built from 200-document samples per vocabulary.
Two choices in it are deliberate:

**Which split.** The `train` arrays cover a different number of documents per vocabulary
(22,000 to 570,000), because each vocabulary needed its own token target. Measuring
fertility across V on those arrays would confound V with whatever text each tokenizer
happened to see. The `selection_val` arrays are **the same 17,749 documents / 78,210,433
characters for every V**, so the cross-V curve below is measured on identical text.

**EOS.** The packer inserts one `<eos>` per document (amendment P4). `f(V)` describes how
text segments, not how it is packed, so the headline column excludes EOS. The `+EOS`
column shows the choice is worth 0.037–0.087% and does not affect any conclusion here.

| V | tok/char | f(V) | ratio | +EOS | train | region |
|---|---|---|---|---|---|---|
| 384 | 0.60774 | 0.49019 | **1.2398** | 0.60796 | 0.56144 | below |
| 512 | 0.53250 | 0.46712 | 1.1400 | 0.53273 | 0.52887 | below |
| 768 | 0.46781 | 0.43640 | 1.0720 | 0.46804 | 0.46421 | below |
| 896 | 0.44963 | 0.42527 | 1.0573 | 0.44986 | 0.42770 | below |
| 1024 | 0.43510 | 0.41588 | 1.0462 | 0.43532 | 0.43287 | below |
| 1152 | 0.42307 | 0.40778 | 1.0375 | 0.42330 | 0.42133 | below |
| 1536 | 0.39642 | 0.38875 | 1.0197 | 0.39664 | 0.39114 | below |
| 1664 | 0.38972 | 0.38364 | 1.0158 | 0.38995 | 0.38776 | below |
| 2176 | 0.36879 | 0.36713 | 1.0045 | 0.36902 | 0.36608 | below |
| 3200 | 0.34168 | 0.34499 | 0.9904 | 0.34191 | 0.33865 | below |
| 3456 | 0.33674 | 0.34080 | 0.9881 | 0.33697 | 0.33121 | below |
| 4224 | 0.32415 | 0.33023 | 0.9816 | 0.32438 | 0.32268 | **inside** |
| 4480 | 0.32062 | 0.32723 | 0.9798 | 0.32085 | 0.31778 | inside |
| 6144 | 0.30334 | 0.31188 | 0.9726 | 0.30357 | 0.30204 | inside |
| 6272 | 0.30225 | 0.31092 | 0.9721 | 0.30248 | 0.29887 | inside |
| 6912 | 0.29738 | 0.30648 | 0.9703 | 0.29761 | 0.28811 | inside |
| 8448 | 0.28803 | 0.29769 | 0.9675 | 0.28826 | 0.27737 | inside |
| 8960 | 0.28546 | 0.29521 | 0.9670 | 0.28569 | 0.28250 | inside |
| 12672 | 0.27153 | 0.28150 | 0.9646 | 0.27175 | 0.27010 | inside |
| 17792 | 0.26013 | 0.26957 | 0.9650 | 0.26036 | 0.25823 | inside |

Ratio: overall **[0.9646, 1.2398]**; inside the fitted range **[0.9646, 0.9816]**; below it
**[0.9881, 1.2398]**.

Reproduce with `scripts/fertility_population.py`.

## A hypothesis that the control refuted

The points below V=4096 suggest downward-extrapolation drift: the ratio rises steeply as V
falls below Tao's fitted range, reaching **1.240 at V=384**, 10.7× below their minimum. The
prediction that follows is that the ratio should flatten toward 1.0 **inside**
[4096, 96256], where the polynomial interpolates rather than extrapolates.

**It does not.** Inside the fitted range the ratio is 0.965–0.982 — a persistent ~3%
offset — and it declines monotonically with **no discontinuity at the V=4096 boundary**.
The curve is smooth through the point where the explanation predicted a break.

So this is not "accurate inside its range, drifting outside." Measured fertility and `f(V)`
are two smooth curves of different shape that cross near V≈2400. The apparent extrapolation
story was an artifact of only looking below the boundary.

**The correct reading** is mundane: `f(V)` is a property of *a corpus and a tokenizer
pipeline*. Both differ here — `SlimPajama-6B` rather than chunk1 (amendment A1, forced by
the preregistered corpus being deleted), and this study's byte-level BPE configuration
rather than whatever theirs was. A shape difference is the expected outcome, not a finding.

## What the population numbers changed

The 200-document estimates were wrong by a **median 2.14%, max 2.47% — and every one of
them low**. A uniformly signed error is not sampling noise. The cause is that those samples
were drawn from the tokenizer-fit split, which is the text the BPE merges were learned on:
a tokenizer segments its own training text more efficiently than held-out text. Measuring
fertility on the fit split understates it by about 2%.

That is a small number, but it inverted a sign. Under the old estimates the curve crossed
`f(V)` near V≈1550 and the inside-range offset read as ~5%; on population counts the
crossing is near V≈2400 and the offset is ~3%.

## Consequences

**Strengthens a plan decision.** Enforcing IsoFLOP on exact token counts — adopted in
review round 4 after the reviewer noticed `f(V)` is *fitted* rather than realized fertility
— is not merely prudent, it is required. Consuming `H = T_target/f(V)` characters would
have mis-budgeted **every run in the study**, by 3% at the top of the grid and **24% at the
bottom**, with no error surfaced anywhere. Enforcing on tokens makes `C` exact by
construction and converts the mismatch into a measured quantity.

That the largest mis-budgeting would have landed at the smallest vocabularies matters
specifically here, because those are the cells where the law is being tested furthest from
its fitted range. A character-based budget would have confounded the effect under test with
a 24% compute error in exactly the region the study is about.

**Narrows a claim.** This must NOT be reported as evidence that the fertility polynomial
degrades under extrapolation. The inside-range control refutes that. It is a
corpus-plus-pipeline difference and says nothing about the vocabulary law under test.

**Corpus sizing re-verified against measurement.** Recomputed with population fertility,
the worst single run is the anchor at V=6144 needing **2.29 GB** of train text (2.51 GB for
its M2 arm at 1.1·C), against **6.00 GB** ingested — 2.39× headroom. Sub-33M runs need
0.12–1.05 GB each.

## Operational note

Two tokenizers took wildly longer than their neighbours — V=4480 at **3890 s** against
68–78 s, and a smaller excursion at V=12672 — while the runs immediately after took 31 s
and 35 s, faster than anything before. This is machine contention, not a property of BPE
training, which is deterministic and passed all invariants in both cases. It cost
wall-clock only. Same lesson as the contaminated 2M throughput benchmark: this host is not
a quiet lab, and timings taken on it need corroboration before they are trusted.
