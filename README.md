# Does the compute-optimal vocabulary law hold below 33M parameters?

A preregistered test of whether Tao et al.'s compute-optimal vocabulary predictor
(*Scaling Laws with Vocabulary*, NeurIPS 2024) still predicts the best vocabulary size for
language models smaller than the range it was fitted on.

**Status: paused at the pilot stage since August 2026. There are no results yet, and
nothing in this repository is a finding about the hypothesis.** What exists is the
approved preregistration, its amendments, the training and evaluation code with tests,
and measurements from the throughput and sensitivity pilots.

## The question

Tao et al. fit their law on models down to about 33.2M non-vocabulary parameters. Pushed
below that range, it recommends vocabularies of roughly 1,700 to 4,500 tokens at 2M to 16M
parameters, far smaller than the 32k-token subword vocabularies small models usually
inherit. Nothing has tested that extrapolation. This study trains models at 2M, 4M, 8M
and 16M non-vocabulary parameters, sweeps five vocabulary sizes around each predicted
optimum under matched compute, and asks whether the predictor lands within preregistered
margins at every scale. Tao et al. ran one run per cell; this design replicates each cell
across seeds.

The exact hypothesis, parameter accounting, architecture family, compute ruler and
statistics are in [`PLAN.md`](PLAN.md).

## How the plan was reviewed

`PLAN.md` was approved after seven rounds of adversarial review by a second model, logged
in full in [`PLAN-REVIEW-LOG.md`](PLAN-REVIEW-LOG.md). The approved plan is never edited.
Every later change is a dated amendment in [`AMENDMENTS.md`](AMENDMENTS.md) (A1 to A7),
recorded with its evidence.

## Repository layout

```text
PLAN.md                   the preregistration (approved, frozen)
AMENDMENTS.md             dated amendments A1 to A7, and proposals not adopted
PLAN-REVIEW-LOG.md        every review round, verdict and response
PROPOSED-RELATED-WORK.md  related-work notes
src/                      tokenizer, corpus tokenization, model, training, evaluation,
                          IsoFLOP fitting, pilot runner, reanalysis
reference/                upstream scripts from sail-sg/scaling-with-vocab, used for
                          parity tests (they keep their original license)
scripts/                  sensitivity checks and pilot probes
tests/                    pytest suite, including parity against the reference code and
                          an audit that checks written claims against recorded results
results/                  pilot measurements (throughput, VRAM, warmup stability,
                          tokenizer fertility, smoothing and packing sensitivity)
```

## Hardware and data

The pilots ran on a single laptop GPU (RTX 4060, 8 GB) with PyTorch and CUDA. The
tokenized corpus, tokenizers, checkpoints and run directories are tens of gigabytes and
are not committed; `.gitignore` lists them.

## Running the tests

```bash
python -m pytest tests/
```

## Reference

Tao, C., Liu, Q., Dou, L., Muennighoff, N., Wan, Z., Luo, P., Lin, M., and Wong, N. (2024).
*Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024.
Code: [sail-sg/scaling-with-vocab](https://github.com/sail-sg/scaling-with-vocab).
