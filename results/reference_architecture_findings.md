# Reference architecture findings (Stage B.2)

Source: `reference/lit_gpt_config.py`, pulled from
`experiments/light_train/lit_gpt/config.py` in `sail-sg/scaling-with-vocab`.

## The anchor architecture, resolved

`tiny_LLaMA_50M`: `n_layer=8, n_head=8, n_embd=512, intermediate_size=2048,
block_size=2048, bias=False, FusedRMSNorm, LLaMAMLP`.

This closes the Stage B.2 item. Two immediate consequences:

* **`padding_multiple=1` — they do not pad the vocabulary at all.** The plan's decision to
  train every tokenizer at a size divisible by the padding quantum so that
  `V_tok = V_head = V` reaches the same place by construction, and the "padded vs nominal
  V" ambiguity raised in review is moot against this reference.
* **`intermediate_size = 4·d` at the anchor**, not `8d/3`. That is the whole of the 3.10%
  miss when the study's width rule was checked against the anchor.

## Their `N_nv` is a NOMINAL LABEL, not a parameter count

Computing the true body count as `n_layer·(4d² + 3·d·d_ffn + 2d) + d` and comparing with
the published `Non_vocab_parameters`:

| family | L | d | d_ffn | ffn/d | true params | published label | true/label |
|---|---|---|---|---|---|---|---|
| 50M | 8 | 512 | 2048 | 4.00 | 33,563,136 | 33,222,784 | 1.0102 |
| 110M | 12 | 768 | 2048 | 2.67 | 84,953,856 | 84,834,176 | 1.0014 |
| 176M | 16 | 768 | 3072 | 4.00 | 151,020,288 | 150,834,176 | 1.0012 |
| 335M | 18 | 1024 | 4096 | 4.00 | 302,027,776 | 316,445,568 | **0.9544** |
| 682M | 20 | 1536 | 4800 | 3.12 | 631,174,656 | 631,668,352 | 0.9992 |
| 1197M | 22 | 2048 | 5632 | 2.75 | 1,130,457,088 | 1,129,891,136 | 1.0005 |

The label is `nominal_family_size − 2·16384·d`, which reproduces the published column
exactly (verified for all six in `tests/test_reference_parity.py`) — but the nominal size
is a round number, not a count. Four families agree with their true count to within 0.15%.
**Two do not: the anchor by +1.0%, and 335M by −4.6%.**

The 335M case is traceable to an inconsistency inside the reference itself:
`model_size_dict` assigns `350e6` to a key named `'335M'`, for a family whose parameters
actually total 302.0M.

### Why this matters for the test

The published law relates `N_v` to this *labeled* `N_nv`. A 4.6% error on the x-axis
propagates through A2 as `1.046^0.8354 = 1.038`, i.e. ~3.8% in predicted `N_v`, or
`ln(1.038) = 0.037` — about **9% of the M1 equivalence margin** (`ln 1.5 = 0.4055`).
Not fatal, but not negligible either, and it is a property of the reference data rather
than of anything measured here.

**Decision for this study:** the sub-50M architectures use their **true** parameter count
as `N_nv`, which `src/model.py` reproduces exactly and which
`tests/test_model.py::test_non_vocab_params_match_the_preregistered_rule` enforces. There
is no "nominal family size" for architectures outside their ladder, so their labelling
convention cannot be applied to them even in principle. The ~1–5% label/count divergence
is recorded as a systematic uncertainty in the comparison and should be reported, with a
sensitivity analysis re-running the M1 test against label-convention `N_nv` at the anchor.

## There is no architecture rule to recover from their ladder

`d_ffn/d` takes the values 4.00, 2.67, 4.00, 4.00, 3.12, 2.75 with no pattern; depth goes
8, 12, 16, 18, 20, 22; and `d = 768` appears twice at different depths. `Nnv_to_d` is a
coarse bucketed lookup, not a generative rule.

This is direct evidence for something the plan already asserted under review pressure: the
study's sub-50M depth/width/FFN rule is a **genuine new extrapolation**, not a
reconstruction of theirs, and the primary claim is correctly scoped to that architecture
family. It also means "why doesn't your rule reproduce their anchor?" has a precise
answer — their anchor is not on any rule, including their own.

---

# The training schedule is warmup-then-CONSTANT, not cosine

From `reference/tinyllama_pretrain.py`:

```python
def get_lr(it, lr_decay_iters, warmup_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    return learning_rate
```

Called at line 297 and applied to the optimizer's param groups. `min_lr = 4e-5` is
defined at line 47 and **referenced nowhere**; `lr_decay_iters` is accepted and ignored;
`decay_lr = True` merely selects this function. **There is no decay phase.**

Full recipe, transcribed rather than chosen: `lr=4e-4`, `weight_decay=1e-1`,
`betas=(0.9, 0.95)`, `grad_clip=1.0`, `block_size=2048`, `global_batch_size=512`
sequences, `warmup_steps=2000` of `max_step=25000` (8%, encoded here as a fraction so it
scales with run length).

## Consequence: the case against checkpoint reuse was premised on cosine

Round 2 of review blocked WSD and intermediate-checkpoint reuse on the grounds that
"under cosine decay, a checkpoint taken partway through a long run has a different
learning-rate history from a model intentionally trained to that lower compute budget."

That reasoning is correct — and does not apply to a constant-rate schedule. Under Tao's
actual schedule, a checkpoint at step *k* has **exactly** the learning-rate history of a
run trained to step *k*: linear warmup, then a constant rate. Checkpoint reuse is
therefore legitimate here.

**Concrete saving, if adopted:** the M2 arm currently trains `V_run` separately at
`1.1·C`, costing `C + 1.1·C = 2.1·C` per scale-seed. Training once to `1.1·C` and
checkpointing at `1.0·C` yields both points for `1.1·C` — saving `1.0·C·n` per scale, or
about **13 GPU-hours** of the measured 89.1 (→ ~76 h).

**Not adopted.** The preregistration specifies separate budget-specific runs. Changing it
after seeing the budget is precisely the post-hoc researcher degree of freedom the
preregistration exists to prevent, even when the change is sound and the justification is
a property of the reference rather than of the data. If it is taken, it must be a
documented amendment to Preregistration I made **before** any confirmatory run, recording
that the trigger was reading the reference schedule, not the cost.
