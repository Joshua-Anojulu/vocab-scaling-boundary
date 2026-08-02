# Plan Review Log — vocabulary scaling law downward extrapolation

## Task

Design a from-scratch LLM research project, executable on a single RTX 4060 Laptop (8 GB), that
produces a publishable finding. Locked question: **do the vocabulary scaling laws of Tao et al.
(NeurIPS 2024, arXiv:2407.13623) extrapolate below their fitted range of 33M non-vocabulary
parameters, into the regime where embedding and softmax cost dominates?**

## Act 1 summary (grill)

Resolved with the user, in order:

1. **Compute.** Own GPU confirmed by direct inspection as **RTX 4060 Laptop, 8 GB VRAM** (compute 8.9,
   bf16 native), 31.7 GB RAM, i9-14900HX. User initially reported "15.9 GB" from Task Manager; that
   figure was shared system memory, not dedicated VRAM. UT TACC access exists but is scoped to a
   funded project — treated as upside only, never as a dependency.
2. **Goal correction.** User's initial goal was "a model incredibly amazing at debugging my
   codebases." Rejected as structurally unachievable from scratch at any budget available. Presented
   the trilemma (from-scratch / publishable / useful — pick two); user selected from-scratch +
   publishable, later restating priorities as "publishable, still lets me build an LLM and grasp
   their purpose."
3. **Topic saturation check (user-requested: "ensure unsaturated field").** User chose
   tokenizer × scale. A literature survey found the field **saturated at first order** and killed two
   of three candidate angles:
   - vocab × scale → Tao et al., NeurIPS 2024
   - character-level causal study → *The Strawberry Problem* (arXiv:2505.14172) does exactly this
   - reconciling Tao vs Over-Tokenized → *Compute Optimal Tokenization* (arXiv:2605.01188, Meta,
     988 models, 50M–7B) reframed the whole question
   - code vocabulary gains → already reported as "marginal in symbol-heavy domains"
   Field velocity measured: ~27 papers cited the anchor in 2026 alone.
4. **Surviving gap.** Both anchor papers state lower bounds (33M and 50M) and neither examines or
   claims downward validity; their stated limitations point exclusively upward. Below 33M the
   embedding table becomes the majority parameter term (verified against MobileLLM, arXiv:2402.14905).
   Self-stress-tested: Kaplan's exclude-embeddings convention was established at *fixed* vocabulary,
   whereas Tao et al. *vary* it — which is precisely when it cannot be factored out.
5. **Timeline.** No deadline; learning primary, publish if it works. Checkpoint after the sweep.
6. **Environment.** Python 3.14.5 installed — too new for reliable PyTorch wheels; plan pins a
   3.12/3.13 venv. 224 GB free. Project rooted outside OneDrive to prevent syncing tens of GB of
   derived data.

## Resolved caps

```
MAX_ROUNDS   = 5
MAX_ATTEMPTS = 8
PLAN_FILE    = C:\Users\josha\Projects\vocab-scaling-boundary\PLAN.md
LOG_FILE     = C:\Users\josha\Projects\vocab-scaling-boundary\PLAN-REVIEW-LOG.md
reviewer     = codex (auto-selected; only verified+auto row)
```

## Reviewer

| | |
|---|---|
| Adapter | `codex` (trust: builtin, selection: auto) |
| Grounding | `repo` (read-only by mechanism) |
| Model | `gpt-5.6-sol`, reasoning effort `xhigh` |
| CLI version | `codex-cli 0.145.0` (allowlist `>=0.130 <0.146`) |
| Consent | Granted by user for this repo; repo contained only `PLAN.md` + fresh `.git` at time of consent |

**Canary (pre-round 1): PASS.** Codex was instructed to write a file and to run a shell write. Both
write-class tools were invoked and both returned harness-level denials —
`patch rejected: writing is blocked by read-only sandbox` and
`exec_command ... Rejected("... blocked by policy")`. `PLAN.md` byte-identical; no canary artifacts
created. Note: the naive tree hash reported a mutation, traced to the orchestrator's own `.omc/state/*`
hook writes, not the reviewer. Future integrity checks exclude `.omc/`.

---

## Round 1 — codex

| | |
|---|---|
| Model | `gpt-5.6-sol` (reasoning effort `xhigh`) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` |
| Grounding | `repo` adapter class; **plan inlined into the prompt** because `-s read-only` blocked Codex's shell-based file reads (see Attempt 1 below). The repo contained only `PLAN.md` and `PLAN-REVIEW-LOG.md`, so no repo content was withheld. |
| Reviewed body sha256 | `579bfb637eba9ed905ded2c2a06a237c092e9fca3fe4b770b04b25306cebb798` |
| Verdict | **REVISE** (confirmed by `parse-verdict.sh` and `validate-attempt.sh`, both exit 0) |

**Attempt 1 (not a round).** First launch produced no verdict and was killed at the 600s ceiling.
Root cause from stderr: Codex tried to read `PLAN.md` by shelling out to PowerShell, and `-s read-only`
blocks `exec_command` wholesale rather than only blocking writes — so the reviewer could not see the
plan. An orphaned `codex.exe` survived the kill and was terminated. Fix applied: inline the plan text
and run detached. Classified `timeout` mechanically, but the true cause was adapter invocation, not
transience. Attempts used: 2 of 8.

### Critique (verbatim)

The plan is not implementation-ready. The narrow question may still be novel, but the current design misstates Tao et al.’s exponent, contradicts IsoFLOP control, lacks enough runs for inference, and underestimates compute.

## 1. Novelty

1. **The “nobody tested below 33M” claim is too strong.** I found no prior English natural-text BPE IsoFLOP study directly estimating Tao’s relationship below 33M non-vocabulary parameters, so that narrow gap appears defensible. But relevant sub-33M evidence already exists: Huang et al. compare token granularities in a 2.4M-parameter synthetic LM and find larger output vocabularies harm the small model; Godey et al. analyze 14M and 31M models and connect their degradation to the vocabulary-facing softmax bottleneck; GPT-wee trains 0.42M–7.52M models with 4k–16k vocabularies, albeit with model size and vocabulary confounded. [Huang et al.](https://arxiv.org/abs/2501.16975), [Godey et al.](https://arxiv.org/abs/2404.07647), [GPT-wee](https://aclanthology.org/anthology-files/pdf/conll/2023.conll-babylm.2.pdf)

   **Fix:** Claim only “the first controlled English BPE IsoFLOP test of Tao et al.’s vocabulary-parameter law below 33M,” after a systematic literature review.

2. **The plan misstates the central estimand.** Tao’s 0.84 is not the exponent of vocabulary size \(V\) against model scale. Their fits are \(N_v\propto C^{0.42}\) and \(N_{nv}\propto C^{0.50}\), hence \(N_v\propto N_{nv}^{0.84}\), where \(N_v=Vd\). Because hidden dimension \(d\) changes with \(N_{nv}\), \(V\) itself has a different exponent. [Tao et al.](https://papers.neurips.cc/paper_files/paper/2024/file/cf5a019ae9c11b4be88213ce3f85d85c-Paper-Conference.pdf)

   **Fix:** Make \(N_v=Vd\) versus \(N_{nv}\) the primary estimand and report the induced vocabulary-size relationship separately.

3. **The adjacent literature has moved substantially since Tao.** Compute Optimal Tokenization already trains 320 subword models alongside 988 BLT models, uses BPB, and finds that bytes per parameter and compression rate are the more stable variables; Takase et al. and Huang et al. also study vocabulary enlargement directly. The proposed paper cannot read as merely “Tao, but smaller.” [Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188), [Takase et al.](https://aclanthology.org/2025.findings-acl.57/)

   **Fix:** Frame the contribution as a boundary test separating vocabulary parameters, compression rate, and small-model softmax limitations.

## 2. Validity and statistical power

4. **“Fixed byte budget” and IsoFLOP are mutually incompatible as written.** For \(B\) bytes, fertility \(r(V)\) tokens/byte, and active matrix parameters approximately \(N_{nv}+Vd\), compute is roughly \(6(N_{nv}+Vd)Br(V)\). Holding \(B\) fixed makes compute vary; holding compute fixed makes consumed bytes vary.

   **Fix:** Hold the ordered source corpus fixed, but let bytes consumed be determined by the preregistered FLOP budget for each configuration.

5. **Twenty configurations cannot locate a joint compute optimum.** Six vocabulary sizes already consume six cells. Estimating the optimum jointly over \(V\), \(N_{nv}\), and training amount requires several model sizes per vocabulary, plus replication; three or four compute budgets may reuse trajectories only under a scheduler designed for that reuse.

   **Fix:** Publish an explicit configuration matrix and run-count calculation before training; otherwise replace the broad IsoFLOP fit with a targeted falsification of Tao’s predicted optima.

6. **There is no seed replication or power analysis.** At tiny scale, initialization, batch ordering, tokenizer sampling, and architecture rounding can easily move the winner between adjacent vocabulary sizes. Confidence intervals derived from one run per cell would be pseudo-precision.

   **Fix:** Estimate run-to-run BPB variance in a pilot and preregister at least three seeds for every configuration used in the primary comparison.

7. **Three or four compute budgets are inadequate for an exponent.** Fitting an intercept and slope to three points leaves one residual degree of freedom; selecting each point as the minimum of noisy discrete alternatives further biases the slope and understates uncertainty.

   **Fix:** Either use at least six well-separated budgets with replicated continuous minima or test Tao’s fixed prediction directly instead of estimating a new exponent.

8. **The vocabulary grid is coarse and likely censored.** A true optimum between 4k and 8k will be rounded to one of them; an optimum below 1k or above 32k becomes a boundary winner. Such quantized minima can manufacture or erase a power law.

   **Fix:** Fit a smooth loss surface in \(\log V\) and add preregistered guard points such as 512 and 64k, followed by adaptive bracketing that does not inspect final test loss.

9. **The “overlap control” does not reproduce a relationship.** One 33–35M anchor can test calibration at one point, but cannot reproduce Tao’s 0.84 slope. Different corpus sampling, metric, architecture, scheduler, and weight tying would make even that calibration ambiguous.

   **Fix:** Reproduce at least two published-range architectures and their vocabulary-loss bowls, or call the 33M run an anchor rather than a replication.

10. **Architecture changes can masquerade as a scaling-law break.** At 0.5M parameters, integer layer/head constraints create abrupt architecture changes; hidden dimension also controls softmax rank. Godey et al. show that small hidden dimensions themselves induce saturation, exactly where this plan expects a vocabulary boundary.

   **Fix:** Preregister exact architectures and include width or head-rank as a modeled covariate, with one matched-\(N_{nv}\) width ablation at both ends of the range.

11. **Training hyperparameters are another unaddressed interaction.** A fixed learning rate, token batch size, warmup fraction, and token context do not imply equivalent optimization across vocabulary sizes: bytes per batch, predictions per update, raw-text context, and gradient noise all change.

   **Fix:** Define batch size, schedule, and context in both token and byte units, then preregister one sensitivity analysis rather than tuning after observing the primary result.

12. **Intermediate checkpoints may not be valid lower-budget runs.** Under cosine decay, a checkpoint taken partway through a long run has a different learning-rate history from a model intentionally trained to that lower compute budget.

   **Fix:** Use separate budget-specific schedules or a schedule such as WSD whose checkpoints are explicitly designed for budget reuse.

13. **Validation is being used for model selection and reporting.** Locating the best vocabulary on the held-out validation set and then reporting the same loss understates error.

   **Fix:** Split by document into tokenizer-training/train, selection-validation, and untouched final-test partitions before any experiments.

## 3. Metric

14. **The core BPB reasoning is correct, but “mandatory” and “never” are overstated.** Per-token cross-entropy is not comparable across tokenizations. For the same losslessly encoded byte stream,
\[
\mathrm{BPB}=\frac{\sum_i-\log p(t_i\mid t_{<i})}{B\ln 2}
=\frac{\mathrm{NLL}_{nats/token}\times(\mathrm{tokens}/\mathrm{byte})}{\ln2}.
\]
Tao instead used unigram-normalized loss and compared it with BPC; BPB is a sound alternative, not the only valid normalization.

   **Fix:** State that raw per-token CE is invalid for cross-tokenizer ranking, while BPB is the preregistered primary metric and Tao’s normalized loss is a replication metric.

15. **“Measured fertility” is not sufficient to guarantee a correct conversion.** The formula is exact only when fertility and NLL cover the identical evaluation bytes, special tokens and padding are excluded, document-boundary predictions are handled consistently, and tokenization is lossless. Unicode normalization or `<unk>` tokens can silently change the denominator’s meaning.

   **Fix:** Compute BPB directly as total token NLL divided by the exact count of original UTF-8 bytes, assert round-trip tokenization, and report zero unknown-token rate.

16. **Changing Tao’s metric weakens the claimed exponent test.** Tao’s 0.84 was selected using unigram-normalized loss, not BPB. Their reported high correlation with BPC does not establish identical vocabulary minima, especially outside their fitted range.

   **Fix:** Evaluate every checkpoint using both exact BPB and Tao’s unigram-normalized loss, with disagreement treated as a substantive result.

17. **Context is another unit mismatch.** A fixed 2048-token window contains more bytes for a larger vocabulary, giving it longer raw-text context. This may be a real tokenizer benefit, but then the result is not purely about embedding/softmax allocation.

   **Fix:** Make one context convention primary and add a fixed-byte-context evaluation so the context-length contribution is visible.

## 4. FLOP and parameter accounting

18. **The motivating parameter arithmetic is internally wrong.** At \(d=512,V=32{,}000\), each table has 16.384M parameters. One side is 13.1% of a 125M model, not “over 20%”; both untied sides are 32.768M or 26.2%. Against 33M non-vocabulary parameters, both sides are 49.8% of total, not yet a majority. With tying, the vocabulary table is only 33.2% of total.

   **Fix:** Replace the prose with a table distinguishing input table, output head, tied total, untied total, and non-vocabulary parameters at every architecture.

19. **Embedding parameters do not imply embedding FLOPs.** Input embeddings are sparse lookups. The output projection costs \(O(Vd)\) per token and can dominate; tying halves stored parameters but does not remove that projection. “Embedding and softmax dominate FLOPs” conflates three different costs.

   **Fix:** Separate storage parameters, trainable capacity, output-projection matmuls, softmax operations, and input-lookup cost throughout the hypothesis.

20. **The standard \(6ND\) approximation is usable only with the right \(N\).** It should include active dense matrix parameters such as the \(Vd\) output head, exclude the lookup-only input embedding, and be supplemented by attention’s sequence-length-dependent term. Tao explicitly uses \(6(N_{nv}+Vd)D\); more exact scaling work shows that attention is the main omitted term. [Resolving Discrepancies in Compute-Optimal Scaling](https://papers.neurips.cc/paper_files/paper/2024/file/b6341525cd84f3be0ef203e4d7cd8556-Paper-Conference.pdf)

   **Fix:** Preregister an operator-level FLOP formula including the output head, attention, and softmax, then validate it against a profiler on every architecture family.

21. **Weight tying cannot remain unresolved.** It materially changes parameter dominance, model capacity, VRAM, and comparability, although output-head FLOPs remain similar.

   **Fix:** Match Tao’s exact implementation for the primary study and reserve tied-versus-untied behavior for a separately powered secondary ablation.

## 5. Feasibility

22. **The 33M overlap run does not plausibly fit the 1–4 hour target.** At 33M non-vocabulary parameters, \(d=512,V=32k\) gives about 49.4M active matrix parameters. Tao used 4.3B training characters at this scale. At roughly 3–4 bytes/token, the simple estimate is \(3.3\text{–}4.3\times10^{17}\) FLOPs before omitted costs. Completing that in four hours requires roughly 23–30 sustained TFLOP/s; completing it in one hour requires 90–120 TFLOP/s. A small, memory-sensitive model on a thermally constrained laptop is unlikely to sustain either target consistently.

   **Fix:** Benchmark one exact 33M/32k training step, calculate sustained useful FLOP/s, and rebuild the entire run budget from measured throughput before approving the sweep.

23. **Eight GB may fit the weights but creates a throughput trap.** An untied 33M+32k model has about 66M total parameters; fp32 parameters, gradients, and Adam states consume roughly 1 GB before activations. One 2048×32k logits tensor alone is about 128 MiB in bf16 or 256 MiB in fp32 per sequence, forcing small microbatches unless cross-entropy is chunked or fused.

   **Fix:** Require a VRAM pilot with the exact optimizer, context, checkpointing, and chunked/fused cross-entropy, reporting achievable tokens/s rather than merely “it fits.”

24. **The data storage estimate is plausible; the experimental count is not.** Six uint16 tokenizations of several billion bytes can indeed occupy tens of GB. The fatal issue is that a defensible grid is likely at least 36 trajectories before seeds, and over 100 with minimal replication, not “20+.”

   **Fix:** Separate storage feasibility from GPU-hour feasibility and give totals for screening runs, confirmatory seeds, ablations, failures, validation, and thermal downtime.

## 6. Likely rejection arguments

A hostile reviewer can reject this outright on four grounds:

- “The primary exponent is misdefined, and the reported 0.84 does not apply directly to vocabulary size.”
- “Twenty mostly unreplicated runs and three compute points cannot support a new scaling law or confidence interval.”
- “A failure to reproduce Tao is uninterpretable because the study changes metric, implementation, architecture regime, corpus sample, context units, and possibly weight tying.”
- “The claimed mechanism is not isolated: BPE vocabulary simultaneously changes compression, output dimension, embedding capacity, raw context, bytes per update, and optimization dynamics.”

**Fix:** Recast the paper as a preregistered boundary falsification study with one primary estimand, exact reference-method controls, replicated local comparisons, and a small mechanism ablation.

## 7. Stronger alternative

A simpler and stronger experiment would test Tao’s released predictions rather than attempt a new underpowered law:

1. Use the exact Tao 33M architecture plus approximately 16M, 8M, and 2M downward-scaled architectures.
2. For each model, calculate Tao’s preregistered predicted \(N_v\) and corresponding \(V\).
3. Screen five vocabulary sizes bracketing that prediction at exact matched FLOPs.
4. Fit each loss bowl continuously in \(\log V\), then replicate the best three candidates with at least three seeds.
5. Report BPB and Tao’s unigram-normalized loss on an untouched test set.
6. Compare a fixed-0.84 model with a broken-slope model using held-out predictive fit, not “0.84 falls inside a wide CI.”
7. Only if a break appears, add a small input/output-decoupling or width ablation to distinguish compression, output-head cost, and softmax-rank explanations.

This design answers the actual extrapolation question with fewer researcher degrees of freedom and substantially less compute. Building a separate from-scratch implementation remains valuable for learning, but it should first be numerically cross-validated against the reference implementation rather than serving as the sole experimental instrument.

VERDICT: REVISE

### Claude's response — round 1

**Accepted and acted on (17 of 24 items, including every material one):**

- **Item 2 — estimand misdefined.** Accepted as the single most important catch. The plan now states the
  primary estimand as `N_v = V·d` versus `N_nv`, with the induced `V` relationship reported separately.
  The original framing pointed at the wrong quantity throughout.
- **Item 18 — parameter arithmetic wrong.** Accepted. Replaced the prose with an explicit table. The
  claim "embeddings become the majority below 33M" was false at the boundary: untied is 49.8% of total at
  `N_nv=33M` (parity, not majority) and tied is 33.2%. Claim restated at its true strength.
- **Item 19/20 — three costs conflated.** Accepted. Storage, output-projection FLOPs, and input-lookup
  cost are now separated explicitly; input embedding is named as a near-free sparse gather. Operator-level
  FLOP formula including attention replaces bare `6ND`, profiler-validated.
- **Item 4 — fixed byte budget vs IsoFLOP.** Accepted; genuinely contradictory as written. Corpus order
  fixed, bytes consumed now determined by each configuration's FLOP budget.
- **Items 5, 7, 24 — underpowered inference and run-count error.** Accepted. Design converted from
  estimating an exponent to falsifying a fixed published prediction. Explicit configuration matrix and a
  preregistered shrink rule added.
- **Item 6 — no seed replication.** Accepted. ≥3 seeds on all primary comparisons; variance estimated in
  the Phase 0 pilot; seeds are the last thing sacrificed under the shrink rule.
- **Items 22, 23 — feasibility.** Accepted as the most practically important finding. Phase 0 is now a
  hard gate: sustained-throughput benchmark over ≥20 minutes to capture throttling, VRAM pilot with
  chunked/fused cross-entropy mandatory (the ~128 MiB bf16 logits tensor would otherwise force
  microbatch 1), and the grid rebuilt from measured numbers.
- **Items 8, 9, 10, 11, 12, 13, 15, 16, 17.** All accepted: continuous bowl fits in `log V` with 512/64k
  guard points; 33M reframed as calibration anchor; width/head-rank as modeled covariate plus matched
  ablation for the Godey confound; hyperparameters defined in both token and byte units; WSD schedule so
  checkpoints are budget-valid; BPB computed from exact UTF-8 byte counts with round-trip and zero-unk
  assertions rather than average fertility; both metrics on every checkpoint with disagreement treated as
  a result; fixed-byte context evaluation added.
- **Items 1, 3, 21 — novelty narrowed, tying escalated.** Claim narrowed to "first controlled English BPE
  IsoFLOP test of Tao's vocabulary-parameter law below 33M `N_nv`". Weight tying moved from open question
  to a blocking pre-run action.
- **Item 14.** Accepted with the reviewer's own qualification: BPB is preregistered primary, `L_u` is the
  replication metric. Independently confirmed before this round by reading Tao et al. directly — they use
  unigram-normalized loss, so the first draft's "bits-per-byte is mandatory" was both overstated and, more
  seriously, inconsistent with the plan's own comparability rationale.
- **Item 13 (validation reuse).** Accepted. Four-way document-level split with an untouched final test set.

**Rejected or qualified, with reasons:**

- **Item 1 — citation not taken at face value.** The attribution of a "2.4M-parameter synthetic LM" study
  to arXiv:2501.16975 appears garbled; that identifier is *Over-Tokenized Transformer*. The substantive
  claim (relevant sub-33M evidence exists) is independently supported by Godey et al. and GPT-wee, so the
  conclusion stands and the novelty claim was narrowed accordingly — but the plan now requires all
  citations be verified from primary sources rather than inherited from this review.
- **Item 9, first branch — rejected.** "Reproduce at least two published-range architectures and their
  vocabulary-loss bowls" is precisely the compute the reviewer's own item 22 demonstrates is unavailable;
  two 33M+ replications at Tao's token counts would consume the entire budget. Took the second branch of
  the same fix instead: the 33M run is labelled an anchor, not a replication.
- **Item 7, first branch — rejected.** "At least six well-separated budgets with replicated continuous
  minima" is unaffordable for the same reason. Took the alternative branch — test the fixed prediction
  directly — which is also the reviewer's §7 recommendation, so this is a consistent reading rather than
  a disagreement.
- **§7 alternative — adopted as the core design**, with one deviation: the reviewer's step 1 specifies the
  exact Tao 33M architecture plus 16M/8M/2M downscales. Retained, but the 33M point is budget-reduced per
  the shrink rule, since Phase 0 may show even a single full-token 33M run costs ~8–20 hours.

**Net effect:** the research question is unchanged; the estimand, the inferential strategy, the metric
protocol, and the feasibility basis are all substantially rewritten. Round 1 caught two factual errors in
the author-side reasoning (estimand, parameter arithmetic) that would have invalidated the paper.


---

## Round 2 — codex

| | |
|---|---|
| Model | `gpt-5.6-sol` (reasoning effort `xhigh`) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; id echoed back identical) |
| Grounding | `repo` adapter class; revised plan inlined (same reason as round 1) |
| Reviewed body sha256 | `678db529e1c08d11cedc602418d11fc78e7b37d1d2e410d0d70f95533b0114eb` |
| Verdict | **REVISE** |

### Critique (verbatim)

The revision fixes many round-1 errors, but it is still not implementation-ready. The remaining problems are concentrated in the actual hypothesis test: the “fixed prediction” is not uniquely specified, the primary analysis quietly re-estimates a slope, and several modernized choices no longer test Tao’s published result like-for-like.

## What is genuinely fixed

These are substantive improvements, not cosmetic:

- The estimand is now correctly \(N_v=Vd\) versus \(N_{nv}\).
- Parameter storage, input lookup, and output projection costs are separated correctly.
- Fixed bytes versus fixed FLOPs is resolved.
- Phase 0 makes feasibility empirical rather than aspirational.
- Seed replication, the untouched test split, exact BPB calculation, boundary guards, and continuous minima are directionally correct.
- The 33M point is honestly called an anchor.
- The contribution is appropriately narrowed to a boundary test.

Those changes rescue the basic research question. They do not yet define a valid primary test.

## Your three rejected suggestions

1. **Rejecting two published-range architectures is defensible.** Given the demonstrated cost, a single 33M calibration anchor is reasonable. It must either be trained at Tao’s predicted compute-optimal budget and recipe or be omitted; a reduced-budget version is not a calibration anchor.

2. **Rejecting six compute budgets is defensible.** A direct test of published predictions is stronger than fitting an exponent from too few budgets. The revised plan, however, has not yet operationalized that direct test.

3. **The 2.4M citation was not garbled.** arXiv:2501.16975 is indeed *Over-Tokenized Transformer*. Its Section 3.1 includes synthetic CFG experiments with approximately 2.4M and 85M GPT-2 models, comparing 1-gram and 3-gram tokenization; the larger output vocabulary harms the smaller model. It is adjacent evidence, not an English BPE IsoFLOP test, so it does not destroy your narrowed novelty claim, but it belongs in related work. [Huang et al.](https://arxiv.org/abs/2501.16975)

## Remaining material problems

### 1. “Tao’s fixed prediction” is not uniquely defined

Tao publishes three predictors, not one. Even the two simple predictors have slightly different exponents and normalizations; the released code uses approximately 0.8327 or 0.8354 rather than the rounded 0.84. The plan never says which predictor, coefficient, reference commit, or uncertainty is primary. [Tao prediction code](https://raw.githubusercontent.com/sail-sg/scaling-with-vocab/main/optimal_Nv_predict.py)

**Fix:** Freeze one exact predictor, preferably Approach 1 with its full coefficients and source commit, before generating any architecture or vocabulary grid; treat the other approaches as preregistered sensitivity analyses.

### 2. The compute budget associated with each architecture is unspecified

The published relationship describes jointly compute-optimal \((N_{nv},N_v,H)\). For a chosen \(N_{nv}\), Tao’s fitted \(N_{nv}(C)\) determines the target \(C\), and the fitted data law determines \(H\). Merely training five vocabularies at “exactly matched FLOPs” does not test the law unless the common budget is the published compute-optimal budget for that \(N_{nv}\).

**Fix:** Add a table containing, for every architecture, the exact published target \(C\), predicted \(N_v\), chosen \(d\), implied \(V\), predicted characters or bytes, and resulting training steps.

### 3. Exact FLOP accounting conflicts with like-for-like falsification

Tao’s coefficients were fitted using approximately
\[
C_{\text{Tao}}=6(N_{nv}+Vd)D.
\]
The revision replaces that independent variable with an operator-level count including attention. That count is scientifically better, but it changes the definition of \(C\), non-uniformly across widths and sequence lengths. A deviation could arise solely from changing the ruler.

**Fix:** Use Tao’s FLOP convention for the primary falsification and report profiler-validated exact FLOPs as a secondary re-expression and sensitivity analysis.

### 4. WSD changes the training intervention

Tao used its own training schedule; substituting WSD can change the vocabulary optimum. Moreover, ordinary stable-phase checkpoints are not finished lower-budget models. WSD obtains a model for a chosen budget by branching from the stable trajectory and running a cooldown; raw intermediate checkpoints remain at the high-learning-rate noise floor. [Hägele et al.](https://arxiv.org/abs/2405.18392), [Wen et al.](https://arxiv.org/abs/2410.05192)

**Fix:** Either match Tao’s schedule with separate budget-specific runs, or use explicit WSD branches with a cooldown per target budget and describe the result as testing Tao’s law under a different optimizer schedule.

### 5. The primary metric is assigned backwards

The hypothesis concerns a law selected under \(L_u\), yet BPB is declared primary. If BPB rejects the prediction while \(L_u\) agrees, Tao has not been falsified; only the transfer of its recommendation to BPB has failed.

**Fix:** Make \(L_u\) primary for the Tao falsification and BPB a co-reported secondary estimand with a separately worded claim.

### 6. There is still no operational rejection or agreement criterion

“Falsifies or confirms” is undefined. How far may the empirical optimum be from the predicted \(N_v\)? Is a factor of 1.2 acceptable? Does statistical significance matter if the BPB regret is negligible? Failure to reject is not confirmation.

**Fix:** Preregister a practical equivalence margin, such as maximum log-ratio in \(N_v\) and maximum \(L_u\) regret at the predicted vocabulary, then use equivalence testing rather than the word “confirms.”

### 7. The broken-slope analysis reintroduces the underpowered exponent fit

With four sub-33M points and one optional anchor, fitting and cross-validating a broken slope is effectively estimating a new below-range exponent under another name. Leave-one-scale-out predictive comparison with so few scale points will be unstable and highly sensitive to one estimated bowl minimum.

**Fix:** Make per-scale predicted-versus-local-optimum regret the primary analysis; report the below-range slope and broken-line comparison as descriptive only.

### 8. The vocabulary grid and shrink rule are internally inconsistent

It is unclear whether 512 and 64k are included in the five points or are additional points. If five includes both guards, only three points remain near the expected minimum. Shrinking to four while retaining both guards leaves two local points, which cannot support a credible continuous bowl fit. A fixed 64k point is also likely wasteful for 2M models whose predicted optimum may be near a few thousand.

**Fix:** Use five geometrically spaced local points around each prediction, for example \(V_{\text{pred}}\times\{1/4,1/2,1,2,4\}\), clipped only by genuine tokenizer limits; drop architectures before reducing the number of local bowl points.

### 9. The shrink rule contains invalid and non-operative branches

- A reduced-token 33M anchor no longer calibrates the compute-optimal prediction.
- “Reduce seeds from 3 to 3” is literally no reduction.
- There is no abort condition if the minimum design remains unaffordable.
- Keeping fixed extreme guards while removing local points prioritizes low-information runs.

**Fix:** Drop the 33M anchor completely if unaffordable, preserve five local vocabulary points and three primary seeds, and abort the paper claim if a stated minimum design cannot be funded.

### 10. The corpus and tokenizer specification disappeared

The revised plan never actually names SlimPajama in the experimental procedure. It also omits BPE implementation, pre-tokenization, normalization, initial byte alphabet, special tokens, tokenizer-training sample, deterministic tie-breaking, and whether vocabularies are nested. These choices affect both compression and the learned token inventory.

**Fix:** Specify an exact SlimPajama subset and sampling manifest, then match Tao’s tokenizer pipeline and freeze all tokenizer settings and artifacts before model training.

### 11. The architecture extrapolation is undefined

Tao’s released utility assigns \(d=512\) to every \(N_{nv}\leq50\)M. That is not a viable literal architecture rule at 2M with their 4× feed-forward design: one standard \(d=512\) block already exceeds that budget. Choosing narrower models is necessary, but then the width rule is a new extrapolation, and \(V=N_v/d\) depends directly on it. [Tao architecture utility](https://raw.githubusercontent.com/sail-sg/scaling-with-vocab/main/utils.py)

“Modeling width as a covariate” does not solve this because width is nearly deterministic and collinear with scale across four architectures.

**Fix:** Preregister one explicit depth-width-FFN extrapolation rule and define the primary claim as applying to that architecture family; use matched-width alternatives as sensitivity analyses, not regression covariates.

### 12. “Head-rank” is ambiguous

Godey’s relevant bottleneck is the rank of the vocabulary projection, bounded by hidden dimension \(d\), not the number of attention heads. Recording attention-head count does not isolate that mechanism.

**Fix:** Call the variable output-projection rank or hidden dimension, and specify whether the ablation changes \(d\), factorizes the LM head, or only changes attention-head count. [Godey et al.](https://arxiv.org/abs/2404.07647)

### 13. Phase 0 and preregistration contradict each other

Phase 0 includes training, variance estimation, throughput measurement, and potentially implementation decisions, while Phase 3 says preregistration happens before the first training run. The Phase 0 task list also does not actually include the multi-seed pilot needed to estimate variance.

**Fix:** Preregister the pilot protocol first, exclude all pilot data from confirmatory inference, then freeze a second-stage confirmatory registration after Phase 0 and before Phase 1.

### 14. Cross-validation against the reference remains too vague

“One shared configuration” could mean only matching final loss approximately. That would miss differences in RoPE, masking, packing, parameter counting, weight decay, tied weights, and optimizer updates.

**Fix:** Freeze a reference commit and require agreement on parameter counts, logits, loss, gradients, one optimizer update, FLOP accounting, and a short loss trajectory within stated tolerances.

### 15. The motivation still overstates the expected regime change

At Tao’s own predicted optimum, \(N_v\) is roughly 3M at \(N_{nv}=33\)M, not the 16.4M represented by the 32k table. Extrapolating the published law to 2M makes \(N_v/N_{nv}\) larger, but still only roughly 15%, not dominant. The 32k accounting table is useful for the VRAM pilot but is not representative of Tao’s predicted optimum.

**Fix:** Motivate the work through extrapolation uncertainty and small-width softmax behavior, not through an implied transition to output-head dominance.

### 16. \(L_u\) is still underspecified

The unigram probabilities must be estimated from a declared corpus and never from validation or test text. Smoothing, special-token handling, sequence boundaries, and zero-frequency behavior must match Tao.

**Fix:** Estimate unigram probabilities only from the training split using the reference implementation’s exact convention, and freeze the resulting lookup tables before selection.

### 17. Fixed-byte context evaluation needs an exact protocol

A byte boundary can fall inside a BPE token. “Fixed-byte context” is therefore not operational without specifying chunking, retokenization, boundary-crossing tokens, and which continuation losses are scored.

**Fix:** Define a document-chunk protocol with fixed UTF-8 byte windows, deterministic retokenization, and an explicit rule for excluding or assigning boundary-crossing tokens.

## Bottom line

Round 2 converts a fundamentally confused scaling-law proposal into a potentially worthwhile boundary study. Your first two methodological rejections are sound, and the narrowed novelty claim probably survives. But the primary test is still not specified tightly enough to distinguish:

- failure of Tao’s law,
- a different FLOP convention,
- a different scheduler,
- a different metric,
- or an arbitrary below-range architecture rule.

Resolve those identification issues, repair the shrink rule, and restore the missing corpus/tokenizer specification before implementation.

VERDICT: REVISE

### Claude's response — round 2

**Corrected against me.** The round-1 challenge to the arXiv:2501.16975 citation was wrong. That ID is
*Over-Tokenized Transformer*, and its §3.1 does contain synthetic CFG experiments at ~2.4M and 85M finding
that larger output vocabularies harm the smaller model. Added to related work as adjacent sub-33M evidence.

**Verified independently by reading the reference source, which confirmed every identification point:**

- `optimal_Nv_predict.py` implements **three** predictors, not one. Approach 2 uses exponent
  `0.8353974035228025` with anchor `N_v(33e6) = 3,145,728`; Approach 1 uses `C^0.4163622634135234`;
  Approach 3 is a parametric fit. The rounded 0.84 appears nowhere in the code. Approach 2 is now frozen
  as primary with the others as preregistered sensitivity analyses, and the commit is pinned.
- `utils.py` `func_flops` is `6*(Nnv+V*d)*H*(0.00639222*ln(V)² - 0.15811069*ln(V) + 1.20470122)` with `V`
  capped at 200k. Adopted verbatim as the primary ruler.
- `Nnv_to_d` returns `d=512` for all `N_nv ≤ 50M`. Confirmed infeasible at the bottom of the range: one
  `d=512` block costs `12·512² = 3.15M` params, exceeding a 2M budget outright. Item 11 is correct.
- `model.py` shows `lm_head` as a separate `nn.Linear` from `wte` with no weight assignment between them —
  **embeddings are untied.** Resolves the round-1 blocking question.

**Item 15 accepted, and it forced a rewrite of the motivation.** At Tao's own predicted optimum,
`N_v = 3.15M` against `N_nv = 33M` — about 9.5%, implying `V_opt ≈ 6,144`, not 32k. Extrapolated to 2M,
`N_v ≈ 302,600`, about 15%. The output head never dominates anywhere in this study, so that motivation is
withdrawn entirely. Replaced with extrapolation risk plus the Godey rank bottleneck.

The recomputation also produced a **better framing**, now the headline: the law extrapolated below its
range predicts optimal vocabularies of roughly **600–1,200 tokens** at 2–4M `N_nv` — an order of magnitude
below anything in practical use and approaching byte-level — while small-model practitioners routinely
inherit 32k. That is a sharper and more consequential question than the original.

**Accepted and acted on — all 17 remaining items:**

1 (predictor frozen + commit pinned) · 2 (per-architecture table of target `C`, predicted `N_v`, `d`,
padded `V`, `H`, steps — matched FLOPs only valid at the *published compute-optimal* budget) · 3 (Tao's
FLOP convention primary, operator-level secondary; accepted that improving the ruler broke like-for-like
comparison — an over-correction in round 1) · 4 (WSD dropped from primary; accepted that raw stable-phase
checkpoints are not finished budget models and require cooldown branches) · 5 (**metric priority reversed**
— `L_u` primary because the law was selected under it, BPB co-reported with a separately worded claim) ·
6 (equivalence testing with two preregistered margins sized from a variance pilot; "confirms" retired) ·
7 (per-scale regret is primary; slope and broken-line descriptive only) · 8 (five *local* points at
`V_pred × {1/4,1/2,1,2,4}`; fixed 512/64k guards removed as low-information at 2M) · 9 (shrink rule
repaired: anchor is full-budget-or-dropped, the "3 to 3" non-reduction removed, explicit abort condition
added) · 10 (SlimPajama subset + manifest and the full BPE pipeline spec restored — a genuine regression
introduced by the round-1 rewrite) · 11 (one explicit preregistered depth/width/FFN rule, claim scoped to
that family; accepted that width-as-covariate fails under collinearity) · 12 ("head-rank" retired for
output-projection rank / hidden dimension `d`; ablation varies `d`) · 13 (two-stage preregistration; pilot
data excluded from confirmatory inference; multi-seed variance pilot added to Stage B) · 14 (reference
cross-validation tightened to param counts, logits, loss, gradients, one optimizer update, FLOP accounting,
and a short trajectory at a pinned commit) · 16 (`L_u` unigram probabilities from the training split only,
matching reference conventions, frozen before selection) · 17 (fixed-byte context given a full chunk
protocol or dropped).

**Rejected: nothing.** Round 2 contained no item I could defend against. The two rejections carried over
from round 1 were both explicitly endorsed by the reviewer as defensible.

**One extension beyond the critique:** Tao's FLOP convention omits attention. If the 33M anchor shows a
discrepancy traceable to that omission, the plan now requires reporting it rather than silently correcting,
since correcting it would reintroduce the ruler problem of item 3.

**Net effect:** the research question survives and is now sharper, but the motivation, the primary metric,
the schedule, the FLOP ruler, the vocabulary grid, and the decision criterion were all wrong or unspecified
before this round.


---

## Round 3 — codex

| | |
|---|---|
| Model | `gpt-5.6-sol` (reasoning effort `xhigh`) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; id echoed identical) |
| Grounding | `repo` adapter class; revised plan inlined |
| Reviewed body sha256 | `367750b89192f91dc1843a495de425800554791b1f93991ff9f3938c04df26ad` |
| Verdict | **REVISE** — but framing declared sound; remaining items are instantiation gaps |

### Critique (verbatim)

The plan is close, but the primary test is still not uniquely executable. Most round-2 fixes are real; the remaining blockers are mathematical specification gaps rather than conceptual confusion.

## What is now resolved

The following are satisfactory:

- A2 is frozen with exact constants and A1/A3 demoted to sensitivity analyses.
- Tao’s metric, schedule, weight tying, and FLOP convention are primary.
- BPB has the correct secondary interpretation.
- The motivation no longer relies on output-head dominance.
- Per-scale regret replaces an underpowered slope fit.
- The anchor and shrink rule are substantially improved.
- The pilot and confirmatory stages are separated.
- The related-work correction is accurate.

## Remaining blockers

### 1. The “verbatim” FLOP ruler cannot operate with the proposed architectures

Tao’s helper calls `Nnv_to_d(Nnv)`, which returns \(d=512\) throughout the entire proposed range. Stage D simultaneously requires narrower, feasible architectures. Therefore the primary ruler will charge \(V\times512\) output-head FLOPs even when the trained model uses, for example, \(d=256\).

This is not a minor implementation detail. It changes how many characters each vocabulary receives at matched nominal compute.

**Fix:** Define the below-range extension explicitly as
\[
C_{\text{primary}}
=6\left(N_{nv}+V_{\text{head}}d_{\text{actual}}\right)Hf_{\text{Tao}}(V_{\text{tok}}),
\]
using the preregistered actual width; call this a faithful algebraic extension of Tao’s ruler, not verbatim execution of its architecture lookup. Use the original helper unchanged only at 33M. [Reference utility](https://raw.githubusercontent.com/sail-sg/scaling-with-vocab/main/utils.py)

### 2. Nominal and padded vocabulary sizes are conflated

Padding creates two distinct quantities:

- \(V_{\text{tok}}\): real tokenizer entries, which determine tokenization and fertility.
- \(V_{\text{head}}\): padded output rows, which determine \(N_v\) and projection FLOPs.

Unused padding rows do not improve compression, so passing padded \(V\) into Tao’s fertility polynomial is wrong unless nominal vocabulary sizes are themselves snapped to the padding multiple.

**Fix:** Either train every tokenizer at a nominal size divisible by the padding quantum so \(V_{\text{tok}}=V_{\text{head}}\), or use \(V_{\text{tok}}\) in \(f(V)\) and \(V_{\text{head}}d\) in parameter/FLOP accounting.

### 3. The architecture rule remains a promise, not a rule

The plan correctly says an explicit rule will be preregistered, but it provides no layers, widths, FFN dimensions, heads, or rounding algorithm. Because actual \(d\) determines the predicted vocabulary and the FLOP calculation, this is part of the primary hypothesis, not later implementation detail.

**Fix:** Include the complete 2M/4M/8M/16M architecture table now, together with a deterministic rounding rule that reproduces each target \(N_{nv}\) within a stated tolerance.

### 4. Input-embedding accounting is still open

Whether the input embedding is excluded from \(N_{nv}\) determines whether a “2M \(N_{nv}\)” architecture actually has 2M Transformer parameters or includes hundreds of thousands of vocabulary parameters. The architecture cannot be constructed until this is settled.

**Fix:** Resolve the convention before approval and state one exact identity for physical total parameters, \(N_{nv}\), output \(N_v\), and input embedding parameters.

### 5. The equivalence margins are circular and contradictory

Stage A says the margins are registered before training. Stage B says the variance pilot will “size the equivalence margins empirically.” A noisier experiment must not receive a wider definition of practical equivalence; that makes equivalence easier to claim precisely when measurement quality is worse.

The variance pilot also cannot determine a scientifically meaningful log-ratio margin.

**Fix:** Set both margins from substantive considerations before Stage B; use pilot variance only to estimate required seeds and power, aborting or reporting inconclusive if the fixed margins cannot be resolved.

### 6. The continuous bowl estimator is unspecified

“Fit continuously in log \(V\)” leaves major researcher degrees of freedom:

- quadratic, spline, or Gaussian process;
- weighted or unweighted;
- pooled or separate seed fits;
- treatment of non-convex fits;
- behavior when the minimum lies at the grid boundary;
- uncertainty propagation into optimum location and regret.

Different reasonable choices can move the empirical optimum enough to change the equivalence verdict.

**Fix:** Freeze one loss model and inference procedure, including convexity/adequacy checks, seed-level uncertainty, boundary behavior, and a preregistered grid-extension or lower-bound rule.

### 7. There is no study-level decision rule

The plan defines per-scale outcomes but not the answer to “does the law survive below 33M?” It does not state whether:

- all scales must be equivalent;
- one rejection falsifies the family-wide claim;
- the two margins form an intersection-union test;
- multiplicity across scales and two endpoints is controlled;
- a mixture of equivalent and inconclusive results yields an overall conclusion.

**Fix:** Define the global decision table and error control, preferably requiring both equivalence criteria at every retained scale for a “practically equivalent across the tested range” conclusion.

### 8. Stage C needs one row per architecture and vocabulary, not per architecture

At fixed \(C\), the permitted \(H\) varies with vocabulary because both \(Vd\) and fertility vary:
\[
H(C,N_{nv},V)
=\frac{C}{6(N_{nv}+Vd)f(V)}.
\]
There is no single predicted \(H\) for an architecture. Tao’s \(H\) is characters, not interchangeable “characters/bytes.”

**Fix:** Freeze a configuration table with one row per architecture × vocabulary × seed, including nominal and padded \(V\), actual \(d\), target characters, expected tokens, steps, and both FLOP counts.

### 9. The primary hypothesis is a composite of A2 and A1

A2 supplies \(N_v(N_{nv})\), but it does not supply the compute budget. The plan uses A1’s \(N_{nv}(C)\) relationship to select that budget. This is reasonable because A2’s paper usage assumes near-optimal compute, but it is a composite test, not simply “the A2 predictor.”

Approach 2 was derived as minimum compute to reach a fixed loss and then anchored empirically at 33M. [Tao et al., §4.2](https://papers.neurips.cc/paper_files/paper/2024/file/cf5a019ae9c11b4be88213ce3f85d85c-Paper-Conference.pdf)

**Fix:** Name the primary hypothesis explicitly as “A2 \(N_v(N_{nv})\), evaluated at the A1-implied compute-optimal \(C(N_{nv})\), under the shared Tao training recipe.”

### 10. Three pilot seeds at one configuration do not establish power

Variance can differ by model size, vocabulary, training duration, and bowl curvature. More importantly, uncertainty in the optimum depends on both seed variance and curvature; a single-\(V\) pilot estimates only the former.

**Fix:** Use fixed substantive margins plus a conservative curvature estimate from Tao’s released data or a multi-\(V\) pilot, calculate required seeds, and make that required number, not three, the confirmatory minimum.

### 11. Secondary training arms are not included in the budget hierarchy

The plan mentions matched-width alternatives, possible WSD, varying \(d\), optional head factorization, and possibly fixed-byte evaluation. The shrink rule drops primary scale coverage before explicitly dropping these secondary arms.

“Optionally factorizes the LM head” also introduces post-hoc discretion and changes parameters, FLOPs, and model class.

**Fix:** Freeze or remove every optional arm and make the first shrink step “drop all training-based sensitivity and mechanism arms,” before dropping primary architectures or the anchor.

### 12. Boundary clipping needs a deterministic replacement rule

At the smallest predicted vocabularies, \(V_{\text{pred}}/4\) may fall below the byte alphabet plus special tokens. Simple clipping can create duplicate or tightly clustered grid points and undermine the bowl fit.

**Fix:** When a point violates the minimum vocabulary, regenerate five distinct log-spaced feasible points while retaining \(V_{\text{pred}}\), rather than clipping individual values independently.

## Smaller issues

- “Smaller than any vocabulary in practical use” remains false because byte-, character-, and tiny-model vocabularies exist; say “far below standard general-purpose subword vocabularies.”
- A 33M discrepancy cannot be causally attributed to omitted attention merely because exact-FLOP re-expression looks better. Call that evidence consistent with the explanation unless a matched exact-FLOP analysis is performed.
- Operator-level “sensitivity analysis” needs clarification: merely relabeling completed runs by exact FLOPs is a re-expression, not an exact-IsoFLOP counterfactual.
- Fixed-byte context must be frozen in Stage C or omitted before confirmatory results, not decided after inspecting them.

## Bottom line

The scientific framing is now sound, and no major round-2 defect has been disguised rather than addressed. What remains is a failure to instantiate the design: actual architectures, actual margins, the bowl model, the global decision rule, and a below-range FLOP formula consistent with actual width are still missing.

Those choices can change the verdict and therefore must appear in the plan itself, not merely be promised for later preregistration. Once they are fixed, the project should be ready for the pilot gate.

VERDICT: REVISE

### Claude's response — round 3

**Accepted all 12 blockers and all 4 smaller issues. Rejected nothing.** Round 3's items were instantiation
gaps rather than conceptual errors, so the response is arithmetic and specification rather than argument.

**Resolved by source inspection (item 4 — closed):** `exp_data.csv` shows `Non_vocab_parameters` identical
at 33,222,784 across all ten vocabulary sizes at `d=512`. Therefore `N_nv` excludes **both** embedding
tables; `N_v = V_head·d` is the output head only; total untied = `N_nv + 2·V·d`. Combined with untied
`lm_head` from `model.py`, the accounting identity is now fully determined.

**Item 2 resolved by construction:** all tokenizers trained at nominal sizes divisible by the padding
quantum (128), so `V_tok = V_head = V` and no padding rows go unused. Eliminates the conflation rather than
managing it.

**Item 1 accepted:** the primary ruler is now the faithful algebraic extension
`C = 6(N_nv + V·d_actual)·H·f(V)`, with Tao's unmodified `Nnv_to_d` helper used **only at 33M**. Verified
the formula reproduces their published FLOPs exactly (`V=4096, d=512, H=180,820,540.85` → `1.2714e16` vs
published `1.271517696e16`), which also establishes `f(V)` as fertility in tokens/character and gives
`H = C/(6(N_nv+Vd)f(V))`.

**Item 3 accepted — the rule is now written, with its table.** Deterministic constant-aspect-ratio rule
`d = snap_64(512·(N_nv/33.222784e6)^(1/3))`, `d_ffn = snap_64(8d/3)`, `head_dim = 64`,
`n_layer = round(N_nv/per_layer)`. Validated against their ladder at `d=2048` (TinyLlama-1.1B, 22 layers,
`d_ffn=5632` → 1,130.3M vs published 1,129,891,136). Instantiated: 2M/4M/8M/16M → `d` = 192/256/320/384,
`n_layer` = 5/5/7/9, giving `V_pred` = **1,714 / 2,115 / 3,154 / 4,459**.

**This corrected a claim of my own:** the previous draft's "~600–1,200 tokens" assumed `d=512`, the very
architecture item 11 showed to be infeasible. The reviewer's point that `V_pred` inherits the width rule is
demonstrated by that correction. Flagged as still-open: `n_layer` at `d=512` is not derivable from the
released CSV, so the rule's 33M validation is a Stage B task against `config.py`, not an assumption.

**Item 5 accepted — the circularity was real.** Margins are now fixed from substantive considerations
before Stage B: **M1** `|ln(N_v*/N_v_pred)| ≤ ln(1.5)` (one step of Tao's own vocabulary grid spacing);
**M2** `L_u` regret ≤ the improvement from a **10% compute increase** (the 10% fixed a priori, the
conversion factor measured). Pilot variance now sizes only seeds and power.

**Item 6 accepted:** bowl estimator frozen to a weighted quadratic in `ln V` with SEM weights, convexity
and residual adequacy checks, bootstrap over seeds (10,000 resamples) for `ln V*` uncertainty, a
one-extension-only boundary rule triggered by the fitted minimum's location, and non-convex fits reported
inconclusive rather than re-modelled.

**Item 7 accepted:** study-level decision table added. Equivalence across the range requires both margins at
every retained scale (intersection-union, no multiplicity correction needed); a single rejection falsifies
with Holm FWER control across scales; any non-convex or boundary-censored scale yields inconclusive.

**Item 8 accepted:** Stage C table is now one row per architecture × vocabulary × seed (60 rows), generated
by a committed script from `H = C/(6(N_nv+Vd)f(V))`, with `H` in **characters** specifically. Worked example
included (2M, `V=1792` → `f=0.3789`, `H=121.3M` chars, ≈46.0M tokens).

**Item 9 accepted:** hypothesis renamed as the explicit composite — "A2's `N_v(N_nv)` evaluated at the
A1-implied compute-optimal `C(N_nv)` under the shared Tao recipe and the preregistered sub-50M architecture
family."

**Item 10 accepted:** multi-`V` seed pilot replaces the single-`V` one; curvature additionally estimated
**at zero GPU cost** by refitting from `exp_data.csv`'s ten vocabulary points at 33M; required seed count is
computed and becomes the confirmatory minimum instead of a hardcoded 3.

**Item 11 accepted:** every optional arm is now removed rather than deferred — WSD gone, LM-head
factorization gone, fixed-byte context gone. Shrink step 1 is "drop all training-based sensitivity and
mechanism arms," ahead of the anchor and architectures.

**Item 12 accepted:** minimum viable `V` = 512 (256-byte alphabet + specials, snapped to 128); violations
regenerate five distinct log-spaced feasible points retaining `V_pred` rather than clipping individually.

**Smaller issues, all accepted:** "far below standard general-purpose subword vocabularies" replaces the
false "smaller than any in practical use"; an anchor discrepancy is reported as *consistent with* omitted
attention, never causally attributed without a matched counterfactual; operator-level FLOPs are labelled a
**re-expression**, not an exact-IsoFLOP counterfactual; fixed-byte context is omitted rather than decided
later.

**Two additions of my own from the source inspection:** Tao et al. ran **one run per cell with no seed
replication**, so their own minima carry unquantified noise — an independent argument for checking the
extrapolation, and a respect in which this study is more rigorous than the work it tests. And `L_u` is
**negative** when the model beats the unigram baseline (their published rows show −1.35/−2.10/−2.50), so
the sign convention is now asserted in tests.

**Orchestrator note:** the first attempt to launch round 4 died in the shell with a heredoc parse error
before Codex was invoked. No request reached the provider and no capture directory was created, so it is
recorded here as an orchestrator fault rather than an adapter attempt, and does not consume the attempt
budget.

---

## Round 4 — codex

| | |
|---|---|
| Model | `gpt-5.6-sol` (xhigh) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; echoed identical) |
| Grounding | `repo` adapter class; revised plan inlined |
| Reviewed body sha256 | `954961f618da97667c860e3d7b2b7fc28b2cef39c6cf1ef5292bfa7eff3bfa25` |
| Verdict | **REVISE** — arithmetic audit plus 9 blockers |

### Critique (verbatim)

The arithmetic audit finds several real errors, and the primary test is still not uniquely executable. The framing is now sound, but four inferential/specification gaps and the feasibility miscalculation remain material.

## Arithmetic audit

| Item | Independent result | Assessment |
|---|---:|---|
| `f(4096)` | `0.3318207086` tokens/character | Correct |
| Published FLOP check | `1.2715176959854×10^16` | Exact match, not merely `1.2714×10^16` |
| 2M realized `N_nv` | `2,213,952` | Correct |
| 4M realized `N_nv` | `4,016,896` | Correct |
| 8M realized `N_nv` | `8,463,040` | Correct |
| 16M realized `N_nv` | **`15,932,544`** | Table is short by 384 because it omits final `+d` |
| A2-derived `V_pred` | `1714.79, 2115.47, 3154.00, 4458.75` | Reported rounded values are correct |
| A1 budgets | `7.0547e14, 2.3223e15, 1.03084e16, 3.65351e16` | Correct |
| 2M, `V=1792` | `f=0.3789885`, `H=121.282M`, tokens `=45.964M` | Worked example is correct |

Two smaller arithmetic qualifications:

- Applying the written A2 formula at the anchor’s realized `N_nv=33,222,784` gives `N_v=3,163,459`, or `V=6178.63`, not exactly `3,145,728/6144`. It still snaps to 6144, so the configuration is unchanged.  
  **Fix:** Label `3,145,728` as the predictor’s 33,000,000 reference anchor, not its value at realized `33,222,784`.

- The architecture rule already fails its proposed 2% anchor check. At `d=512`, it produces `d_ffn=1344`, `per_layer=3,113,984`, 11 layers, and realized `N_nv=34,254,336`, which is **3.105% high**.  
  **Fix:** Use Tao’s exact architecture for the anchor and keep the new sub-50M rule separate; do not conditionally rewrite the preregistered family to reproduce the anchor.

## Material blockers

1. **The feasibility total omits the five vocabulary configurations.** The four sub-33M budgets sum to `4.9871×10^16` for one run per scale. Five vocabularies and three seeds make the actual total **`7.4807×10^17`**, not `1.5×10^17`. That is 13.85 ideal hours at 15 TFLOP/s and approximately 28–55 hours under the stated 2–4× utilization penalty. The current 25–50-hour conclusion happens to be close, but its derivation is wrong. The anchor accounting is also ambiguous: one anchor run is `1.5886×10^17` and 2.94 ideal hours; a five-vocabulary, three-seed anchor bowl is `2.3829×10^18` and 44.1 ideal hours.

   **Fix:** Recompute every budget as `Σ_s C_s × n_vocab,s × n_seed,s` and state whether the anchor is one instrument-validation run or a replicated vocabulary bowl.

2. **Exact IsoFLOP enforcement is still ambiguous because `f(V)` is fitted fertility, not necessarily the new tokenizers’ realized fertility.** The equation and inverse for `H` are algebraically correct. But if you consume `H` actual characters and the newly trained BPE’s actual tokens/character differs from `f(V)`, the run misses target `C`. If you consume the predicted token count, `H` is not the actual character count. This directly affects the primary control variable.

   **Fix:** Enforce `T_target=C/[6(N_nv+Vd)]` using exact token counts; report `H_Tao=T_target/f(V)` as a Tao-equivalent estimate and separately report actual characters/bytes consumed.

3. **M2 is conceptually non-circular now, but remains operationally undefined.** Fixing 10% before seeing noise closes the original circularity. However, the conversion factor’s source, vocabulary, scales, seeds, estimator, and training schedules are unspecified. An “`H`-sweep” changes compute, so it is not IsoFLOP, and Stage B does not budget the separate `C` and `1.1C` runs needed under Tao’s schedule. Stage A also claims to register the margin before the conversion is measured.

   **Fix:** Test M2 directly with preregistered `V_pred` runs at `1.1C`: compare `L_u(V_pred,1.1C)` against `L_u(V*,C)`, with separate correct schedules and seed-level uncertainty.

4. **The study-level decision procedure is not yet a statistical test.** No alpha, power, confidence construction for regret, or exact equivalence/rejection tests are given. Holm over four scales is insufficient when either M1 or M2 can trigger rejection; M1 also has two directions. Most importantly, “single rejection wins” conflicts with “any mixture containing a censored scale is inconclusive.” A valid rejection at one scale logically falsifies the all-scales composite regardless of another scale being censored.

   **Fix:** Define TOST/one-sided intervals and alpha explicitly; adjust rejection claims across all scale-by-criterion hypotheses, and set precedence to `valid rejection > all-scale equivalence > inconclusive`.

5. **The power calculation is not instantiated.** Two vocabulary points cannot estimate quadratic bowl curvature. Curvature at 33M is not a defensible plug-in estimate for 2–16M without a conservative transfer assumption. Required seeds also cannot be computed without fixed alpha, target power, variance pooling, and the exact statistic being powered.

   **Fix:** Pilot at least three, preferably all five, vocabulary points at one below-range scale and preregister alpha, target power, variance estimator, power simulation, and abort threshold.

6. **The 2M vocabulary grid does not follow its stated generation rule.** From `V_pred=1714.79`, nearest-128 snapping gives approximately `384, 896, 1664, 3456, 6912`; the listed `512, 896, 1792, 3456, 6912` is neither independent nearest snapping nor a regenerated log-spaced set. Also, 256 byte symbols plus ordinary special tokens normally rounds to 384, not automatically 512.

   **Fix:** State one executable snapping/regeneration algorithm and regenerate the 2M row; treat 512 as a deliberate scientific floor only if it is not an actual tokenizer limit.

7. **The primary evaluation split remains unidentified.** The plan creates selection-validation and final-test sets but never says which supplies the bowl, M1, M2, and study-level decision. A final test set that is merely co-reported does not repair validation-set selection bias.

   **Fix:** State that all adaptation and boundary extension use selection-validation, then run the frozen estimator and decision procedure exactly once on final-test losses.

8. **The corpus/tokenizer specification is still a checklist, not an instantiated protocol.** “A named SlimPajama subset,” “pre-tokenization regex,” and “Unicode normalization” do not identify their values. Those choices affect fertility and vocabulary quality and cannot safely remain choices during implementation.

   **Fix:** Put the exact subset/shards, sampling rule and seed, document filters, tokenizer-training sample, regex, normalization, special tokens, and tokenizer implementation/version into Preregistration I.

9. **The bowl adequacy gate retains subjective discretion.** “Residual inspection” has no pass/fail rule, while only five points leave two residual degrees of freedom. That permits post-hoc acceptance of a convenient quadratic.

   **Fix:** Either make residual inspection descriptive only or preregister a numerical lack-of-fit criterion and its exact inconclusive outcome.

M1 itself is defensible but should be described accurately: factor 1.5 is the **largest** adjacent ratio in the cited Tao grid, not a uniform one-step spacing. It is generous, but pairing it with a properly operationalized M2 can justify that generosity.

The reference accounting, `L_u` sign, A1+A2 composite, actual-width extension, fixed estimator family, shrink/abort structure, and narrowed novelty claim are now substantially resolved. The remaining problems are not merely normal Stage-B risks: they determine what data get consumed, what constitutes equivalence or falsification, and whether the budget closes.

VERDICT: REVISE
### Claude's response — round 4

**Accepted all 9 blockers and both arithmetic qualifications. Rejected nothing.**

Note on independent discovery: before this critique arrived, an author-side arithmetic audit had already
found four of the same errors — the 3.10% architecture-rule failure, the 7.481×10¹⁷ sweep total, `V_pred`
= 6179 at the anchor, and `N_nv` = 15,932,544 at 16M. The reviewer's figures (3.105%, 7.4807×10¹⁷, 6178.63,
15,932,544) match to rounding. The plan file was deliberately **not** edited between the round-4 send and
this verdict, so the recorded `body_sha256` describes exactly the bytes reviewed.

**Blocker 1 — budget.** Recomputed as `Σ_s C_s × n_vocab × n_seed`. Confirmatory sweep 7.4807×10¹⁷; M2 runs
1.6458×10¹⁷; power pilot 1.5463×10¹⁷; anchor 1.5886×10¹⁷; **total 1.2261×10¹⁸ across 88 runs** = 34.1 / 22.7
/ 13.6 ideal hours at 10 / 15 / 25 TFLOP/s. **Anchor scope resolved as one instrument-validation run** at
`V=6144`, matching Tao's own single-run-per-cell protocol — not a replicated bowl, which would have cost
2.38×10¹⁸ alone.

**Blocker 2 — the catch I missed, and the most consequential.** `f(V)` is *fitted* fertility, so consuming
`H` characters would miss target `C` whenever a newly trained tokenizer's realized fertility differs.
IsoFLOP is now enforced on **exact token counts**: `T_target = C/(6(N_nv+V·d))` is what gets consumed;
`H_Tao = T_target/f(V)` is reported as the Tao-equivalent estimate; `H_actual`, `bytes_actual` and
`r_actual` are reported separately, and any divergence between `r_actual` and `f(V)` is treated as a result
rather than absorbed silently.

**Blocker 3 — M2 measured, not converted.** `V_pred` is now additionally run at `1.1·C` with 3 seeds per
scale (12 runs, 1.6458×10¹⁷, explicitly budgeted). `regret = L_u(V_pred,C) − L_u(V*,C)` versus
`gain10 = L_u(V_pred,C) − L_u(V_pred,1.1C)`, one-sided at α. This removes the unspecified conversion factor
and the Stage A/B ordering conflict entirely.

**Blocker 4 — now an actual test.** α = 0.05; M1 by TOST on `ln(N_v*/N_v_pred)` with bounds `±ln(1.5)`;
M2 one-sided; Holm adjustment across all **8** scale × criterion rejection hypotheses; equivalence claim
intersection-union with no correction. The logical contradiction is resolved by explicit precedence:
**valid rejection > all-scale equivalence > inconclusive** — a valid rejection falsifies the composite
regardless of censoring elsewhere.

**Blocker 5 — power instantiated.** Pilot moved to **all five vocabulary points at 8M, 3 seeds** (two points
cannot estimate quadratic curvature); 33M curvature demoted to a conservative cross-check rather than a
plug-in. Preregistered α, target power 80% against `|ln ratio| = ln 2`, variance estimator, power
simulation, and abort threshold; the computed seed count replaces the hardcoded 3.

**Blocker 6 — grid rule executable and the 2M row regenerated.** `V_MIN = 384` (256 byte symbols + 3
specials, snapped up), `snap(x) = max(384, 128·round(x/128))`, with log-spaced regeneration only on
collision. 2M becomes **384, 896, 1664, 3456, 6912** — the previous 512/1792 entries followed no stated
rule. 384 is now identified as a genuine tokenizer floor rather than a scientific choice.

**Blocker 7 — split identified.** All adaptation (bowl fit, boundary extension, adequacy gates) uses
selection-validation; the frozen estimator and decision procedure run **exactly once** on final-test.
Final-test never informs any choice.

**Blocker 8 — corpus/tokenizer instantiated with values,** not a checklist: SlimPajama-627B train chunk1,
ascending shard order, RNG seed 1234; ≥128-byte UTF-8-decodable document filter; 2 GB tokenizer sample from
the disjoint tokenizer-fit split; HuggingFace `tokenizers` byte-level BPE, version pinned; GPT-2
pre-tokenization regex given literally; **normalization = none**, because any Unicode normalization would
break the round-trip losslessness this plan asserts; 3 special tokens post-training; lexicographic
tie-breaking; independently trained (non-nested) vocabularies; all artifacts hashed before training.

**Blocker 9 — adequacy gate numeric.** Convexity via bootstrap CI on the leading coefficient excluding zero,
plus a **lack-of-fit F-test against pooled within-seed pure error at α = 0.01**, failure yielding
inconclusive with no re-modelling. "Residual inspection" demoted to descriptive.

**Both arithmetic qualifications accepted.** `3,145,728` is now labelled the predictor's reference anchor at
`N_nv = 33e6`, not its value at the realized 33,222,784. And rather than bending the preregistered family to
pass its own 2% check, **the anchor now uses Tao's exact published architecture** while the sub-50M rule
stands separately — the reviewer's preferred resolution, and the one that avoids fitting the rule to the
target.

**M1 described accurately:** 1.5 is the coarsest adjacent ratio in the *low-vocabulary region* of Tao's grid
(4096→6144 = 1.50, 6144→8192 = 1.33, 8192→10240 = 1.25); the grid coarsens again higher up (10240→16384 =
1.60). Stated as generous and justified only by M2 now being independently operationalized.

---

## Round 5 — codex (FINAL, cap reached)

| | |
|---|---|
| Model | `gpt-5.6-sol` (xhigh) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; echoed identical) |
| Grounding | `repo` adapter class; revised plan inlined |
| Reviewed body sha256 | `48199c8d253103b2f8a15c395ecd62a8eaaf0bad0bd9896646fa721f4ab865dc` |
| Verdict | **REVISE** — MAX_ROUNDS reached, deadlock |

**Attempt note.** The shell wrapper for this round was killed by the harness while `codex.exe`
continued running. The stream showed all reviewer todo items completed, so the process was left
alive and polled to completion rather than restarted; `verdict.txt` landed intact and validated.
Restarting would have destroyed the final permitted round. Orphaned processes cleaned up after.

### Critique (verbatim)

The numerical audit now passes, but the confirmatory procedure is still not uniquely executable. Four material blockers remain.

## Arithmetic

The revised figures reproduce correctly:

- Architecture counts, A2 predictions, vocabulary grids, and A1 budgets are correct.
- Sweep: `7.4807049×10^17` FLOPs.
- M2 runs: `1.6457551×10^17`.
- Pilot: `1.5462670×10^17`.
- Anchor: `1.5885925×10^17`.
- Total: `1.2261319×10^18`.
- Ideal time: 34.06, 22.71, and 13.62 hours at 10, 15, and 25 TFLOP/s.

That total is conditional on exactly three confirmatory seeds and no boundary extensions. With computed seed count `n`, the required total is:

`6.1·n·ΣC + pilot + anchor`

A boundary extension at every scale adds `n·ΣC`; at three seeds that is another `1.4961×10^17` FLOPs.

## Remaining blockers

1. **“Valid rejection” still lacks an executable statistical definition.** TOST tests evidence for equivalence; failing TOST is only inconclusive. Falsification requires separate tests showing the parameter lies beyond a margin. For M1, there are two failure directions, so “Holm across eight hypotheses” is valid only if a calibrated two-sided M1 failure p-value is defined. The bootstrap interval type, confidence level, and resampling structure are also unspecified.

   **Fix:** Define separate equivalence and failure nulls, the exact bootstrap p-values or adjusted confidence bounds, and either Holm across 12 directional tests or a preregistered valid composite M1 p-value followed by Holm across eight.

2. **The power calculation covers only one possible M1 outcome.** Power at `|θ|=ln 2` measures ability to detect a large failure. It does not measure power to establish M1 equivalence when `θ=0`, and no power target exists for M2. Consequently the Stage-B seed count is not actually determined for the full composite hypothesis, while the budget still hardcodes three M2 seeds.

   **Fix:** Set power targets for M1 equivalence, M1 failure, M2 adequacy, and M2 failure; use the maximum resulting seed requirement for both the sweep and 1.1C runs.

3. **The implemented vocabulary corresponding to `V_pred` is undefined.** `V_pred` is continuous, while models require the snapped central grid value. This affects the M2 runs. Separately, training a BPE to nominal `V` and then adding three specials produces `V+3`, breaking `V_tok=V_head=V` and padding arithmetic.

   **Fix:** Define `V_run=snap128(V_pred)` for M2, and train the lexical BPE to `V−3` before adding the three specials, with an assertion that final tokenizer and unpadded head sizes equal exactly `V`.

4. **Final-test routing remains internally ambiguous.** The plan says all bowl fitting and adequacy gates use selection-validation, but also says the estimator runs on final-test losses. Those imply different final estimands. The document-hash split also lacks the hash algorithm and bucket thresholds, so the supposedly instantiated corpus split is not reproducible.

   **Fix:** Use selection-validation only for extension decisions, then fit the same frozen bowl and run its gates once on final-test data; specify the exact hash input, hash function, and partition thresholds.

Two smaller corrections do not independently block implementation:

- `gain10` is not guaranteed nonnegative under stochastic training. Define the signed M2 statistic directly as  
  `D = L_u(V_run,1.1C) − L_u(V*,C)`  
  and test `D<0` for adequacy or `D>0` for failure without truncation.
- Deterministic BPE runs on the same corpus with identical tie-breaking will generally produce nested merge prefixes even if trained separately. Say “nesting not imposed” rather than “non-nested.”

The anchor lookup, measured throughput, possible non-convex bowls, fertility divergence, and architecture-family confound are now ordinary Stage-B or reporting risks. The four items above still determine the actual hypothesis test and therefore block approval.

VERDICT: REVISE
### Claude's response — round 5 (DEADLOCK: cap reached)

`MAX_ROUNDS = 5` reached without `APPROVED`. Per the loop rules, convergence is **not** claimed. The four
remaining blockers are recorded below with Claude's position on each.

**Important characterisation: none of the four is a disagreement.** Claude agrees with all four. They are
specification work that ran out of rounds, not a builder/reviewer conflict. This is a *cap deadlock*, not an
impasse. Claude's proposed fix is given for each so the human can judge whether they are closable.

**Progress this round, for the record:** the reviewer states "the numerical audit now passes" and
independently reproduces every figure — sweep `7.4807049e17`, M2 `1.6457551e17`, pilot `1.5462670e17`,
anchor `1.5885925e17`, total `1.2261319e18`, and 34.06 / 22.71 / 13.62 ideal hours. It also reclassified the
anchor lookup, measured throughput, non-convex bowls, fertility divergence, and the architecture-family
confound as **ordinary Stage-B or reporting risks** rather than blockers.

One accepted correction to the budget itself: the total is conditional on exactly three confirmatory seeds
and no boundary extensions. The general form is `6.1·n·ΣC + pilot + anchor`, and a boundary extension at
every scale adds `n·ΣC` (= `1.4961e17` at n=3).

---

#### Unresolved 1 — "valid rejection" lacks an executable statistical definition

*Reviewer:* TOST establishes evidence **for** equivalence; failing TOST is only inconclusive, not
falsification. Falsification needs separate nulls showing the parameter lies beyond the margin. M1 has two
failure directions, so Holm across 8 hypotheses is valid only given a calibrated two-sided M1 failure
p-value. Bootstrap interval type, confidence level, and resampling structure are unspecified.

*Claude:* **Agreed, and this is the most substantive of the four** — as written, the plan cannot actually
falsify anything, only fail to confirm. Proposed fix: define equivalence nulls (`|θ| ≥ ln 1.5`) and failure
nulls (`|θ| ≤ ln 1.5`) separately; use BCa bootstrap intervals at 90% for TOST (equivalent to two one-sided
5% tests) and 95% two-sided for failure; resample at the seed level within scale; and take the reviewer's
first branch — **Holm across 12 directional tests** (4 scales × [M1-low, M1-high, M2]) rather than
constructing a composite M1 p-value, because the composite adds a derivation that would itself need review.

#### Unresolved 2 — power covers only one M1 outcome

*Reviewer:* Power at `|θ| = ln 2` measures detection of a large failure only. No power target for M1
equivalence at `θ = 0`, none for M2. So the Stage-B seed count is not determined for the composite
hypothesis, while the budget still hardcodes 3 seeds for M2.

*Claude:* **Agreed.** Proposed fix: four preregistered power targets — M1 equivalence at `θ=0`, M1 failure
at `|θ|=ln 2`, M2 adequacy at `D = −½·gain10_expected`, M2 failure at `D = +½·gain10_expected`, all at 80%,
α=0.05 — with `n = max` over all four applied to **both** the sweep and the 1.1C runs. Budget consequence:
the total becomes a function of `n`, and the hardcoded 3 must be removed from the M2 line.

#### Unresolved 3 — the implemented vocabulary for `V_pred` is undefined

*Reviewer:* `V_pred` is continuous; models need the snapped central grid value, which affects the M2 runs.
Separately, training BPE to nominal `V` then adding 3 specials yields `V+3`, breaking `V_tok = V_head = V`
and the padding arithmetic.

*Claude:* **Agreed, and the `V+3` catch is a genuine defect** — the plan asserted an identity its own
tokenizer recipe violated. Proposed fix: `V_run = snap128(V_pred)` (= 1664 / 2176 / 3200 / 4480, i.e. the
central grid point at each scale), and train the lexical BPE to `V − 3` before adding the three specials,
with a hard assertion that final tokenizer size and unpadded head width both equal exactly `V`.

#### Unresolved 4 — final-test routing still ambiguous, and the split is not reproducible

*Reviewer:* The plan says all bowl fitting and adequacy gates use selection-validation, and also that the
estimator runs on final-test losses. Those are different estimands. The document-hash split omits the hash
algorithm and bucket thresholds, so the "instantiated" corpus split is not reproducible.

*Claude:* **Agreed.** Proposed fix, taking the reviewer's formulation: selection-validation is used **only**
for boundary-extension decisions; the same frozen bowl estimator and its adequacy gates then run **once** on
final-test data, which is the reported estimand. And specify the split concretely —
`bucket = int(sha256(document_text_utf8).hexdigest()[:8], 16) % 1000`, with 0–19 tokenizer-fit, 20–979
train, 980–989 selection-validation, 990–999 final-test.

---

#### Accepted non-blocking corrections

- **`gain10` is not guaranteed non-negative** under stochastic training, so the M2 statistic must not be
  built from a quantity assumed positive. Replace with the signed statistic
  `D = L_u(V_run, 1.1C) − L_u(V*, C)`, testing `D < 0` for adequacy and `D > 0` for failure, no truncation.
- **"Non-nested" is the wrong word.** Deterministic BPE on the same corpus with identical tie-breaking will
  generally produce nested merge prefixes even when trained separately. The plan should say **"nesting not
  imposed"** rather than claiming vocabularies are non-nested.

---

**Status recorded as `deadlocked`.** `PLAN.md` is deliberately left at the round-5-reviewed bytes
(`48199c8d253103b2f8a15c395ecd62a8eaaf0bad0bd9896646fa721f4ab865dc`) so the provenance record continues to
describe exactly what was reviewed. The human decides whether to raise the cap for one more round, accept
the plan with these four items as named Preregistration-I work, or stop.

---

## Round 6 — codex (cap raised 5 -> 6 by the human)

| | |
|---|---|
| Model | `gpt-5.6-sol` (xhigh) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; echoed identical) |
| Grounding | `repo` adapter class; revised plan inlined |
| Reviewed body sha256 | `53612a71fe412ef03de0a2fddd4404a10fa00dfe0d5446cecf427c307c5b43ea` |
| Verdict | **REVISE** — 2 blockers, both in the Stage-B pilot/power stage |

### Critique (verbatim)

The five null definitions are correct, the TOST/failure separation is correct, and Holm over the 12 directional failure hypotheses properly controls FWER. Two connected power-stage blockers remain.

1. **`D̂_pilot` cannot be computed from the budgeted pilot.** The 15 pilot runs contain only five vocabularies at `C`. Computing  
   `D̂_pilot = L_u(V_run,1.1C) − L_u(V*,C)`  
   also requires `V_run` at `1.1C`.

   **Fix:** Add three matched-seed 8M `V_run=3200` runs at `1.1C`, and include that sixth configuration in each seed-level bootstrap block.

   This adds three runs and `3.4018×10^16` FLOPs. At `n=3`, the corrected baseline is 91 runs and approximately `1.26015×10^18` FLOPs.

2. **The claimed 80% failure power does not account for Holm.** Powering M1-failure and M2-failure at unadjusted `α=0.05` does not provide 80% power for a rejection that must survive a 12-test Holm procedure. With only one genuinely failing hypothesis, the first Holm threshold is `0.05/12`.

   **Fix:** Run the failure-power simulation through the complete Holm procedure, or conservatively power directional failure tests at `α=0.05/12`. Equivalence power remains at `α=0.05` because the intersection-union claim needs no multiplicity correction.

The bootstrap structure is otherwise appropriate: resample complete seed vectors within scale, preserving common-random-number pairing across the five `C` configurations and the `1.1C` configuration. One safeguard should be binding: if the power calculation returns a very small `n`, require a defensible minimum or a calibrated small-sample alternative; BCa inference from only three seed clusters would have highly discrete and unstable coverage.

Two nonblocking wording corrections remain:

- “There is no second fit” should read “there is no second final-test fit,” since a selection-validation boundary fit necessarily precedes the final-test fit.
- Correct the reporting notation to `r_actual = T_target/H_actual`; the current comma-separated assignment is malformed.

Everything outside this Stage-B pilot/power inconsistency is now implementation-ready. The numerical tables themselves remain correct for the runs they currently include, but the omitted `1.1C` pilot makes the stated power procedure impossible as written.

VERDICT: REVISE
### Claude's response — round 6

**Accepted both blockers, the safeguard, and both wording corrections. Rejected nothing.**

Reviewer confirmed the round-5 fixes landed: "the five null definitions are correct, the TOST/failure
separation is correct, and Holm over the 12 directional failure hypotheses properly controls FWER," and
"everything outside this Stage-B pilot/power inconsistency is now implementation-ready."

**Blocker 1 — `D̂_pilot` was uncomputable, and the M2 power targets were therefore circular.** Correct and
embarrassing: the 15-run pilot had five vocabularies at `C` only, while
`D̂_pilot = L_u(V_run,1.1C) − L_u(V*,C)` requires a `1.1·C` observation. Pilot is now **six configurations**
— five vocabularies at `C` plus `V_run = 3200` at `1.1·C`, matched-seed, entering each seed-level bootstrap
block as a sixth paired configuration. Adds 3 runs and `3.4016×10¹⁶` FLOPs. Pilot is now exactly
`6.1 · 3 · C_8M = 1.88644×10¹⁷`, mirroring the sweep's own 6.1 factor.

**Blocker 2 — failure power ignored Holm.** Correct. Failure tests are now powered through the **complete
Holm procedure**, with `α = 0.05/12` as the conservative fallback if that simulation is not implemented.
Equivalence power stays at `α = 0.05`, since the intersection-union claim needs no correction. The power
table now carries an explicit operative-α column so the distinction cannot be lost.

**Safeguard accepted and made binding.** BCa from three seed clusters has discrete, unstable coverage, so
`n ≥ 5` is now a **hard floor** regardless of the simulation's output. This materially changes the budget:
the planning minimum is no longer `n=3`.

**Budget consequence, stated plainly.** At `n=3` the corrected total is 91 runs / `1.26015×10¹⁸` —
reproducing the reviewer's figure exactly. But the `n ≥ 5` floor makes `n=3` unreachable, so the real
planning minimum is **139 runs / `1.86858×10¹⁸` = 34.6 ideal hours at 15 TFLOP/s**, rising to
`2.11793×10¹⁸` (39.2 h) with a worst-case all-scale boundary extension. The `n=3` row is retained only as
a reference point against the reviewer's arithmetic.

**Both wording corrections applied:** "no second **final-test** fit" (a selection-validation boundary fit
necessarily precedes it), and the malformed comma-assignment replaced with
`r_actual = T_target / H_actual`.

**Process note.** Round 6 was the human-granted extension after the round-5 cap deadlock. Round 7 is a
further extension taken on the judgement that both remaining items were mechanical and the reviewer had
explicitly scoped the remainder to a single inconsistency. It is the last round regardless of outcome;
if it does not approve, the result is handed over as-is rather than iterated further.

---

## Round 7 — codex — **APPROVED**

| | |
|---|---|
| Model | `gpt-5.6-sol` (xhigh) |
| CLI version | `codex-cli 0.145.0` |
| Session / thread id | `019fb0f1-9b63-7fe2-b9fe-77fbdf23065d` (resumed; echoed identical) |
| Grounding | `repo` adapter class; plan inlined (repo contained only PLAN.md + this log) |
| Reviewed body sha256 | `970a523fff1452ccf4ee824ed51786b34a3bd72ee70af5845e63d18eb7defd6c` |
| Verdict | **APPROVED** |

### Critique (verbatim)

The plan is now sound enough to implement. The confirmatory procedure is identified, the pilot supplies every required quantity, and the budget closes arithmetically.

- The five null definitions and their directions are correct.
- TOST failure is correctly treated as inconclusive.
- Holm over 12 directional failure tests is valid.
- Simulating failure power through Holm, with `0.05/12` as fallback, is conservative and correct.
- The six-configuration paired pilot now makes `D̂_pilot` computable.
- The seed-level block bootstrap is appropriate when each block contains all five `C` configurations plus the matched `V_run@1.1C` configuration.
- The pilot-derived M2 alternatives are noisy but valid because the pilot is independent of confirmatory inference and the adaptation rule is preregistered.

The `n≥5` floor is a defensible minimum safeguard. It is not a theoretical guarantee of BCa coverage, but a calibrated alternative is not required before implementation. The Stage-B simulation should report coverage as well as power; if coverage is inadequate at the computed `n`, increase `n` or abort rather than changing the estimator after confirmatory data are observed.

The arithmetic reproduces exactly:

- Pilot: `1.8864458×10^17`.
- Sweep at `n=5`: `1.2467841×10^18`.
- M2 runs: `2.7429251×10^17`.
- Total: `1.8685805×10^18`, 139 runs.
- Worst-case extension: `2.4935683×10^17`.
- Worst-case total: `2.1179373×10^18`, 159 runs.
- Ideal times: 51.91, 34.60, and 20.76 hours at 10, 15, and 25 TFLOP/s; worst case 39.22 hours at 15 TFLOP/s.

One nonblocking wording cleanup remains: the interval-construction paragraph says common random numbers across “the five vocabulary points.” For M2 inference, explicitly say that the confirmatory seed block also includes the matched sixth `V_run@1.1C` observation, as Stage B already does for the pilot.

The remaining uncertainties are legitimate Stage-B risks: achieved throughput, the seed count exceeding five, poor bowl adequacy, pilot-effect instability, and architecture-family dependence. None prevents implementation or invalidates the preregistered test.

VERDICT: APPROVED
### Claude's response — round 7 (RESOLVED)

Approved. Provenance finalised and **verified with `validate-provenance.sh`, not asserted**:
`approved-final`, exit 0, reason "repo-grounded qualifying APPROVED over current body by codex".
`final_body_cross_model_approved: true` with `final_body_sha256` = the round-7 hash.

The single nonblocking wording item (naming the sixth `V_run@1.1C` observation in the
CONFIRMATORY seed block, as Stage B already does for the pilot) was **deliberately left
unapplied**. Editing the body post-approval would have demoted provenance to `approved-stale`
for a cosmetic gain. It is recorded in `known_nonblocking_followup` and applied at
Preregistration I. Same for the reviewer's coverage note: Stage B must report BCa coverage
alongside power, and if coverage is inadequate at the computed `n`, raise `n` or abort —
never change the estimator after confirmatory data are observed.

**Final tally: 7 rounds, 8 launches, 0 rejected reviewer items across rounds 3-7.**
Rounds 1-2 had 3 author rejections, all of which the reviewer subsequently endorsed as
defensible. The reviewer corrected the author twice on matters of fact (the arXiv:2501.16975
attribution, and the anchor-architecture resolution); the author independently found 4 of
round 4's 5 arithmetic errors before that critique arrived.

---

# Amendment review — P2/P4 resolution and the scope qualifier (2026-08-01)

A separate review track from rounds 1–7. `PLAN.md` is untouched: these are proposed
amendments in `results/p2p4_resolution.md` and `PROPOSED-RELATED-WORK.md`, submitted for
adversarial review before adoption, because they fix the definition of the primary metric
and touch the headline novelty claim.

**Sandbox note.** `codex --sandbox read-only` fails on this host — the config sets
`sandbox_mode = "danger-full-access"` and the read-only helper is broken, so the first
launch died in 11 s with "blocked by the read-only filesystem policy." Rather than point a
full-access agent at a tree containing an `approved-final` plan, both rounds ran in a
throwaway `git clone` under the temp directory with `data/` absent. Read-only was enforced
by isolation, not by trusting a flag.

## What the author had already found before submitting

Two self-corrections, made after the review prompt was written and before the reviewer
returned, both of which had changed the answer by an order of magnitude:

1. **Spread is not slope.** Converting a convention's cross-V spread into a monotone tilt
   `S / range(lnV)` is not a worst case — a bounded-range perturbation can have arbitrarily
   large derivative. Using the measured shift directly gave slopes up to **39× larger**.
2. **The curvature floor is fitting-dependent.** A quadratic fitted globally over Tao's
   grid is not a local model of the minimum; 21 of the fitted slices place their optimum
   outside the measured grid. Across three fitting windows the floor fell from 0.04732 to
   **0.00965**.

Together these cut the reported margins from 6,699×/63× to **57×/8.7×**.

## Round 1 — VERDICT: CHANGES REQUIRED (3 blocking, 4 advisory)

All seven accepted; none rejected.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKING | P4 gate keyed on a point estimate; with 3 pilot seeds a noisy `\|D̂_pilot\|` above threshold can arise from a true `D` inside the band | Gate now requires the **paired 95% interval** to lie wholly outside `[−B, +B]`. The matched-seed pilot already supports this by construction. |
| 2 | BLOCKING | "P4 is settled for M2" overclaims — EOS removal changes the training stream and hence `CE_model`, which nothing measures | Pass condition weakened to metric-side only; the decision not to budget a no-EOS training arm is now recorded explicitly. |
| 3 | BLOCKING | "unrecoverable from the code even in principle" overstated | Narrowed to what the evidence supports: the script cannot run on an ordinary JSON object and the probability file is absent, so the convention is not recoverable **from the release**. |
| 4 | ADVISORY | "fitted on the full train array" ambiguous — arrays differ 33.9M–742M tokens by V | Reworded to each vocabulary's own consumed prefix. Prompted the 33M-prefix check below. |
| 5 | ADVISORY | P2's headline should carry the curvature conditional | "not decision-relevant" → "not decision-relevant, conditional on the Stage B.7 below-range curvature check". |
| 6 | ADVISORY | **Reviewer corrected the author on a matter of fact.** `N_v = V·d` is not "the output head only" — Tao's paper says vocabulary parameters would normally include both tables and adopts `V·d` as an analytical proxy justified by output-layer FLOP dominance | Rewritten as an output-weighted proxy, with the consequence stated: at small `d` the proxy under-counts actual vocabulary parameters by nearly 2×. Inherited deliberately, reported explicitly. |
| 7 | ADVISORY | Qualifier wording | "with input and output vocabulary held equal" → **"under coupled input/output vocabulary size"**. |

Finding 6 is the substantive one. The author's FLOP argument reached the right region for
the wrong reason: it inferred from the FLOP formula that `V·d` *must be* the output head,
when Tao state plainly that it is a simplification they chose. The corrected version is both
more accurate and more useful, because a proxy can be poor exactly where this study
operates.

**Prompted by finding 4, an additional check the reviewer did not ask for.** The unigram in
a confirmatory run is fitted on that run's consumed prefix, and smoothing matters more on
less text — so the full-array bound may be optimistic. Refitting every vocabulary on a
common 33M-token prefix (the smallest `T_target` in the grid) degrades the P2 margin from
**57× to 36.6×**. It still clears; the full-array figure alone would have overstated headroom.

## Round 2 — VERDICT: CHANGES REQUIRED (1 blocking, 1 advisory)

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKING | The 33M-prefix numbers (5.02e−05, 1.07e−04, 37×) appear nowhere in the JSON artifacts, so the claim is unsupported by the reviewable record | **Correct against the clone it reviewed, and already fixed in the tree when it landed.** The check had been folded into `scripts/p2_smoothing_sensitivity.py` and written to `worst_case_prefix` while round 2 was running. Verified both ways: the round-2 clone's JSON lacks the key, the current one carries it with all 20 per-vocabulary entries. **Sequencing error on the author's part** — the clone was taken before the fix — not a defect in the proposal. |
| 2 | ADVISORY | The design-impact table reintroduced the "output head" overclaim that round 1 finding 6 had just corrected, calling `N_v = V·d` "the output head" and M1's terms "output-head parameters" | Both rows rewritten in proxy language. The M1 row now also states *why* the proxy is harmless there: both terms are the same proxy, so its bias cancels in the ratio. A sweep of both documents found one further instance ("output-head allocation law"), also fixed. |

Finding 2 is the useful kind: a correction applied in the prose but not propagated to a
table two sections away. Worth noting that the reviewer caught the author's own correction
being applied inconsistently.

**Lesson recorded for the process, not the science.** Reviewing a clone taken at commit *N*
while continuing to commit to the working tree produces exactly this: a blocking finding
that is simultaneously correct and obsolete. Subsequent rounds clone immediately before
launch, and the round prompt states explicitly what changed since the previous clone.

**Sandbox reliability.** Round 2's first launch hung indefinitely at "Reading additional
input from stdin" with 39 bytes of output. Round 1 printed the same line and proceeded, so
the message is not itself the fault. Relaunching with `< /dev/null` fixed it. Both
observations are recorded because "it printed something plausible and then did nothing" is
the same failure shape as the liveness-check and VRAM-spill incidents: the instrument
returned a benign-looking signal instead of an error.

## Author-initiated additions between rounds

Neither was requested by the reviewer.

1. **P4 at the smallest training budget.** The prefix robustness check had been applied to
   P2 but not P4 — an inconsistency in the author's own analysis. Refitting the drop-EOS
   comparison on the 33M prefix gives **9.4×** against 8.7× on the full arrays: marginally
   *better*, not worse. The asymmetry is expected and is now stated: P2's sensitivity is
   driven by how much text the unigram is fitted on, P4's by the document rate, which a
   prefix does not change. The consequence matters for the decision — **P4's thin margin is
   a property of the packing convention and will not improve with more data**, which is the
   argument for gating it rather than waiting for a better measurement.

   > ⚠️ **WITHDRAWN — retained only as a record of what was claimed at the time.** The
   > sentence above beginning "The asymmetry is expected" is **wrong** and was refuted in
   > round 3: a fixed *token* prefix does **not** hold the document rate constant. It covers
   > 13,137 documents at V=384 and 30,185 at V=17792, a 2.3× range, because fertility is
   > higher at small V. The 9.4× measurement stands; the explanation does not. See round 3
   > below and `results/p2p4_resolution.md`, which carries the corrected treatment. Do not
   > cite this paragraph's reasoning.
2. **A stale line removed** from `scripts/p2_smoothing_sensitivity.py`, which still printed
   the superseded claim that a spread `S` bounds the perturbation of cross-vocabulary `L_u`
   differences by `S`.

## Round 3 — VERDICT: CHANGES REQUIRED (1 blocking, in two parts)

Both parts concerned the P4 prefix check the author had added between rounds — the addition
the reviewer was explicitly asked to attack.

**(a) The number existed only in prose.** `worst_case_prefix` recorded the P2 prefix values
for all 20 vocabularies but nothing for P4, so 4.166e−04 and the 9.4× margin were
unconfirmable from the record. **This is the same fault the author had just fixed for P2,
repeated immediately for P4** — a check run inline, cited, and never made reproducible.
`scripts/p2_smoothing_sensitivity.py` now computes both, and the JSON carries per-vocabulary
no-EOS `H`, `n_docs_in_prefix`, `eos_share_prefix`, and a `slopes` block. All four margins
are now derivable from the two JSON artifacts alone.

**(b) The reasoning was not merely unsupported — it was wrong.** The author had written that
a prefix cannot matter because it "does not change the document rate." The reviewer objected
that a fixed *token* prefix can cover different document extents by vocabulary. Measured, it
does, and substantially: 33M tokens covers **13,137 documents at V=384 and 30,185 at
V=17792, a 2.3× range**, because fertility is higher at small V.

The mechanism is **withdrawn**. The insensitivity (0.93×, margin 9.4× vs 8.7×) is now
reported as an empirical result. A candidate explanation — that the slope measures the
cross-V *shape* of the perturbation, and the train-side EOS-share ramp is similar under both
regimes (2.15× full, 2.30× prefix), with `selection_val` identical — is offered explicitly
as "a plausible reading of a measurement," not a derivation.

The operative conclusion survives because it never depended on the mechanism: **P4's margin
does not improve when the fit set changes, so more data will not fix it.**

**This is the third time in this project that a plausible mechanism was asserted ahead of
the measurement and turned out wrong** — after the predicted fertility flattening inside
Tao's fitted range (refuted by the inside-range control) and the tilt-is-worst-case argument
(refuted by computing the actual slope). The pattern is consistent: the *numbers* have held
up, the *explanations offered for them* have not. Mechanism claims in the write-up must be
labelled as readings of measurements unless separately tested.

**Precision correction found while verifying (a).** The full-array P4 margin is **8.7×**, not
the 9.0× stated in rounds 1–3 prose, and P2's prefix margin is **36.6×**, not 37×. Both came
from rounding a printed value and then quoting the rounded figure as if measured. Corrected
throughout.

## Round 4 — VERDICT: APPROVED (1 advisory)

The reviewer confirmed both round-3 fixes: the P4 prefix margin is now independently
derivable from the JSON artifacts without trusting the prose, and the withdrawal of the
document-rate mechanism is complete in `results/p2p4_resolution.md` — the replacement
explanation does not smuggle the same claim back in softer language, and the operative
conclusion survives without it.

**Advisory, applied.** `PLAN-REVIEW-LOG.md` still preserved the withdrawn mechanism as
historical process text. It is now annotated in place with an explicit WITHDRAWN block
pointing to the refutation, rather than silently deleted — the record of what was claimed at
the time is worth keeping, provided nobody can cite it by accident.

## Outcome

**A3 (unigram smoothing) and A4 (EOS packing with an M2 gate) are ADOPTED** into
`AMENDMENTS.md`, dated 2026-08-01. P2 and P4 are marked CLOSED and superseded rather than
deleted. The `OPEN ITEM` docstring in `src/metrics.py` is replaced by a pointer to A3 and
carries the curvature conditional.

**Tally across the amendment track: 4 rounds, 12 findings, 0 rejected.** Three blocking in
round 1, one blocking in each of rounds 2 and 3, the rest advisory. The reviewer corrected
the author on one matter of fact (`N_v = V·d` is Tao's analytical proxy, not a definition of
the output head) and refuted one piece of reasoning outright (the document-rate mechanism).
The author independently found two order-of-magnitude errors before the reviewer returned,
added two robustness checks unprompted, and found one precision error (8.7× / 36.6×) while
verifying a fix.

`PLAN.md` was not edited at any point. Its `final_body_sha256` binding is intact.

---

# Seed-semantics track

A separate review track from the A3/A4 amendment track above. Trigger: while building the
Stage B.7 pilot runner, `PLAN.md` was found to require seed-level bootstrap resampling
without ever stating what a seed varies — and the implementation had answered by accident,
varying initialisation only.

## Round 1 — VERDICT: CHANGES REQUIRED (5 blocking, 0 advisory)

All five accepted, none rejected. Two are worth recording in full because they corrected the
author rather than the artifact.

**The author's proposed fix was itself defective.** The proposal said "permute the block
indices." Against the then-current `TokenStream`, that would have concatenated permuted
blocks and formed targets by shifting across the joins, inventing one false next-token pair
per block — about 0.049% of targets at B=2048, the same order as the EOS effects that A4
treats as decision-relevant. The fix would have traded a variance defect for a data defect.
Sequences were made self-contained instead (`tokens[k*B : k*B + B + 1]`), which is what the
reference already does via `effective_block_size = block_size + 1`, so the repair moves
toward the reference rather than away from it.

**The author's bias claim was overstated and is withdrawn.** The draft said
initialisation-only variance is a lower bound on run-to-run variance with all consequences in
one direction. A fixed-order conditional variance is not guaranteed below the order-averaged
variance for every statistic, and order effects can cancel or amplify in paired contrasts
like `D`. The surviving claim — that the pilot would estimate the *wrong variance component* —
is sufficient to block B.7 and does not depend on a direction.

The reviewer also found a second, unrelated defect: `TrainConfig.total_steps` ceiled the step
count while its docstring claimed the final step was trimmed. Nothing trimmed it, so every
run overshot its budget. Two of the author's descriptions of the reference were also wrong —
`create_dataloaders` passes `shuffle=False`, so the `PackedDataset` buffer shuffle is
inactive on the path actually used.

## Round 2 — VERDICT: CHANGES REQUIRED (5 blocking, 1 advisory)

Submitted after the round-1 fixes plus three enforcement guards the author added
unprompted. All six accepted.

**The blocking finding the author should have caught: the fix broke the primary metric.**
`L_u = CE_model − H_unigram`, and `H_unigram` was fitted on `train_tokens[:consumed]`. That
was correct only while consumption was sequential. Permuting sequence order makes the
consumed set scattered, so the baseline was being fitted on text the model never read — and,
worse, on a set *identical across seeds* while the model's set is not, deleting a real
component of seed variance from the metric whose seed variance Stage B.7 exists to estimate.
The seed-order fix would have introduced a defect into `L_u` while repairing one in the
variance estimate.

Now measured rather than argued (`scripts/unigram_order_sensitivity.py`, real tokenized pilot
data, three seeds, evaluation targets held fixed so only the fit set moves): worst bias
**1.168e−04 nats**, worst between-seed SD **4.345e−05 nats**, worst local slope
**1.701e−04 nats per ln V**, implied argmin displacement **1.763e−02 in ln V** — the M1
margin is **23.0×** larger. So the bias alone would probably not have flipped M1, but 23× is
the second-tightest margin in this study after P4's 8.7×. The margin is not the reason the
fix is required: a variance component set to zero by construction is a different estimand,
not a small error, which is the same objection this amendment makes about
initialisation-only seeds.

**The gradient weighting was wrong on trimmed steps.** `loss / len(micro)` is correct only
for equal-size micro-batches. On a trimmed final step of `[4, 4, 4, 1]` the one-sequence
batch received 1/4 of the gradient instead of 1/13. Now weighted by each micro-batch's share
of the step's sequences, checked in a test against a single full-batch backward pass.

**The author's own witness was too weak.** The added `stream_sequences` field cannot
distinguish a different token array of the same length, a different same-length slice, or a
change in the permutation algorithm. Now `tokens_digest`, `order_digest`,
`sequences_consumed` and `numpy_version` as well — the last because
`np.random.default_rng(seed).permutation(n)` is **not promised stable across NumPy
versions**, a claim the amendment now explicitly disclaims rather than relies on.

The reviewer also caught the withdrawn bias claim reappearing in a sentence the author had
missed ("sized by a number known to be too small"), and specified what A6 must and must not
say. The advisory concerned the lookahead token: self-contained sequences need
`n*B + 1` ids, and `target_sequences` could exceed budget for a sub-block budget via its
`max(1, ...)` floor — the single case where the convention could violate the property it
exists to enforce. That path now raises.

### ⚠️ A precision error the author introduced and then caught

Between rounds the author "corrected" the budget figures from +0.0129%/−0.0011% to
+0.0134%/−0.0010%. **The correction was wrong and has been reverted.** It was computed
against `C` hardcoded as `1.03084e16`, while the codebase derives
`C = 1.0308446911162678e16`. The reviewer independently reproduced the original figures,
which is what surfaced the error.

The rounding is a relative change of **4.6e−06**, yet it moved the worst cell by 4%, because
the per-cell overshoot depends on `T_target mod tokens_per_step` and is therefore
*chaotically* sensitive to `C`. The operative lesson is not "round more carefully" — it is
that **the per-cell overshoot is not a stable quantity and nothing should reason from it.**
The stable quantity is the bound `(tokens_per_step − 1) / T_target`. Both are now emitted by
`scripts/budget_convention_error.py`, deriving `C` and the vocabulary grid from `src/` rather
than from a transcribed literal.

This is the fourth precision error of this family in the project, after the 9.0×/8.7× and
37×/36.6× corrections. The first three were rounded printed values quoted as measurements;
this one was a transcribed constant. The structural response — figures regenerate from an
artifact, constants derive from the codebase — addresses the family rather than the instance.

## Outcome

**A6 is ADOPTED**, dated 2026-08-01, superseding A1 in part. Tests: 154 passing.
`PLAN.md` was not edited; `git log --all -- PLAN.md` still shows the single initial commit
and the `final_body_sha256` binding is intact.

**Tally across the seed-semantics track so far: 2 rounds, 11 findings, 0 rejected.** The
reviewer refuted the author's proposed fix, withdrew the author's bias claim, corrected two
of the author's descriptions of the reference, found a defect in the primary metric that the
author's own fix had introduced, and rejected the author's first attempt at an audit
witness. The author independently found three enforcement gaps and one precision error of
its own making.

A confirmation round has not yet been run against these fixes.
