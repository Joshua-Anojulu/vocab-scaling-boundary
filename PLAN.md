---
review_provenance:
  schema_version: 2
  status: approved-final
  max_rounds_note: "MAX_ROUNDS raised 5 -> 6 by the human after the round-5 cap deadlock, to apply four
                    agreed fixes. Round 7 was a further extension taken by Claude on the judgement that
                    both remaining round-6 items were mechanical and the reviewer had scoped everything
                    else as implementation-ready. Both extensions are recorded, neither was silent."
  rounds:
    - {schema_version: 2, round: 1, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 579bfb637eba9ed905ded2c2a06a237c092e9fca3fe4b770b04b25306cebb798,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "Plan inlined; -s read-only blocks Codex shell reads. Repo held only PLAN.md + log."}
    - {schema_version: 2, round: 2, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 678db529e1c08d11cedc602418d11fc78e7b37d1d2e410d0d70f95533b0114eb,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d, note: "Resumed; thread id echoed identical."}
    - {schema_version: 2, round: 3, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 367750b89192f91dc1843a495de425800554791b1f93991ff9f3938c04df26ad,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "Framing declared sound; remaining items were instantiation gaps."}
    - {schema_version: 2, round: 4, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 954961f618da97667c860e3d7b2b7fc28b2cef39c6cf1ef5292bfa7eff3bfa25,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "Arithmetic audit; confirmed 4 author-found errors and added the fitted-vs-realized fertility catch."}
    - {schema_version: 2, round: 5, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 48199c8d253103b2f8a15c395ecd62a8eaaf0bad0bd9896646fa721f4ab865dc,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "MAX_ROUNDS reached. Numerical audit passes; 4 inference-procedure blockers remain, all
              agreed by the author. Cap deadlock, not an impasse. See PLAN-REVIEW-LOG.md Round 5."}
    - {schema_version: 2, round: 6, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: REVISE,
       body_sha256: 53612a71fe412ef03de0a2fddd4404a10fa00dfe0d5446cecf427c307c5b43ea,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "Null definitions and Holm-over-12 confirmed correct. 2 blockers: D_pilot uncomputable from
              the budgeted pilot, and failure power ignored Holm adjustment."}
    - {schema_version: 2, round: 7, reviewer: codex, model: gpt-5.6-sol, cli_version: codex-cli/0.145.0,
       grounding: repo, qualifying: true, verdict: APPROVED,
       body_sha256: 970a523fff1452ccf4ee824ed51786b34a3bd72ee70af5845e63d18eb7defd6c,
       session_id: 019fb0f1-9b63-7fe2-b9fe-77fbdf23065d,
       note: "APPROVED. Budget independently reproduced to 8 s.f. (1.8685805e18, 139 runs). One
              nonblocking wording item left unapplied on purpose to preserve this body hash."}
  historical_cross_model_review: true
  final_body_cross_model_approved: true
  final_body_sha256: 970a523fff1452ccf4ee824ed51786b34a3bd72ee70af5845e63d18eb7defd6c
  degraded_rounds: []
  known_nonblocking_followup:
    - "Interval-construction paragraph says common random numbers across 'the five vocabulary points';
       should also name the matched sixth V_run@1.1C observation in the CONFIRMATORY seed block, as
       Stage B already does for the pilot. Deliberately NOT applied post-approval so the approved body
       hash stays valid; apply at Preregistration I."
    - "Stage-B simulation should report BCa COVERAGE alongside power; if coverage is inadequate at the
       computed n, raise n or abort — never change the estimator after confirmatory data are observed."
  addressed_after_cap:
    - "Separate equivalence/failure nulls + BCa bootstrap spec + Holm across 12 directional tests"
    - "Four power targets (M1-equiv, M1-fail, M2-adeq, M2-fail); n = max, applied to sweep and 1.1C"
    - "V_run = snap128(V_pred) defined; BPE trained to V-3 with hard assertion that sizes equal V"
    - "Final-test routing: selection-val used ONLY for boundary extension; estimator runs once on test"
    - "Non-blocking: signed statistic D replaces gain10; 'nesting not imposed' replaces 'non-nested'"
---

# Plan: Does Tao et al.'s vocabulary law survive extrapolation below 33M?
_Locked via grill — by Claude + josha. Revised after Codex rounds 1–4._

## The hypothesis, named exactly

> **H₀:** Approach 2's `N_v(N_nv)` predictor, evaluated at the Approach-1-implied compute-optimal
> `C(N_nv)`, under the shared Tao training recipe and the preregistered sub-50M architecture family,
> predicts the empirical compute-optimal vocabulary within margins M1 and M2, at every tested scale
> below 33M.

A **composite** of A2 and A1: A2 gives `N_v(N_nv)` but not the budget, so A1's `N_nv(C)` is inverted.

```
A2:  N_v = 3,145,728 · (N_nv / 33e6) ^ 0.8353974035228025      # 3,145,728 is the predictor's
A1:  C   = (N_nv / exp(-2.4846510161625193))²                  # reference anchor AT N_nv = 33e6,
                                                                # not its value at realized 33,222,784
```
Commit of `sail-sg/scaling-with-vocab` pinned at Preregistration I. The rounded "0.84" is never used.
A1's own `N_v(C)` and A3's parametric fit are preregistered sensitivity analyses.

## Why it matters

Extrapolated below its fitted range the law recommends vocabularies of **~1,715–4,459 tokens** at 2–16M
`N_nv` — **7–19× below standard general-purpose subword vocabularies**, which small-model practitioners
routinely inherit at 32k. Nothing validates that extrapolation: Tao et al. stop at 33.2M, *Compute Optimal
Tokenization* (arXiv:2605.01188) at 50M, and both sets of limitations point upward. Godey et al.
(arXiv:2404.07647) supply a concrete failure mechanism in range: small hidden dimensions induce a **rank
bottleneck in the vocabulary projection** at 14–31M.

**Withdrawn claims** (both were mine, both false): that embeddings dominate parameters below 33M — at the
predicted optimum `N_v/N_nv` is ~9.5% at 33M and ~15% at 2M; and "smaller than any vocabulary in practical
use" — byte-, character- and tiny-model vocabularies exist.

**Independent argument from their data:** `exp_data.csv` shows Tao et al. ran **one run per cell, no seed
replication**. Their own bowl minima carry unquantified noise, weakening any extrapolation derived from
them. This study replicates and is in that respect more rigorous than the work it tests.

## Accounting, resolved from source

`exp_data.csv`: `Non_vocab_parameters` is identical at **33,222,784 across all ten vocabulary sizes** at
`d=512`. `lit_gpt/model.py`: `lm_head` is a separate `nn.Linear` from `wte`, no weight assignment.

```
N_nv           = transformer body only — EXCLUDES both embedding tables
N_v            = V · d   — output head only (Tao's definition)
total (untied) = N_nv + 2·V·d
```
`V_tok = V_head = V` by construction: every nominal vocabulary size is divisible by the padding quantum
(128), so no padding rows are unused and passing `V` to the fertility polynomial is correct.
`L_u` is **negative** when the model beats the unigram baseline (their rows: −1.35 / −2.10 / −2.50); more
negative is better, and the sign convention is asserted in tests.

## The ruler, and how IsoFLOP is actually enforced

```python
f(V) = 0.00639222*ln(V)**2 - 0.15811069*ln(V) + 1.20470122      # V capped at 200_000
C    = 6*(N_nv + V*d)*H*f(V)
```
Verified exact against their published row (`V=4096, d=512, H=180,820,540.85` → `1.2715176959854e16`;
published `1.271517696e16`).

**Enforcement is on tokens, not characters.** `f(V)` is Tao's *fitted* fertility, which will not equal a
newly trained tokenizer's *realized* fertility; consuming `H` characters would therefore miss target `C`.
So:

```
T_target = C / (6·(N_nv + V·d))          # exact token budget — this is what is consumed
H_Tao    = T_target / f(V)               # Tao-equivalent character estimate, reported
r_actual = T_target / H_actual           # realized fertility; H_actual and bytes_actual also reported
                                         # r_actual vs f(V) divergence is a reported result, not noise
```
This makes `C` exact by construction and surfaces fertility mismatch as data rather than hiding it.

**Below-range extension.** Tao's `Nnv_to_d` hardcodes `d=512` for all `N_nv ≤ 50M`, infeasible here (one
`d=512` block costs `12·512² = 3.15M` > a 2M budget). The primary ruler is the faithful algebraic extension
using `d_actual`; their unmodified helper and their exact architecture are used **only at the anchor**.
Operator-level FLOPs are a **re-expression** of completed runs, never an exact-IsoFLOP counterfactual.

## Architecture family and grid

```
d         = snap64( 512 · (N_nv_target / 33.222784e6)^(1/3) )   # constant aspect ratio
d_ffn     = snap64( 8d/3 );  head_dim = 64;  n_head = d/64
per_layer = 4d² + 3·d·d_ffn + 2d
n_layer   = round(N_nv_target / per_layer);   N_nv = n_layer·per_layer + d
```

| target | `d` | `d_ffn` | `n_head` | `n_layer` | realized `N_nv` | A2 `N_v` | `V_pred` | A1 `C` |
|---|---|---|---|---|---|---|---|---|
| 2M | 192 | 512 | 3 | 5 | 2,213,952 | 329,239 | 1,714.79 | 7.0547×10¹⁴ |
| 4M | 256 | 704 | 4 | 5 | 4,016,896 | 541,561 | 2,115.47 | 2.3223×10¹⁵ |
| 8M | 320 | 832 | 5 | 7 | 8,463,040 | 1,009,280 | 3,154.00 | 1.0308×10¹⁶ |
| 16M | 384 | 1024 | 6 | 9 | **15,932,544** | 1,712,160 | 4,458.75 | 3.6535×10¹⁶ |

**The rule is NOT applied to the anchor.** At `d=512` it yields `d_ffn=1344, n_layer=11 → 34,254,336`,
**3.105% high** — it fails its own 2% check. Rather than bend the preregistered family to fit, **the anchor
uses Tao's exact published architecture** and the sub-50M rule stands separately. (Their `n_layer` at
`d=512` is not derivable from the released CSV and is read from `lit_gpt/config.py` in Stage B.)

**Grid generation algorithm** (executable, deterministic):
```
V_MIN   = snap128_up(256 byte symbols + 3 special tokens) = 384
snap(x) = max(V_MIN, 128·round(x/128))
grid    = [ snap(V_pred·m) for m in (1/4, 1/2, 1, 2, 4) ]
if len(set(grid)) < 5:  regenerate 5 log-spaced distinct points on [max(V_MIN, V_pred/4), V_pred·4]
                        retaining the point nearest V_pred
```

| scale | grid | distinct |
|---|---|---|
| 2M | 384, 896, 1664, 3456, 6912 | ✓ |
| 4M | 512, 1024, 2176, 4224, 8448 | ✓ |
| 8M | 768, 1536, 3200, 6272, 12672 | ✓ |
| 16M | 1152, 2176, 4480, 8960, 17792 | ✓ |

384 is a genuine tokenizer floor (byte alphabet + specials), not a scientific choice.

**`V_run` — the implemented vocabulary.** `V_pred` is continuous; a model needs an integer. Define
`V_run = snap128(V_pred)`, which is by construction the central grid point at each scale:

| scale | `V_pred` | **`V_run`** |
|---|---|---|
| 2M | 1,714.79 | **1,664** |
| 4M | 2,115.47 | **2,176** |
| 8M | 3,154.00 | **3,200** |
| 16M | 4,458.75 | **4,480** |

`V_run` is what the M2 runs at `1.1·C` use. M1 compares the fitted continuous optimum `N_v*` against the
continuous `N_v_pred`, so no snapping enters M1.

## Decision procedure (a statistical test, not a rubric)

**α = 0.05 throughout.** Let `θ = ln(N_v* / N_v_pred)` at a given scale, and let
`D = L_u(V_run, 1.1·C) − L_u(V*, C)` be the signed M2 statistic. `D` is used **directly and untruncated** —
an earlier draft built M2 from `gain10`, which stochastic training does not guarantee non-negative.

**Equivalence and failure are separate nulls.** Failing an equivalence test is *inconclusive*, never
falsification — this was the central defect of the previous draft, which could not reject anything.

| | Null | Rejected when | Meaning |
|---|---|---|---|
| **M1-equiv** | `\|θ\| ≥ ln 1.5` | TOST: both one-sided tests reject at α | prediction is practically right |
| **M1-fail-lo** | `θ ≥ −ln 1.5` | upper 95% bound `< −ln 1.5` | empirical optimum is materially *below* prediction |
| **M1-fail-hi** | `θ ≤ +ln 1.5` | lower 95% bound `> +ln 1.5` | empirical optimum is materially *above* prediction |
| **M2-adeq** | `D ≥ 0` | upper 95% bound `< 0` | predicted vocabulary costs under a 10% compute increase |
| **M2-fail** | `D ≤ 0` | lower 95% bound `> 0` | it costs more |

`ln 1.5 = 0.4055` is the coarsest adjacent spacing in the *low-vocabulary* region of Tao's grid
(4096→6144 = 1.50, 6144→8192 = 1.33, 8192→10240 = 1.25; the grid coarsens again above, 10240→16384 = 1.60).
Generous, and justified only because M2 is independently measured.

**Interval construction, fully specified.** BCa bootstrap, resampling **at the seed level within scale**
(seeds are the exchangeable unit; vocabulary points are fixed design), 10,000 resamples, common random
numbers across the five vocabulary points so the bowl refits coherently. TOST uses 90% two-sided intervals
(equivalent to two one-sided 5% tests); failure tests use 95% two-sided intervals. Bootstrap p-values are
the standard inversion of these intervals.

**Multiplicity.** Rejection claims are Holm-adjusted across **12 directional tests**
(4 scales × {M1-fail-lo, M1-fail-hi, M2-fail}). The 12-test route is taken rather than constructing a
composite two-sided M1 p-value, because the composite would need its own calibration argument. The
equivalence claim is intersection-union across scales and needs no correction (conservative by construction).

**Precedence:** **valid rejection > all-scale equivalence > inconclusive.** Any Holm-surviving rejection at
any scale falsifies the all-scales composite, regardless of another scale being censored or non-convex.
Failure to reject is never reported as confirmation.

**Power, four targets — and each at its own operative α.** Powering a *failure* test at unadjusted α would
overstate power, because a rejection must survive Holm over 12 directional tests; with a single genuinely
failing hypothesis the first Holm threshold is `0.05/12`. Equivalence keeps α = 0.05, because the
intersection-union claim needs no multiplicity correction. All targets at 80% power:

| target | alternative | operative α |
|---|---|---|
| M1 equivalence | `θ = 0` | 0.05 |
| M1 failure | `\|θ\| = ln 2` | **0.05 / 12** |
| M2 adequacy | `D = −½·\|D̂_pilot\|` | 0.05 |
| M2 failure | `D = +½·\|D̂_pilot\|` | **0.05 / 12** |

Failure power is simulated **through the complete Holm procedure**; `0.05/12` is the conservative fallback
if that simulation is not implemented. `n = max` over all four, applied to **both** the confirmatory sweep
and the 1.1·C runs. No seed count is hardcoded anywhere.

**Seed floor — a binding safeguard.** BCa inference from only three seed clusters has highly discrete and
unstable coverage, so a small computed `n` is not accepted at face value: **`n ≥ 5` is a hard floor**
regardless of what the power simulation returns. If the simulation returns `n < 5`, the floor binds and the
extra seeds are spent; the alternative — trusting a BCa interval built on three clusters — would make every
downstream inference unreliable in exactly the regime the study depends on.

**Bowl estimator, frozen.** Weighted quadratic in `ln V`, weights from seed-level SEM, fit per scale.
Adequacy gates, both numeric:
- **Convexity:** bootstrap CI on the leading coefficient must exclude 0 from below.
- **Lack-of-fit F-test** against pooled within-seed pure error, α = 0.01. Fail → **inconclusive at that
  scale**; no re-modelling. ("Residual inspection" is descriptive only.)

`ln V*` uncertainty by nonparametric bootstrap over seeds, 10,000 resamples. **Boundary rule:** if `V*`
falls outside the grid, report boundary-censored and extend by **one** preregistered point in the indicated
direction — one extension only, triggered by the fitted minimum's location, never by inspecting test loss.

**Evaluation split — one estimand, no ambiguity.** The previous draft said adaptation *and* the estimator
both ran on different splits, which implied two different estimands. Corrected:

- **Selection-validation is used for exactly one thing: the boundary-extension decision** (whether the
  fitted minimum falls outside the grid, and in which direction). Nothing else.
- **The frozen bowl estimator, its adequacy gates, and the entire decision procedure then run once on
  final-test losses.** That is the reported estimand. There is **no second final-test fit** and no
  re-selection — a selection-validation boundary fit necessarily precedes it, and that one is not the
  reported estimand.
- Final-test is touched exactly once, after every adaptive choice is already locked.

## Corpus and tokenizer, instantiated

| | |
|---|---|
| Corpus | SlimPajama-627B, `train` split, chunk1; shards selected by ascending index, RNG seed 1234 |
| Document filter | keep documents ≥128 UTF-8 bytes and UTF-8-decodable; no other filtering |
| Split | `bucket = int(sha256(document_text.encode("utf-8")).hexdigest()[:8], 16) % 1000` → 0–19 tokenizer-fit, 20–979 train, 980–989 selection-validation, 990–999 final-test. Deterministic, reproducible, disjoint. |
| Tokenizer sample | 2 GB sampled from the tokenizer-fit split only |
| Implementation | HuggingFace `tokenizers`, byte-level BPE, version pinned at Preregistration I |
| Pre-tokenization | GPT-2 regex: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` |
| Normalization | **none** — byte-level BPE; any Unicode normalization would break the round-trip losslessness this plan asserts |
| Special tokens | 3 (BOS, EOS, PAD), added after BPE training — **the lexical BPE is trained to `V − 3`**, so the final tokenizer and the unpadded head are both exactly `V` |
| Assertion | `len(tokenizer) == V` and `lm_head.out_features == V` are hard-asserted before every run |
| Tie-breaking | equal-frequency merges broken by lexicographic order of the pair |
| Vocabularies | trained independently per size; **nesting is not imposed** — but deterministic BPE on the same corpus with identical tie-breaking will generally produce nested merge prefixes anyway, so no non-nesting claim is made |

All tokenizer artifacts frozen and hashed before any model training.

## Budget

With `ΣC = 4.98713×10¹⁶` over the four scales, `C_8M = 1.03084×10¹⁶`, and `n` the Stage-B-computed seed
count (floor 5):

```
pilot = 6.1 · 3 · C_8M = 1.88644e17          # 5 vocab at C + V_run at 1.1·C, 3 seeds, 18 runs
total = 6.1·n·ΣC + pilot + anchor            # 6.1 = 5 sweep vocabularies + 1.1 for the M2 run
        + n·ΣC per boundary extension applied at every scale
```

| component | runs | FLOPs at `n=5` |
|---|---|---|
| Confirmatory sweep (4 scales × 5 vocab × `n`) | 20`n` = 100 | 1.24678×10¹⁸ |
| M2 runs (`V_run` @ 1.1·C, `n` seeds) | 4`n` = 20 | 2.74292×10¹⁷ |
| Power pilot (8M; 5 vocab @ `C` **+ `V_run=3200` @ `1.1·C`**, 3 seeds; excluded from confirmatory) | 18 | 1.88644×10¹⁷ |
| Anchor — **one** run at `V=6144`, Tao's exact architecture, matching their single-run protocol | 1 | 1.58859×10¹⁷ |
| **Total at the `n=5` floor** | **139** | **1.86858×10¹⁸** |
| Worst case: boundary extension at all four scales | +4`n` = +20 | +2.4936×10¹⁷ → 2.11793×10¹⁸ |

Ideal hours at `n=5`: **51.9 @ 10 TFLOP/s · 34.6 @ 15 · 20.8 @ 25**; worst case 39.2 h @ 15.
For reference, `n=3` would be 91 runs and `1.26015×10¹⁸` (23.3 h @ 15) — but the `n ≥ 5` floor makes that
unreachable, so **`n=5` is the planning minimum, not `n=3`**.

These are *achieved-throughput* figures; Stage B measures the rate rather than assuming it, and the abort
threshold is evaluated at the computed `n` including the worst-case extension. (An earlier draft stated
1.5×10¹⁷ for the sweep, omitting ×5 vocabularies and ×`n` seeds — a 5× error.)

## Approach

**Stage A — Preregistration I.** Register: pinned commit and predictors; the ruler and its token-based
enforcement; the architecture rule and table; grid algorithm; corpus/tokenizer table above; metric
definitions; **M1 and M2**; α, target power, bootstrap and lack-of-fit procedures; the decision precedence;
the shrink/abort rule. All Stage B pilot data is excluded from confirmatory inference.

**Stage B — Instrument and budget (hard gate).**
1. Python **3.12/3.13** venv (not the installed 3.14.5 — PyTorch wheels lag), CUDA 12.x verified on the
   RTX 4060 Laptop (8 GB, compute 8.9). Root outside OneDrive.
2. Read `lit_gpt/config.py` to fix the anchor's exact architecture.
3. **Zero-GPU re-analysis** of `exp_data.csv`: refit A2, estimate bowl curvature at 33M. Validates the
   analysis pipeline at no compute cost.
4. **Reference cross-validation** at the pinned commit: parameter counts, logits, loss, gradients, one
   optimizer update, FLOP accounting, short loss trajectory, within stated tolerances.
5. **Sustained-throughput benchmark** ≥20 min on the 16M architecture (captures thermal throttling).
6. **VRAM pilot** with exact optimizer, context and **fused/chunked cross-entropy** (the reference ships
   `fused_cross_entropy.py` for this reason).
7. **Power pilot: 8M, 3 seeds, at SIX configurations** — the five vocabulary points at `C`, **plus
   `V_run = 3200` at `1.1·C`**. Five points are the minimum that estimates quadratic curvature; two cannot.
   The sixth configuration is not optional: `D̂_pilot = L_u(V_run, 1.1C) − L_u(V*, C)` is *uncomputable*
   without a `1.1·C` observation, so the previous draft's M2 power targets were circular. The `1.1·C` runs
   are matched-seed and enter each seed-level bootstrap block as a sixth paired configuration. Curvature at
   33M remains a conservative cross-check only, never a plug-in for below-range scales.
8. **Power simulation over all four targets** (M1-equivalence at `θ=0`, M1-failure at `|θ|=ln 2`,
   M2-adequacy and M2-failure at `∓½·|D̂_pilot|`), 80% power, α=0.05. `n = max` over the four is the
   confirmatory minimum for **both** the sweep and the 1.1·C runs. The pilot also yields `D̂_pilot`, the
   observed M2 effect at 8M, which parameterises the two M2 targets. **Abort** if the total at that `n`,
   including a worst-case all-scale boundary extension, is unaffordable.

**Stage C — Preregistration II.** Freeze the generated configuration table, one row per architecture ×
vocabulary × seed, each carrying nominal = padded `V`, `d`, target `C`, `T_target`, `H_Tao`, expected steps,
and both FLOP counts. Generated by a committed script from the formulas above.

**Stage D — Runs.** Grids as tabled. **Tao's schedule, separate budget-specific runs** (WSD removed
entirely — stable-phase checkpoints are not finished lower-budget models). Log sustained throughput per run.

**Stage E — Inference.** Primary **`L_u`** (their metric, their released column `Lossu`), unigram
probabilities from the **training split only**, matching reference smoothing / special-token / boundary /
zero-frequency conventions, frozen before selection. **BPB co-reported secondary** from token NLL over exact
UTF-8 byte counts, round-trip losslessness asserted, zero unknown-token rate — with a separately worded
claim, since BPB rejecting while `L_u` agrees falsifies only the transfer of the recommendation.
Mechanism ablation varies **`d`** (output-projection rank), not attention-head count; LM-head factorization
removed. An anchor discrepancy is reported as *consistent with* Tao's omission of attention, never causally
attributed without a matched counterfactual.

**Stage F — Write-up.** Related work: Tao et al.; *Compute Optimal Tokenization*; **Over-Tokenized
Transformer §3.1 (arXiv:2501.16975)**, whose ~2.4M/85M synthetic CFG experiments find larger output
vocabularies harm the smaller model; Godey et al.; GPT-wee; Takase et al. (2025). Claim: *the first
controlled English BPE test of Tao et al.'s published vocabulary-parameter prediction below 33M `N_nv`, at
their own compute-optimal budgets and under their own metric.* All citations verified from primary sources.
Workshop or IEEE URTC-class venue; no hard deadline; decision checkpoint after Stage B.

### Shrink rule

1. Drop all training-based sensitivity and mechanism arms (`d`-ablation, matched-width alternatives).
2. Drop the anchor.
3. Reduce architectures 4 → 3 (`{2M, 8M, 16M}`) → 2 (`{2M, 16M}`).
4. **Never** below five local vocabulary points, or below the Stage-B-computed seed count.
5. **Abort:** if 2 architectures × 5 vocabularies × required seeds cannot be funded, the paper claim is
   abandoned and the work is reported as an engineering/learning artifact only.

## Risks / open questions

- **Anchor architecture** must be read from `config.py`; the sub-50M rule provably does not reproduce it.
- **Realized vs fitted fertility** may diverge enough that `H_Tao` and `H_actual` tell different stories.
  This is reported as a finding, not smoothed over.
- **Mechanism only partially isolable.** BPE vocabulary simultaneously changes compression, output
  dimension, embedding capacity, raw context per window, tokens per update, and optimization dynamics.
- **The architecture-rule extrapolation is an irreducible confound**, scoped rather than removed.
- **Null result risk** — the prediction may hold. Publishable as a validated boundary, weaker.
- **Laptop thermals** — GPU exclusive; the author's separate simulation project cannot run concurrently.
- **Field velocity** — ~27 papers cited the anchor in 2026 alone. Monthly arXiv check with a preregistered
  reframe-as-replication decision if scooped.

## Out of scope

- Any claim above ~35M `N_nv`, or outside the preregistered architecture family.
- Estimating a new scaling-law exponent as the primary contribution.
- Novel tokenizer algorithms, byte-level/tokenizer-free architectures, learned tokenization.
- Multilingual or multimodal settings; fixed-byte context evaluation.
- Building a useful tool; the debugging-assistant interest is a separate engineering project.
- Beating any published model on any benchmark.
