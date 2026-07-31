# Stage B.5 / B.6 — throughput and VRAM pilot: results

Hardware: RTX 4060 Laptop, 8.59 GB, compute 8.9, bf16 native. PyTorch 2.11.0+cu128,
Python 3.12.13. All figures in the budget's own FLOP convention,
`C_per_token = 6·(N_nv + V·d)`, so they convert to wall-clock directly.

## Measured throughput

| scale | V | batch | tok/s | TFLOP/s | peak GB | verdict |
|---|---|---|---|---|---|---|
| 16M | 4480 | 4 | 30,192 | 3.20 | 2.08 | under-utilised |
| 16M | 4480 | **8** | 48,183 | **5.10** | 3.69 | **operating point** |
| 16M | 4480 | 16 | 47,716 | 5.05 | 6.92 | plateau, 2× memory for nothing |
| 16M | 4480 | 32 | 2,388 | 0.25 | **13.45** | **SPILLED** |
| 16M | 17792 | 4 | 35,290 | 4.82 | 2.75 | |
| 8M | 3200 | **8** | 76,708 | **4.37** | 2.38 | **operating point** |
| 8M | 3200 | 16 | 72,914 | 4.15 | 4.47 | plateau |
| 8M | 3200 | 32 | 10,386 | 0.59 | **8.73** | **SPILLED** |
| 2M | 1664 | 8 | 100,624 | 1.53 | 1.04 | under-utilised |
| 2M | 1664 | **32** | 196,336 | **2.98** | 3.85 | **operating point** |
| 2M | 1664 | 64 | 193,255 | 2.94 | 7.64 | plateau |

Optimal batch is **scale-dependent**: 2M is launch-overhead-bound and nearly doubles from
b=8 to b=32, while 8M and 16M are already saturated at b=8. Run-to-run variance on a
repeat pass was **5–6%** (16M 5.10→5.41, 8M 4.37→4.65, 2M 2.98→2.92) — worth remembering
when the confirmatory runs are compared.

## ⚠ Windows does not OOM when you exceed VRAM — it silently spills

The two `SPILLED` rows above reported `ok=True` and **raised nothing**. On Windows/WDDM
the driver backs the over-allocation with host memory over PCIe and training continues at
roughly a twentieth of the rate: 16M b=32 reported a peak of **13.45 GB on an 8.59 GB
card** and fell from 5.05 to 0.25 TFLOP/s.

**A max-batch search that asks "did it OOM?" selects that configuration.** The whole
sweep would then have run ~20× slower with no error anywhere. `find_max_batch` now
rejects on two additional rules, covered by `tests/test_bench_guards.py`:

* **spill** — peak allocation above 95% of total VRAM
* **collapse** — throughput below 70% of the best seen

Two further platform findings, both of which cost measurement time:

* **`torch.AcceleratorError` is not recoverable in-process.** It is a driver-level
  `cudaErrorMemoryAllocation`, distinct from the allocator-level `torch.OutOfMemoryError`.
  After it fires, even `empty_cache()` in the handler raises again. Every probe therefore
  runs in its own subprocess — required, not defensive.
* **`torch.compile` is unavailable on Windows** (`TritonMissing`; inductor needs Triton,
  which has no Windows build). Measured A/B: all three scales failed to compile. This
  removes the single largest remaining speed lever.

## Budget against measured rates

| component | rate (TFLOP/s) | FLOPs | hours |
|---|---|---|---|
| 2M sweep + M2 | 2.92 | 2.1517e16 | 2.0 |
| 4M sweep + M2 | 3.50 *(estimated — 4M was never measured)* | 7.0831e16 | 5.6 |
| 8M sweep + M2 | 4.37 | 3.1441e17 | 20.0 |
| 16M sweep + M2 | 5.10 | 1.1143e18 | **60.7** |
| power pilot | 4.37 | 1.8864e17 | 12.0 |
| anchor | 5.10 | 1.5886e17 | 8.7 |
| **total at n=5** | | **1.8686e18** | **109.0** |

**109 GPU-hours ≈ 4.5 days of continuous compute — 3.2× the planned 34.6 h.** The plan's
10/15/25 TFLOP/s bracket was optimistic by roughly 3×; nothing on this machine reaches
even the bottom of it.

**16M alone is 55% of the total.** That matters, because the preregistered shrink rule
sheds architectures from the *middle*:

| shrink step | total | saves |
|---|---|---|
| drop 4M → {2M, 8M, 16M} | 103.4 h | 5.6 h |
| drop 4M, 8M → {2M, 16M} | 83.4 h | 25.6 h |

So **the preregistered shrink rule does not solve this.** Dropping 16M instead would save
~61 h in one step — but 16M is the scale nearest the anchor and therefore the most
informative about whether the extrapolation degrades smoothly. Re-ordering the shrink rule
now, after seeing the cost, is exactly the post-hoc researcher degree of freedom the
preregistration exists to prevent. It must be a deliberate, documented amendment made
before any confirmatory run — not a quiet convenience.

## Status against the plan's Stage B gate

| step | status |
|---|---|
| B.1 environment | ✅ Python 3.12.13, torch 2.11.0+cu128, CUDA, bf16 |
| B.2 anchor architecture | ✅ RESOLVED — `tiny_LLaMA_50M`: L=8, d=512, d_ffn=2048, `padding_multiple=1`. See `results/reference_architecture_findings.md` |
| B.3 zero-GPU re-analysis | ✅ curvature 0.1147 median at the anchor |
| B.4 reference cross-validation | ✅ all three published constants recovered to ~1e-10 |
| B.5 throughput | ✅ 20-min sustained run done; 2.6× worse than assumed, no meaningful throttling |
| B.6 VRAM pilot | ✅ operating points fixed; spill trap found and guarded |
| B.7 power pilot | ⛔ **blocked on the budget decision** |
| B.8 power simulation | ⛔ blocked on B.7 |

The abort condition is **not** triggered: the minimum design (2 architectures × 5
vocabularies × 5 seeds) is fundable at ~83 h. But the full design costs 109 h, and that is
a decision about how much of a laptop's life to spend, not a technical blocker.

---

# ADDENDUM — the ≥20-minute sustained run (plan requirement B.5)

The short probes above were **25–30 seconds**, and that turned out to matter in the
opposite direction from the one anticipated.

## Result: essentially no thermal throttling

16M, V=4480, b=8, logged in 15-second windows (`results/sustained_windows.csv`):

| | |
|---|---|
| throttle ratio (last quartile / first) | **0.9953** |
| steady-state mean | **6.214 TFLOP/s** |
| steady-state range | 6.194 – 6.234 |
| peak memory | 3.688 GB, constant |

After the first 30 seconds the rate is flat to within ±0.3% for the rest of the run. GPU
telemetry corroborates: temperature settles at ~75 °C, SM clock holds 1860–1905 MHz, power
~52 W (a chassis power cap, not a thermal wall). **This laptop does not meaningfully
throttle at this workload** — better than the plan assumed.

## The short probes UNDERSTATED throughput by 22%

Sustained 6.214 vs probe 5.10 TFLOP/s at the same configuration: a ramp correction of
**1.218×**. The cause is visible in the telemetry — the GPU climbs from 210 MHz idle to
1905 MHz over the first seconds, and a 25-second window averages over that ramp. A short
benchmark measures the ramp; only a long one measures the steady state.

This is worth stating plainly because it cuts against the usual intuition. The reason the
plan demanded ≥20 minutes was to catch throttling making things *worse*. What it actually
caught was the short-probe method making things look worse than they are. The requirement
earned its place either way.

## Corrected budget

| | GPU-hours |
|---|---|
| with probe rates (pessimistic, ramp-contaminated) | 108.9 |
| **with sustained rates** | **89.4** (3.7 days continuous) |
| plan's original assumption | 34.6 |

Ratio to plan: **2.6×**, down from 3.2×.

| scale | rate | hours | basis |
|---|---|---|---|
| 2M | 3.63 | 1.6 | ramp-corrected probe |
| 4M | 4.26 | 4.6 | ramp-corrected probe *(4M never measured directly)* |
| 8M | 5.32 | 16.4 | ramp-corrected probe |
| 16M | **6.21** | **49.8** | **measured sustained** |
| pilot | 5.32 | 9.8 | ramp-corrected |
| anchor | 6.21 | 7.1 | ramp-corrected |

**Caveat on honesty of these numbers:** only 16M was measured under sustained load. The
other three scales apply the 1.218× correction measured at 16M, which assumes the ramp
effect is scale-independent. It probably is — it is a clock-boost artefact, not a
model property — but it is an assumption, and 4M has never been benchmarked at all. Before
committing to the full sweep, each scale should get one sustained run of its own.

16M remains **56% of the total**, so the tension with the preregistered shrink rule
described above is unchanged.

---

# FINAL — all four scales measured under sustained load

10-minute sustained runs per scale (rather than 20): the 16M run established that steady
state is reached inside 30 s and holds flat, so 10 min yields ~39 windows well past the
ramp. Deviation from the plan's ">= 20 min" is recorded here deliberately rather than
silently.

| scale | batch | rate (TFLOP/s) | throttle ratio | windows | basis |
|---|---|---|---|---|---|
| 2M | 32 | **3.330** | — (see below) | 14 clean | uncontended tail |
| 4M | 16 | **4.502** | 0.9985 | 39 | clean |
| 8M | 8 | **5.369** | 0.9993 | 39 | clean |
| 16M | 8 | **6.214** | 0.9953 | 37 | clean |

Ramp-correction check — the 1.218x factor measured at 16M, applied blind to the others,
predicted 3.63 / 4.26 / 5.32 against measured 3.330 / 4.502 / 5.369: **-8.3%, +5.7%,
+0.9%**. Good enough that the assumption was sound, and now retired in favour of
measurement.

## A measurement error, caught only because windows were logged incrementally

The 2M run reported an aggregate 2.048 TFLOP/s and a "throttle ratio" of **2.6161** — the
last quartile running 2.6x FASTER than the first. The window curve shows why:

```
  15s -> 375s :  1.267 TFLOP/s   (flat, 24 windows)
  391s -> 589s:  3.330 TFLOP/s   (flat, 14 windows)
```

A step function, not a thermal curve. The 20-minute 16M sustained run was still occupying
the GPU when this one was launched; the step at t≈390 s is where it finished. Two jobs on
one GPU, entirely self-inflicted.

**The aggregate alone (2.048) would have entered the budget as a plausible wrong number
with nothing to flag it.** Only the per-window log made the contamination visible and the
uncontended tail salvageable. This is the concrete argument for the plan's requirement to
log throughput per run rather than per configuration — it was written to catch throttled
runs, and it caught a contaminated one instead.

Rule adopted: **one GPU job at a time, verified before launch**, not assumed.

## Final budget — every rate measured

| component | rate | hours |
|---|---|---|
| 2M sweep + M2 | 3.330 | 1.8 |
| 4M sweep + M2 | 4.502 | 4.4 |
| 8M sweep + M2 | 5.369 | 16.3 |
| 16M sweep + M2 | 6.214 | **49.8** |
| power pilot | 5.369 | 9.8 |
| anchor | 6.214 | 7.1 |
| **TOTAL at n=5** | | **89.1 GPU-hours** |

**89.1 GPU-hours ≈ 3.7 days continuous. 2.6x the plan's 34.6 h assumption, and 82% of the
earlier probe-based 108.9 h figure.** No scale rests on an inferred rate any more.

16M remains **56%** of the total, so the tension with the preregistered shrink rule is
unchanged and still requires a deliberate decision if the budget is to come down further.

## Stage B: COMPLETE except the power pilot

B.1–B.6 done. B.7 (power pilot) and B.8 (power simulation) require the data path that does
not exist yet: SlimPajama ingestion, BPE training at each vocabulary, `L_u` with unigram
probabilities frozen from the training split, and the training loop on Tao's schedule.
