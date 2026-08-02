"""How much does the unigram baseline move when a seed permutes sequence order?

The seed-semantics review found that `L_u`'s baseline was fitted on a contiguous prefix
`train_tokens[:consumed]` while a seed-permuted run consumes a SCATTERED subset. That is now
fixed. This measures what the fix is worth, because this project's standing rule is that a
mechanism is not established until it is measured -- four claims here have been asserted
ahead of measurement and three of them were wrong.

Two questions, both answered against real tokenized pilot data:

  1. **Bias.** How far is the old prefix-fitted `H_unigram` from the correct fit on the
     tokens the run actually read? This is a systematic error in the primary metric.
  2. **Seed spread.** How much does the correct `H_unigram` move BETWEEN seeds? This is a
     real component of seed variance that a prefix fit would have set to exactly zero --
     and Stage B.7 exists to estimate seed variance.

Both are reported in nats/token against the M1 decision margin, so the answer is
decision-relevant rather than merely small.

    python scripts/unigram_order_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate as E, extension as ext, reference as ref, train as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "tokens"
OUT = ROOT / "results" / "unigram_order_sensitivity.json"
SEEDS = tuple(range(8))
"""Eight, not the three of the first run.

An SD from n=3 carries a relative standard error near 50%, which is too loose to lean on
even for a "this component is real but small" claim. Eight brings that to about 27%. The
seed count was raised after seeing the first result, which is legitimate here and would not
be in the study itself: this measures a property of the implementation, not a preregistered
outcome, and nothing about the study's inference depends on it.
"""
# Slope of L_u in ln V near the optimum, from the P2/P4 curvature work: the local-5
# curvature floor. Used only to convert nats into a decision-margin fraction.
CURVATURE_FLOOR = 0.00965
M1_MARGIN = float(np.log(1.5))


def h_unigram(logp: np.ndarray, targets: np.ndarray) -> float:
    """Mean unigram NLL in nats over the evaluation targets -- the H_unigram in L_u."""
    return float(-logp[targets].sum() / len(targets))


def main() -> None:
    arch = ext.arch_for(8_000_000)
    pilot_c = float(ref.Nnvopt_to_flops(arch.nnv))
    block = T.BLOCK_SIZE
    rows = []

    for V in ext.vocab_grid(arch):
        train_p = TOKENS / f"v{V}_train.npy"
        val_p = TOKENS / f"v{V}_selection_val.npy"
        if not (train_p.exists() and val_p.exists()):
            print(f"V={V}: tokens missing, skipped")
            continue

        train = np.load(train_p, mmap_mode="r")
        val = np.load(val_p, mmap_mode="r")
        target_tokens = int(ext.target_tokens(pilot_c, arch.nnv, arch.d, V))
        n_seq = target_tokens // block

        # Evaluation targets are identical across seeds; only the FIT set changes.
        plan = E.plan_blocks(len(val), block, batch=4)
        targets = E.scored_targets(val, plan)

        prefix_logp = E.unigram_logp(train, V, consumed=target_tokens)
        h_prefix = h_unigram(prefix_logp, targets)

        h_seeds = []
        for s in SEEDS:
            stream = T.TokenStream(np.asarray(train), block, order_seed=s)
            logp = E.unigram_logp(train, V, consumed=target_tokens,
                                  order=stream.order, block_size=block)
            h_seeds.append(h_unigram(logp, targets))

        h_mean = float(np.mean(h_seeds))
        bias = h_mean - h_prefix
        spread = float(np.std(h_seeds, ddof=1))
        rows.append({
            "V": V, "target_tokens": target_tokens, "sequences": n_seq,
            "h_prefix": h_prefix, "h_permuted_mean": h_mean, "h_permuted_seeds": h_seeds,
            "bias_nats": bias, "seed_sd_nats": spread,
        })
        print(f"V={V:>6}  H_prefix={h_prefix:.8f}  H_perm={h_mean:.8f}  "
              f"bias={bias:+.3e}  seed_sd={spread:.3e}")

    if not rows:
        print("no vocabularies had tokenized data; nothing measured")
        return

    # A cross-V TILT in the baseline shifts the argmin; a common offset does not. Convert
    # the worst local slope in ln V into an M1-margin fraction, the same conversion the
    # P2/P4 resolution used.
    lnv = np.log([r["V"] for r in rows])

    def margin_factor(h_by_v: np.ndarray) -> tuple[float, float, float]:
        """(worst slope, displacement, margin factor) from per-V mean permuted H."""
        bias = h_by_v - np.array([r["h_prefix"] for r in rows])
        s = np.abs(np.diff(bias) / np.diff(lnv))
        worst = float(s.max()) if s.size else 0.0
        disp = worst / CURVATURE_FLOOR
        return worst, disp, (M1_MARGIN / disp if disp else float("inf"))

    per_seed = np.array([r["h_permuted_seeds"] for r in rows])       # (n_V, n_seeds)
    bias_v = np.array([r["bias_nats"] for r in rows])
    worst_slope, displacement, factor = margin_factor(per_seed.mean(axis=1))

    # The slope is a finite difference of seed-AVERAGED biases, so its noise is set by
    # seed_sd/sqrt(n) at each point -- and that noise is not small next to the slope itself.
    # Bootstrap the seed vector jointly, matching the study's seed-level block bootstrap,
    # so the reported margin comes with an interval instead of a misleadingly exact figure.
    rng = np.random.default_rng(0)
    n_seeds = per_seed.shape[1]
    boot = []
    for _ in range(4000):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot.append(margin_factor(per_seed[:, idx].mean(axis=1))[2])
    boot = np.array(boot)
    factor_lo, factor_hi = (float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5)))

    out = {
        "seeds": list(SEEDS),
        "curvature_floor": CURVATURE_FLOOR,
        "m1_margin_ln": M1_MARGIN,
        "rows": rows,
        "n_seeds": len(SEEDS),
        "sd_relative_standard_error": float(1.0 / np.sqrt(2 * (len(SEEDS) - 1))),
        "worst_abs_bias_nats": float(np.abs(bias_v).max()),
        "worst_seed_sd_nats": max(r["seed_sd_nats"] for r in rows),
        "pooled_seed_sd_nats": float(
            np.sqrt(np.mean([r["seed_sd_nats"] ** 2 for r in rows]))
        ),
        "worst_seed_sd_note": (
            "A max over per-vocabulary SD estimates is biased UPWARD by selection. That is "
            "conservative for the conclusion drawn from it -- it overstates the variance "
            "component whose existence argues FOR fitting the baseline on the consumed set "
            "-- so no correction is applied. pooled_seed_sd_nats is the unselected summary."
        ),
        "worst_local_slope_nats_per_lnv": worst_slope,
        "implied_lnv_displacement": displacement,
        "margin_factor": factor,
        "margin_factor_ci95": [factor_lo, factor_hi],
        "margin_factor_note": (
            "The point estimate is NOT stable at small seed counts: it read 23.0x at three "
            "seeds and 31.0x at eight, because the slope is a finite difference of "
            "seed-averaged biases whose noise is seed_sd/sqrt(n) and is not small next to "
            "the slope itself. Quote the interval, or quote the conclusion that the "
            "displacement is an order of magnitude inside the margin -- never the point "
            "estimate's last digit."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nworst |bias|      {out['worst_abs_bias_nats']:.3e} nats")
    print(f"worst seed SD     {out['worst_seed_sd_nats']:.3e} nats")
    print(f"worst local slope {worst_slope:.3e} nats per ln V")
    print(f"implied argmin displacement {displacement:.3e} in ln V")
    print(f"M1 margin is {factor:.1f}x larger  (95% CI {factor_lo:.1f}x - {factor_hi:.1f}x)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
