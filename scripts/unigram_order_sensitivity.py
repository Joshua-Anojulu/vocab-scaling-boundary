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
SEEDS = (0, 1, 2)
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
    bias_v = np.array([r["bias_nats"] for r in rows])
    slopes = np.abs(np.diff(bias_v) / np.diff(lnv))
    worst_slope = float(slopes.max()) if slopes.size else 0.0
    displacement = worst_slope / CURVATURE_FLOOR

    out = {
        "seeds": list(SEEDS),
        "curvature_floor": CURVATURE_FLOOR,
        "m1_margin_ln": M1_MARGIN,
        "rows": rows,
        "worst_abs_bias_nats": float(np.abs(bias_v).max()),
        "worst_seed_sd_nats": max(r["seed_sd_nats"] for r in rows),
        "worst_local_slope_nats_per_lnv": worst_slope,
        "implied_lnv_displacement": displacement,
        "margin_factor": (M1_MARGIN / displacement) if displacement else float("inf"),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nworst |bias|      {out['worst_abs_bias_nats']:.3e} nats")
    print(f"worst seed SD     {out['worst_seed_sd_nats']:.3e} nats")
    print(f"worst local slope {worst_slope:.3e} nats per ln V")
    print(f"implied argmin displacement {displacement:.3e} in ln V")
    print(f"M1 margin is {out['margin_factor']:.1f}x larger")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
