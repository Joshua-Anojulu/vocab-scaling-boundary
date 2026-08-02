"""Exact budget error under the old and new step-count conventions.

Why this exists: the seed-semantics review found that `TrainConfig.total_steps` ceiled the
step count while claiming to trim, so every run overshot its token budget. IsoFLOP here is
enforced on exact token counts, which makes an overshoot a direct, `V`-dependent
perturbation of `C` -- the same character as the EOS effect that amendment A4 treats as
decision-relevant. The size of that perturbation therefore has to be a measured number in an
artifact, not a figure quoted in prose. Three earlier numbers in this project were quoted
from rounded printed values and had to be corrected; this closes that route.

Scope: the Stage B.7 pilot only. The confirmatory `C` at the other three scales is not fixed
until the pilot completes, so no claim is made about them here.

    python scripts/budget_convention_error.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import extension as ext, train as T  # noqa: E402

PILOT_C = 1.03084e16
PILOT_NNV = 8_000_000
V_GRID = (768, 1536, 3200, 6272, 12672)
V_RUN = 3200
BATCH_SHAPES = ((4, 4), (8, 8))
OUT = Path(__file__).resolve().parents[1] / "results" / "budget_convention_error.json"


def target_tokens(nnv: int, d: int, V: int, budget: float) -> int:
    """IsoFLOP is enforced on TOKENS, not characters: T = C / (6 * (N_nv + V*d))."""
    return int(budget / (6 * (nnv + V * d)))


def main() -> None:
    arch = ext.arch_for(PILOT_NNV)
    block = T.BLOCK_SIZE
    cells = []

    arms = [(V, PILOT_C, "C") for V in V_GRID] + [(V_RUN, 1.1 * PILOT_C, "1.1C")]
    for V, budget, arm in arms:
        tt = target_tokens(arch.nnv, arch.d, V, budget)
        for mb, ga in BATCH_SHAPES:
            cfg = T.TrainConfig(target_tokens=tt, micro_batch=mb, block_size=block,
                                grad_accum=ga)
            tps = cfg.tokens_per_step
            # OLD: ceil the STEP count and run every step at full width -- never trimmed.
            old = math.ceil(tt / tps) * tps
            # NEW: largest whole number of sequences not exceeding the budget.
            new = cfg.target_sequences * block
            cells.append({
                "V": V, "arm": arm, "micro_batch": mb, "grad_accum": ga,
                "target_tokens": tt, "tokens_per_step": tps,
                "old_tokens": old, "new_tokens": new,
                "old_err_pct": 100.0 * (old - tt) / tt,
                "new_err_pct": 100.0 * (new - tt) / tt,
            })

    by_shape = {}
    for mb, ga in BATCH_SHAPES:
        sel = [c for c in cells if (c["micro_batch"], c["grad_accum"]) == (mb, ga)]
        by_shape[f"mb{mb}_ga{ga}"] = {
            "worst_old_overshoot_pct": max(c["old_err_pct"] for c in sel),
            "worst_new_undershoot_pct": min(c["new_err_pct"] for c in sel),
        }

    out = {
        "pilot_C": PILOT_C,
        "arch": {"target_nnv": PILOT_NNV, "d": arch.d, "n_layer": arch.n_layer,
                 "nnv": arch.nnv},
        "block_size": block,
        "cells": cells,
        "worst_by_batch_shape": by_shape,
        "worst_old_overshoot_pct": max(c["old_err_pct"] for c in cells),
        "worst_new_undershoot_pct": min(c["new_err_pct"] for c in cells),
        "new_never_exceeds_target": all(c["new_tokens"] <= c["target_tokens"] for c in cells),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'V':>6} {'arm':>5} {'mb/ga':>7} {'T_target':>14} {'old %':>10} {'new %':>10}")
    for c in cells:
        shape = f"{c['micro_batch']}/{c['grad_accum']}"
        print(f"{c['V']:>6} {c['arm']:>5} {shape:>7} {c['target_tokens']:>14,} "
              f"{c['old_err_pct']:>+10.4f} {c['new_err_pct']:>+10.4f}")
    print()
    for shape, w in by_shape.items():
        print(f"{shape}: worst old {w['worst_old_overshoot_pct']:+.4f}%, "
              f"worst new {w['worst_new_undershoot_pct']:+.4f}%")
    print(f"\nnew convention never exceeds target: {out['new_never_exceeds_target']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
