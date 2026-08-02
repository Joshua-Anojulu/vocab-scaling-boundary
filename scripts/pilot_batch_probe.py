"""Which MICRO-batch should Stage B.7 run at?

Only the micro-batch. The effective batch is not up for measurement: the reference fixes it
at 512 sequences per optimizer step and its `learning_rate = 4e-4` is tuned to that, so
`grad_accum` is derived rather than chosen (see `pilot.GLOBAL_BATCH_SEQUENCES`). What remains
free is how that global batch is split across memory, which is a pure
throughput-and-VRAM question with no bearing on the optimization regime.

Measured at the PILOT scale (8M non-vocabulary parameters) across the pilot vocabularies,
because the existing probe in `results/bench_probe.json` was run at 16M with a single
vocabulary and does not transfer: the output head is `V*d` and these vocabularies span 768 to
12672, a 16.5x range in the term that dominates at small `N_nv`.

    python scripts/pilot_batch_probe.py --seconds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bench, pilot, train as T  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "pilot_batch_probe.json"

MICRO_BATCHES = (2, 4, 8, 16)
"""Candidate micro-batches. `grad_accum` is DERIVED, never swept alongside it.

An earlier version of this probe compared `(mb=4, ga=4)` against `(mb=8, ga=8)`, which
conflated two independent things: the micro-batch is a memory/throughput knob, while the
product `mb * ga` is the effective batch and a RECIPE parameter that the reference fixes at
512 sequences. Sweeping them together would have measured throughput at effective batches of
16 and 64 -- neither of which the pilot will run at -- and the optimizer-step overhead is
amortised very differently at `ga=128` than at `ga=4`. Each candidate is now measured at the
`grad_accum` it will actually use.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="measurement window per cell; the pilot itself runs for hours")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cells = pilot.pilot_cells()
    # One cell per vocabulary; the 1.1C arm has the same shape as its C arm and would only
    # duplicate the measurement.
    vocabs = sorted({c.vocab_size for c in cells})
    rows = []

    for vocab in vocabs:
        for mb in MICRO_BATCHES:
            ga = pilot.grad_accum_for(mb)
            try:
                r = bench.run_bench(
                    target=pilot.PILOT_NNV, vocab=vocab, block=T.BLOCK_SIZE,
                    batch=mb, grad_accum=ga, seconds=args.seconds, device=args.device,
                )
                row = asdict(r)
                row["micro_batch"], row["grad_accum"] = mb, ga
            except Exception as exc:                      # OOM is a RESULT, not a crash
                row = {
                    "vocab": vocab, "micro_batch": mb, "grad_accum": ga, "ok": False,
                    "note": bench.classify_failure(exc),
                    "error": traceback.format_exception_only(type(exc), exc)[-1].strip(),
                    "tokens_per_sec": 0.0, "peak_mem_gb": float("nan"),
                }
            rows.append(row)
            status = "ok" if row.get("ok") else f"FAILED ({row.get('note')})"
            print(f"V={vocab:>6} mb={mb:<3}ga={ga:<4}: {row.get('tokens_per_sec', 0):>10,.0f} tok/s  "
                  f"peak {row.get('peak_mem_gb', float('nan')):.2f} GB  {status}", flush=True)

    # Decide on total projected pilot wall-clock, not on peak throughput at one vocabulary:
    # the vocabularies have different budgets, so the cheapest shape overall is what matters.
    summary = {}
    for mb in MICRO_BATCHES:
        ga = pilot.grad_accum_for(mb)
        sel = {r["vocab"]: r for r in rows
               if (r.get("micro_batch"), r.get("grad_accum")) == (mb, ga)}
        if not all(sel.get(c.vocab_size, {}).get("ok") for c in cells):
            summary[f"mb{mb}_ga{ga}"] = {"usable": False,
                                         "reason": "at least one vocabulary failed"}
            continue
        secs = sum(c.target_tokens / sel[c.vocab_size]["tokens_per_sec"] for c in cells)
        summary[f"mb{mb}_ga{ga}"] = {
            "usable": True,
            "projected_hours_per_seed": secs / 3600.0,
            "projected_hours_18_runs": secs * len(pilot.PILOT_SEEDS) / 3600.0,
            "worst_peak_mem_gb": max(sel[c.vocab_size]["peak_mem_gb"] for c in cells),
            "worst_throttle_ratio": min(sel[c.vocab_size]["throttle_ratio"] for c in cells),
        }

    usable = {k: v for k, v in summary.items() if v.get("usable")}
    best = min(usable, key=lambda k: usable[k]["projected_hours_18_runs"]) if usable else None

    out = {
        "pilot_nnv": pilot.PILOT_NNV, "block_size": T.BLOCK_SIZE,
        "measurement_seconds": args.seconds,
        "global_batch_sequences": pilot.GLOBAL_BATCH_SEQUENCES,
        "note": (
            "Throughput is measured on random token ids, which is faithful for compute but "
            "not for cache behaviour on real data; treat the projection as a ranking of the "
            "two shapes rather than as a wall-clock estimate."
        ),
        "rows": rows, "summary": summary, "recommended": best,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    for k, v in summary.items():
        if v.get("usable"):
            print(f"{k}: {v['projected_hours_18_runs']:.2f} h for 18 runs, "
                  f"peak {v['worst_peak_mem_gb']:.2f} GB, "
                  f"worst throttle {v['worst_throttle_ratio']:.3f}")
        else:
            print(f"{k}: unusable -- {v['reason']}")
    print(f"\nrecommended: {best}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
