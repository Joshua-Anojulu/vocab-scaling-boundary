"""Does a 13-19 step warmup destabilise training at the pilot configuration?

NOT a pilot run. This trains a few tens of optimizer steps and looks at the loss curve; it
produces no `L_u` and nothing here enters inference. Output goes to
`results/warmup_stability.json` -- committed, because a claim cited in `AMENDMENTS.md` must
rest on evidence a reader can open.

The question is specific. A7 fixes warmup at 10% of run length. At Tao's 33M scale that was
~114 optimizer steps; at this study's budgets the same fraction is **13-19 steps**, because
the runs are short. Warmup exists to keep Adam's second
moment estimates from producing an enormous early step, and 13 steps is not obviously enough.
If it is not, the loss curve shows it in exactly this window: a spike, a plateau at chance, or
a NaN.

Checking costs minutes. Discovering it 30 minutes into an 18-run pilot costs the pilot.

    python scripts/warmup_stability_check.py --steps 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import model as M, pilot as P, train as T  # noqa: E402

# results/, not runs/ -- `runs/` is gitignored, so an artifact there cannot support a claim
# anyone else can check. An earlier version wrote to runs/diagnostics/ while AMENDMENTS.md
# cited its numbers: the same "evidence nobody can see" failure as the skipping test.
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30,
                    help="optimizer steps; the point is to cover warmup and a little past it")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ga = P.grad_accum_for(args.micro_batch)
    results = []

    # Smallest and largest vocabulary: the extremes of both run length and head size.
    cells = P.pilot_cells()
    probe_cells = [min(cells, key=lambda c: c.vocab_size),
                   max(cells, key=lambda c: c.vocab_size)]

    for cell in probe_cells:
        full = T.TrainConfig(target_tokens=cell.target_tokens, micro_batch=args.micro_batch,
                             block_size=T.BLOCK_SIZE, grad_accum=ga, seed=0)
        real_warmup = full.warmup_steps
        real_total = full.total_steps

        # Train only `--steps` steps, but with the warmup length the REAL run would use.
        #
        # An earlier version set `warmup_fraction = real_warmup / real_total`, which is
        # WRONG: `warmup_steps` is `round(total_steps * warmup_fraction)` and `total_steps`
        # here is the PROBE's 30, not the real 131-190. That produced a 3-step warmup while
        # claiming to test 13-19 -- the check did not test what its output was cited for.
        # The fraction must be taken against the probe's own step count.
        if real_warmup >= args.steps:
            raise SystemExit(f"--steps {args.steps} must exceed the real warmup {real_warmup}")
        budget = args.steps * ga * args.micro_batch * T.BLOCK_SIZE
        cfg = T.TrainConfig(target_tokens=budget, micro_batch=args.micro_batch,
                            block_size=T.BLOCK_SIZE, grad_accum=ga, seed=0,
                            device=args.device, log_every_s=0.0)
        cfg = type(cfg)(**{**cfg.__dict__, "warmup_fraction": real_warmup / args.steps})
        assert cfg.warmup_steps == real_warmup, (
            f"probe warmup {cfg.warmup_steps} != real {real_warmup}; the bug this assert "
            f"exists to prevent is exactly the one that shipped once already"
        )

        train = np.load(P.TOKENS / f"v{cell.vocab_size}_train.npy", mmap_mode="r")
        stream = T.TokenStream(np.asarray(train), T.BLOCK_SIZE, order_seed=0)
        torch.manual_seed(0)
        model = M.build(M.ModelConfig(vocab_size=cell.vocab_size, d=cell.d,
                                      n_layer=cell.n_layer, n_head=cell.n_head,
                                      d_ffn=cell.d_ffn, block_size=T.BLOCK_SIZE))

        print(f"\n=== V={cell.vocab_size}  real run: {real_total} steps, "
              f"warmup {real_warmup}  |  probing {args.steps} steps ===", flush=True)
        res = T.train_run(model, stream, cfg)
        curve = [l for _, l in res.loss_curve]

        chance = float(np.log(cell.vocab_size))
        finite = all(np.isfinite(curve))
        post = curve[real_warmup:]
        peak_after_warmup = max(post) if post else None
        # A transient dip below chance is not stability. Require the loss to END below
        # chance and to STAY there over the last few logged steps, and measure the spike
        # against the value at the end of warmup rather than against the global max.
        tail = post[-5:] if len(post) >= 5 else post
        sustained = bool(tail) and all(v < chance for v in tail)
        at_warmup_end = curve[real_warmup - 1] if real_warmup <= len(curve) else curve[0]
        spike = (peak_after_warmup is not None and peak_after_warmup > at_warmup_end * 1.5)
        row = {
            "vocab_size": cell.vocab_size,
            "real_total_steps": real_total, "real_warmup_steps": real_warmup,
            "probe_steps": args.steps,
            "chance_loss": chance,
            "first_loss": curve[0], "last_loss": curve[-1],
            "min_loss": min(curve), "max_loss": max(curve),
            "all_finite": finite,
            "dipped_below_chance": min(curve) < chance,
            "ended_below_chance": curve[-1] < chance,
            "sustained_below_chance": sustained,
            "loss_at_warmup_end": at_warmup_end,
            "max_after_warmup": peak_after_warmup,
            "spike_after_warmup": spike,
            "curve": curve,
        }
        results.append(row)
        print(f"  chance {chance:.3f}  first {curve[0]:.3f}  last {curve[-1]:.3f}  "
              f"min {min(curve):.3f}  finite={finite}", flush=True)

    (OUT / "warmup_stability.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = all(r["all_finite"] and r["sustained_below_chance"] and not r["spike_after_warmup"]
             for r in results)
    print("\n" + ("STABLE: no NaN, loss below chance, no post-warmup spike"
                  if ok else "UNSTABLE -- inspect the curves before running the pilot"))
    print(f"wrote {OUT / 'warmup_stability.json'}")


if __name__ == "__main__":
    main()
