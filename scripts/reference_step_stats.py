"""Derive Tao's optimizer-step distribution into a COMMITTED artifact.

`reference/exp_data.csv` is gitignored as regenerable upstream data, which meant the test
pinning A7's step-count claim silently SKIPPED in any clone that had not downloaded it --
including the reviewer's. A test that guards a claim and does not run is worse than no test,
because it reads as coverage.

This writes `results/reference_step_stats.json`, which is committed, so the test always runs.
The CSV's sha256 is recorded alongside, so when the CSV IS present the test can verify the
artifact still describes it rather than trusting a stale summary.

Two step conventions are reported, because they differ and the distinction matters:

* `nominal` -- derived as `num_characters * f(V) / (512 * 2048)`, which is what the CSV
  supports. These are fractional (57.22, 114.44, ...).
* `artifact` -- the released checkpoint step numbers, which are the integers Tao's own
  tooling emitted. The reviewer noted these run `step-000060` to `step-001200` for the 33M
  family, so the nominal derivation and the released artifacts do not agree exactly.

The nominal values are used for comparison because they are what this repository can
recompute; the artifact values are recorded so the discrepancy is visible rather than
buried.

    python scripts/reference_step_stats.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import reference as ref, train as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reference" / "exp_data.csv"
OUT = ROOT / "results" / "reference_step_stats.json"
GLOBAL_BATCH = 512
# Released checkpoint filenames for the 33M family, step-000060 .. step-001200 at even
# spacing -- 20 evaluations, matching compute_eval_steps(max_steps, evals_per_interval=20).
# Quantiles are recorded, not just the endpoints, because AMENDMENTS.md claims the
# lower-tail conclusion holds under BOTH conventions and that is only checkable if both
# distributions are present.
ARTIFACT_STEPS_33M = [60 * k for k in range(1, 21)]


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found; fetch it from sail-sg/scaling-with-vocab first")

    raw = CSV.read_bytes()
    d = pd.read_csv(CSV)
    d["steps"] = (d.num_characters * d.vocab_size.map(ref.fertility)
                  / (GLOBAL_BATCH * T.BLOCK_SIZE))

    # Exact smallest family rather than a magic threshold: the value is a specific
    # architecture's parameter count, not a rounded cutoff that happens to work.
    smallest_nnv = float(d.Non_vocab_parameters.min())
    sm = d[d.Non_vocab_parameters == smallest_nnv]

    qs = {f"q{int(p*100):02d}": float(np.percentile(sm.steps, p * 100))
          for p in (0, 0.10, 0.25, 0.50, 0.75, 1.0)}
    out = {
        "source": "sail-sg/scaling-with-vocab exp_data.csv",
        "csv_sha256": hashlib.sha256(raw).hexdigest(),
        "csv_rows": int(len(d)),
        "global_batch_sequences": GLOBAL_BATCH,
        "block_size": T.BLOCK_SIZE,
        "smallest_non_vocab_parameters": smallest_nnv,
        "smallest_family_rows": int(len(sm)),
        "evals_per_run": 20,
        "runs_in_smallest_family": int(len(sm) // 20),
        "nominal_steps": qs,
        "artifact_steps_33m": {
            "steps": ARTIFACT_STEPS_33M,
            "note": "released checkpoint filenames, step-000060 .. step-001200",
            "quantiles": {f"q{int(x*100):02d}": float(np.percentile(ARTIFACT_STEPS_33M, x*100))
                          for x in (0, 0.10, 0.25, 0.50, 0.75, 1.0)},
        },
        "note": (
            "Rows are 20 in-training evaluations per (vocabulary, scale), from "
            "compute_eval_steps(max_steps, evals_per_interval=20) -- NOT separate runs. "
            "The step values are therefore checkpoint positions within a run, and the "
            "largest is that run's length."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
