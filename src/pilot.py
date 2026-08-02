"""Stage B.7: the 18-run power pilot, and the run contract every A6 run must satisfy.

Six configurations x three seeds at the 8M scale. Its only purpose is to estimate the
seed-level variance that sizes `n` for the 139 confirmatory runs, which is why amendment A6
had to be settled first: a pilot run under initialisation-only seeds would estimate the
wrong variance component and the whole pilot would have to be repeated.

**The design rule here is that a correct run must be the only reachable one.** Two escape
hatches exist for smoke tests -- `TrainConfig.unseeded_order_ok` and
`evaluate_run(sequential_order_ok=True)` -- and each is fatal if set on a study run. Rather
than ask a caller to remember not to set them, `run_cell` constructs the stream, the config
and the evaluation itself and never exposes either flag. The defect A6 exists to fix was a
contract that lived in prose while the code satisfied it by accident; leaving the runner free
to violate it would rebuild that at one remove.

The specific obligations, each enforced below rather than documented:

* the stream is built over the WHOLE token array, never a budget slice, or the permutation
  domain shrinks and the `1.1*C` arm stops nesting its `C` arm;
* `order_seed` is the run's seed, so data order and initialisation vary together;
* both arms of a `(vocabulary, seed)` share a permutation, which the audit re-verifies from
  the recorded digests rather than trusting;
* `H_unigram` is fitted on the sequences the run actually consumed;
* the witness fields travel with the metrics, so nesting is checkable after the fact.

    python -m src.pilot --dry-run     # plan and verify, no GPU
    python -m src.pilot               # execute
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import extension as ext
from . import model as M
from . import reference as ref
from . import tokenizer as tk
from . import train as T

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "tokens"
TOKENIZERS = ROOT / "tokenizers"
RUNS = ROOT / "runs" / "pilot"

PILOT_NNV = 8_000_000
PILOT_SEEDS = (0, 1, 2)
M2_MULTIPLIER = 1.1

GLOBAL_BATCH_SEQUENCES = 512
"""Sequences per OPTIMIZER step, matching `reference/tinyllama_pretrain.py:37`.

This is a recipe parameter, not a hardware one, and `PLAN.md` never fixed it -- it says
only "the shared Tao training recipe". It is fixed here because the released recipe PAIRS
`global_batch_size = 512` with `learning_rate = 4e-4`, so changing one while keeping the
other departs from that pairing. Note what this does NOT claim: the reference gives no
evidence of how `4e-4` was selected, and an earlier version of this docstring asserted the
two were tuned together. That was withdrawn in review. `micro_batch` is purely a
memory-partitioning knob, exactly as in the reference, where
`gradient_accumulation_steps = batch_size // micro_batch_size` is DERIVED (`:125`).

Checked against Tao's released runs rather than assumed. At their smallest fitted scale
(33M non-vocabulary parameters) their runs span **57 to 1,144 optimizer steps**, median
601 (nominal; their released checkpoint filenames imply a median nearer 630). This pilot at
8M lands at **131-190 optimizer updates** -- in the LOWER TAIL, roughly the 10th-15th
percentile and 0.21-0.30x their median, above their minimum but not typical of their grid.
Range inclusion alone would be a weak test, since their range spans 20x.

Running instead at an effective batch of 16 sequences would give ~6,000 updates, **10x their
median**, at a learning rate the released recipe pairs with a batch 32x larger; that is the
larger departure.
"""


@dataclass(frozen=True)
class Cell:
    """One (vocabulary, arm) configuration. Seeds are applied on top."""

    vocab_size: int
    arm: str                    # "C" or "1.1C"
    budget_flops: float
    target_tokens: int
    d: int
    n_layer: int
    n_head: int
    d_ffn: int
    nnv: int

    @property
    def label(self) -> str:
        return f"v{self.vocab_size}_{self.arm.replace('.', 'p')}"


def pilot_cells() -> list[Cell]:
    """The six configurations: five vocabularies at C, plus V_run at 1.1*C for M2."""
    arch = ext.arch_for(PILOT_NNV)
    c = float(ref.Nnvopt_to_flops(arch.nnv))
    grid = ext.vocab_grid(arch)
    v_run = ext.v_run(arch)
    if v_run not in grid:
        raise ValueError(f"V_run={v_run} is not on the grid {grid}; the M2 arm would not "
                         f"pair with any C arm")

    spec = [(v, c, "C") for v in grid] + [(v_run, M2_MULTIPLIER * c, "1.1C")]
    return [
        Cell(vocab_size=v, arm=arm, budget_flops=budget,
             target_tokens=int(ext.target_tokens(budget, arch.nnv, arch.d, v)),
             d=arch.d, n_layer=arch.n_layer, n_head=arch.n_head, d_ffn=arch.d_ffn,
             nnv=arch.nnv)
        for v, budget, arm in spec
    ]


def _load_split(vocab_size: int, split: str) -> np.ndarray:
    path = TOKENS / f"v{vocab_size}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing tokenized split: {path}")
    return np.load(path, mmap_mode="r")


def check_capacity(cells: list[Cell], block_size: int = T.BLOCK_SIZE) -> list[dict]:
    """Every cell must have `target_sequences * B + 1` ids available BEFORE any GPU time.

    The `+1` is the lookahead token that supplies the last sequence's final target. Checked
    up front because discovering it 40 minutes into a run wastes the run, and because the
    arrays were sized to budget with only ~2% headroom.
    """
    rows = []
    for cell in cells:
        train = _load_split(cell.vocab_size, "train")
        need_seq = cell.target_tokens // block_size
        need_ids = need_seq * block_size + 1
        rows.append({
            "cell": cell.label, "vocab_size": cell.vocab_size, "arm": cell.arm,
            "target_tokens": cell.target_tokens, "need_sequences": need_seq,
            "need_ids": need_ids, "have_ids": int(len(train)),
            "headroom_ids": int(len(train)) - need_ids,
            "ok": int(len(train)) >= need_ids,
        })
    return rows


def grad_accum_for(micro_batch: int) -> int:
    """Derived, never chosen: `grad_accum = GLOBAL_BATCH_SEQUENCES // micro_batch`.

    Mirrors the reference, where the global batch is the recipe and the micro-batch only
    decides how it is split across memory. Refusing a micro-batch that does not divide the
    global batch keeps the effective batch EXACTLY 512 rather than silently near it.
    """
    if micro_batch <= 0 or GLOBAL_BATCH_SEQUENCES % micro_batch:
        raise ValueError(
            f"micro_batch={micro_batch} does not divide the global batch of "
            f"{GLOBAL_BATCH_SEQUENCES} sequences; the effective batch would not be the "
            f"recipe's, which the released code pairs with learning_rate=4e-4"
        )
    return GLOBAL_BATCH_SEQUENCES // micro_batch


def run_cell(
    cell: Cell,
    seed: int,
    micro_batch: int,
    grad_accum: int | None = None,
    device: str = "cuda",
    block_size: int = T.BLOCK_SIZE,
    eval_batch: int = 4,
    log_dir: Path | None = None,
) -> dict:
    """Train and evaluate one (cell, seed). The A6 contract is enforced, not assumed.

    Neither escape hatch is reachable from here: `TrainConfig` is constructed without
    `unseeded_order_ok`, and `evaluate_run` is called with a real `train_order`.

    `grad_accum` defaults to whatever holds the effective batch at the recipe's 512
    sequences; passing it explicitly is for tests only.
    """
    grad_accum = grad_accum_for(micro_batch) if grad_accum is None else grad_accum
    train_tokens = _load_split(cell.vocab_size, "train")
    eval_tokens = _load_split(cell.vocab_size, "selection_val")
    tokenizer = tk.load(TOKENIZERS / f"bpe_v{cell.vocab_size}.json")

    # Over the WHOLE array. Slicing to budget here would shrink the permutation domain and
    # silently break nesting between this cell's C and 1.1C arms.
    stream = T.TokenStream(np.asarray(train_tokens), block_size, order_seed=seed)

    cfg = T.TrainConfig(
        target_tokens=cell.target_tokens, micro_batch=micro_batch, block_size=block_size,
        grad_accum=grad_accum, seed=seed, device=device,
    )
    torch.manual_seed(seed)
    model = M.build(M.ModelConfig(vocab_size=cell.vocab_size, d=cell.d, n_layer=cell.n_layer,
                                  n_head=cell.n_head, d_ffn=cell.d_ffn,
                                  block_size=block_size))

    log_path = (log_dir / f"{cell.label}_s{seed}.csv") if log_dir else None
    res = T.train_run(model, stream, cfg, log_path=log_path)

    if res.consumed_tokens != stream.cursor * block_size:
        raise AssertionError(
            f"consumed tokens {res.consumed_tokens} disagree with sequences read "
            f"{stream.cursor} x {block_size}; the budget accounting is wrong"
        )

    ev = E.evaluate_run(
        model, np.asarray(eval_tokens), np.asarray(train_tokens), cell.vocab_size,
        tokenizer, consumed_train_tokens=res.consumed_tokens, block_size=block_size,
        batch=eval_batch, device=device, train_order=stream.consumed_order,
    )

    out = {
        "cell": cell.label, "vocab_size": cell.vocab_size, "arm": cell.arm, "seed": seed,
        "budget_flops": cell.budget_flops, "target_tokens": cell.target_tokens,
        "micro_batch": micro_batch, "grad_accum": grad_accum,
        **{f"train_{k}": v for k, v in asdict(res).items()
           if k not in ("loss_curve", "throughput_windows")},
        **{f"eval_{k}": v for k, v in ev.to_dict().items()},
    }
    return out


def audit_nesting(records: list[dict]) -> list[dict]:
    """Re-derive A6's audit rule from the recorded fields alone.

    Two runs sharing a `(vocabulary, seed)` are correctly nested iff they agree on
    `order_seed`, `stream_sequences`, `tokens_digest` and `order_digest`. Recomputed from
    the artifacts rather than trusted, because "the runner did it right" is exactly the kind
    of claim this project has learned not to accept without a check.
    """
    groups: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        groups.setdefault((r["vocab_size"], r["seed"]), []).append(r)

    findings = []
    for (v, seed), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue                      # only V_run has both arms
        keys = ("train_order_seed", "train_stream_sequences", "train_tokens_digest",
                "train_order_digest")
        agree = {k: len({r[k] for r in rs}) == 1 for k in keys}
        consumed = sorted(r["train_sequences_consumed"] for r in rs)
        findings.append({
            "vocab_size": v, "seed": seed, "arms": [r["arm"] for r in rs],
            **agree,
            "nested": all(agree.values()),
            "sequences_consumed": consumed,
            "strictly_increasing": consumed == sorted(set(consumed)) and len(
                set(consumed)) == len(consumed),
        })
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and check capacity without touching the GPU")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="memory knob only; grad_accum is derived to hold the effective "
                         "batch at the recipe's 512 sequences")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(RUNS))
    args = ap.parse_args()

    cells = pilot_cells()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'cell':>14} {'V':>7} {'arm':>5} {'target_tokens':>15} {'sequences':>11}")
    for c in cells:
        print(f"{c.label:>14} {c.vocab_size:>7} {c.arm:>5} {c.target_tokens:>15,} "
              f"{c.target_tokens // T.BLOCK_SIZE:>11,}")
    ga = grad_accum_for(args.micro_batch)
    print(f"\nrecipe: {GLOBAL_BATCH_SEQUENCES} sequences per optimizer step "
          f"(micro_batch={args.micro_batch} x grad_accum={ga})")
    steps = [c.target_tokens // T.BLOCK_SIZE // GLOBAL_BATCH_SEQUENCES for c in cells]
    print(f"optimizer steps per run: {min(steps)}-{max(steps)}  "
          f"(Tao at 33M nnv: 57-1144, median 601)")
    total = sum(c.target_tokens for c in cells)
    print(f"\nper seed: {total:,} tokens   x{len(PILOT_SEEDS)} seeds = "
          f"{total * len(PILOT_SEEDS):,}   ({len(cells) * len(PILOT_SEEDS)} runs)")

    cap = check_capacity(cells)
    (out_dir / "capacity.json").write_text(json.dumps(cap, indent=2), encoding="utf-8")
    bad = [r for r in cap if not r["ok"]]
    print(f"\ncapacity: {len(cap) - len(bad)}/{len(cap)} cells have enough tokens")
    for r in bad:
        print(f"  SHORT {r['cell']}: need {r['need_ids']:,}, have {r['have_ids']:,}")
    if bad:
        raise SystemExit("capacity check failed; not starting the pilot")

    if args.dry_run:
        print("\ndry run: planned and verified, nothing executed")
        return

    records = []
    results_path = out_dir / "results.jsonl"
    t0 = time.perf_counter()
    for cell in cells:
        for seed in PILOT_SEEDS:
            print(f"\n=== {cell.label} seed={seed} ===", flush=True)
            rec = run_cell(cell, seed, args.micro_batch,
                           device=args.device, log_dir=out_dir)
            records.append(rec)
            with open(results_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"    L_u={rec['eval_l_u']:+.6f}  BPB={rec['eval_bpb']:.6f}  "
                  f"{rec['train_seconds']:.0f}s", flush=True)

    findings = audit_nesting(records)
    (out_dir / "nesting_audit.json").write_text(json.dumps(findings, indent=2),
                                                encoding="utf-8")
    broken = [f for f in findings if not f["nested"]]
    print(f"\nnesting audit: {len(findings) - len(broken)}/{len(findings)} groups nested")
    if broken:
        raise SystemExit(f"NESTING BROKEN in {len(broken)} group(s); M2 pairing is invalid")
    print(f"\n{len(records)} runs in {(time.perf_counter() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
