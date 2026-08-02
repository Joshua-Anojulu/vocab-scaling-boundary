"""Training loop, matching Tao et al.'s recipe.

Hyperparameters transcribed from `reference/tinyllama_pretrain.py`, not chosen:

    learning_rate  4e-4        weight_decay 1e-1        betas (0.9, 0.95)
    grad_clip      1.0         block_size   2048        global_batch 512 sequences
    warmup         10% of the run  -- NOT transcribed; see WARMUP_FRACTION

Warmup is the one line above that is not a transcription. The module defaults are
`warmup_steps=2000, max_step=25000` (8%); the upstream experiment script passes
`5480/54800` (10%); and **neither is provably what produced the released data.** `PLAN.md`
did not fix it either. It is a study choice, recorded in amendment A7, and this header says
so rather than presenting it alongside the values that really were transcribed.

**The schedule is linear warmup then CONSTANT.** Their `get_lr` returns `learning_rate`
unchanged after warmup; `min_lr = 4e-5` is defined in the file and referenced nowhere,
and `lr_decay_iters` is accepted and ignored. There is no cosine decay. Getting this
wrong would mean testing their law under a different training intervention than the one
that produced it.

**IsoFLOP is enforced on TOKENS, not characters.** Each run consumes exactly
`T_target = C / (6·(N_nv + V·d))` tokens, which makes `C` exact by construction. Tao's
fitted fertility `f(V)` is NOT used to decide how much data to read -- a newly trained
tokenizer's realized fertility will differ, and that divergence is a reported result
rather than a silent budget error.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from . import model as M

# --- Tao's recipe, pinned -----------------------------------------------------------

LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0
BLOCK_SIZE = 2048
WARMUP_FRACTION = 5480 / 54800
"""10%, from the upstream experiment script rather than the module defaults.

`tinyllama.py` defaults to `warmup_steps=2000, max_step=25000` (8%), and this constant was
originally derived from those. The upstream single-node experiment script
`experiments/light_train/scripts/run.sh` instead passes `warmup_steps=5480, max_step=54800`
-- exactly **10%**.

**Provenance is NOT established, and an earlier version of this docstring overclaimed it.**
It said run.sh "actually produced the released IsoFLOP data". The repository does not show
that: `exp_data.csv` predates the script in git history, and the script loops over
`vocab=4096` only. So neither figure is proven to be what generated the released data. 8% is
a module default that may never have been passed to anything; 10% is the only warmup ratio
the project is on record as actually passing. 10% is therefore chosen as the
better-evidenced of two weak options, and the weakness is recorded rather than hidden --
this study cannot claim to have matched their warmup, only to have matched the one value
they published a script for.

Expressed as a fraction because the ratio is what carries over: `max_step` differs per
configuration, so a fixed step count would mean something different in every run.
"""


def lr_at(step: int, warmup_steps: int, base_lr: float = LEARNING_RATE) -> float:
    """Linear warmup, then CONSTANT. Faithful to reference `get_lr`.

    Note the consequence: because the rate never decays, an intermediate checkpoint has
    exactly the learning-rate history of a run trained to that step. Checkpoint reuse for
    lower budgets is therefore valid under THIS schedule, unlike under cosine -- but the
    preregistration currently specifies separate budget-specific runs, so any change is a
    documented amendment, not an optimisation applied here.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


@dataclass
class TrainConfig:
    target_tokens: int
    micro_batch: int
    block_size: int = BLOCK_SIZE
    grad_accum: int = 1
    seed: int = 0
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    betas: tuple[float, float] = (BETA1, BETA2)
    grad_clip: float = GRAD_CLIP
    warmup_fraction: float = WARMUP_FRACTION
    chunk_size: int = 4096
    log_every_s: float = 30.0
    device: str = "cuda"
    unseeded_order_ok: bool = False
    """Declare that this run is NOT a confirmatory run and may read corpus order.

    Confirmatory runs must vary data order with the seed; see the seed-semantics
    amendment. That was previously true only if the caller remembered to pass
    `order_seed`, which is the same failure mode as the defect it fixes -- the original
    bug was a contract that lived in prose while the code satisfied it by accident.
    `train_run` therefore REQUIRES `stream.order_seed == cfg.seed` unless this is set,
    so reading corpus order becomes a declaration rather than an omission.
    """

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.block_size * self.grad_accum

    @property
    def sequences_per_step(self) -> int:
        return self.micro_batch * self.grad_accum

    @property
    def target_sequences(self) -> int:
        """Whole sequences that fit in the budget, never exceeding it.

        IsoFLOP is enforced on exact token counts, so the budget must not be rounded UP.
        One sequence (`block_size` tokens) is the finest granularity available, so the
        convention is the largest whole number of sequences that does not exceed
        `target_tokens`; the shortfall is under one sequence and is reported as
        `consumed_tokens` rather than assumed away.

        An earlier version ceiled the STEP count instead and never trimmed, despite a
        docstring claiming it did, overshooting by up to `tokens_per_step - 1` tokens --
        a V-dependent perturbation of `C` of the same character as the EOS effect that A4
        treats as decision-relevant. Measured on the Stage B.7 pilot configurations, that
        was +0.0129% at micro_batch=4/grad_accum=4 and +0.0845% at 8/8, against a worst
        case here of -0.0011%. Those figures are regenerated by
        `scripts/budget_convention_error.py` into `results/budget_convention_error.json`
        rather than quoted, because three earlier numbers in this project were quoted from
        rounded output and had to be corrected.

        A budget below one sequence is rejected rather than rounded up to one. The old
        `max(1, ...)` floor was the single case where this property could be violated: it
        returned one whole sequence for a sub-block budget, EXCEEDING the target, which is
        the one thing this convention exists to prevent. No study configuration is near
        that boundary, so this only ever fired in toy configurations -- where silently
        overshooting is worse than refusing.
        """
        if self.target_tokens < self.block_size:
            raise ValueError(
                f"budget of {self.target_tokens} tokens is under one sequence of "
                f"{self.block_size}; rounding up would exceed the IsoFLOP target"
            )
        return self.target_tokens // self.block_size

    @property
    def total_steps(self) -> int:
        """Steps needed to consume the budget; the final step is trimmed to fit."""
        return max(1, math.ceil(self.target_sequences / self.sequences_per_step))

    @property
    def warmup_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.warmup_fraction)))


@dataclass
class TrainResult:
    target_tokens: int
    consumed_tokens: int
    steps: int
    warmup_steps: int
    seconds: float
    tokens_per_sec: float
    final_train_loss: float
    loss_curve: list[tuple[int, float]] = field(default_factory=list)
    throughput_windows: list[float] = field(default_factory=list)
    peak_mem_gb: float = 0.0
    seed: int = 0
    order_seed: int | None = None
    stream_sequences: int = 0
    sequences_consumed: int = 0
    tokens_digest: str = ""
    order_digest: str = ""
    consumed_order_digest: str = ""
    numpy_version: str = ""
    """Witness fields that make nesting auditable from the artifacts alone.

    **The audit rule:** two runs sharing a `(vocabulary, seed)` are correctly nested iff
    they agree on `order_seed`, `stream_sequences`, `tokens_digest` and `order_digest` --
    identical arrays and identical permutations -- with `sequences_consumed` recording how
    far each read. `order_digest` is over the WHOLE permutation, not the consumed prefix,
    because runs at `C` and `1.1*C` consume different amounts and prefix digests would
    differ even when nesting is perfect. `consumed_order_digest` is retained as a
    descriptive record of what each run actually read, not as the nesting witness.

    Two earlier versions of this witness were rejected in review. The first recorded only
    `stream_sequences`, which cannot distinguish a different token array of the same
    length, a different same-length slice, or a changed permutation algorithm. The second
    hashed a strided SAMPLE of the token array for large arrays -- a sample can miss the
    localised difference the audit exists to catch.

    `numpy_version` is recorded because `np.random.default_rng(seed).permutation(n)` is
    **not promised stable across NumPy versions**. Nothing in this design requires
    cross-version reproducibility -- the permutation only has to be fixed within a
    `(vocabulary, seed)` group, and all runs in a group are produced by one process --
    but a run repeated after an upgrade may read a different order, and `order_digest`
    is what turns that from a silent difference into a visible one.
    """


class TokenStream:
    """Non-repeating view over a flat token array, in units of self-contained sequences.

    **Sequences are self-contained.** Sequence `k` spans `tokens[k*B : k*B + B + 1]` -- `B`
    inputs plus the ONE lookahead token that supplies the final target. This mirrors the
    reference, which requests `effective_block_size = block_size + 1` for exactly this
    reason. It matters because it is what makes reordering safe: a sequence carries its own
    targets, so permuting sequence ORDER never manufactures a next-token pair that does not
    occur in the corpus. Concatenating permuted blocks and shifting across the join would
    invent one false target per block -- about 0.049% of targets at B=2048, the same order
    as the EOS effects that amendment A4 treats as decision-relevant.

    **What a seed varies.** With `order_seed=None` the order is the corpus order. With an
    integer, sequence order is permuted by that seed while the sequence CONTENTS are
    untouched. Seeds then vary initialisation and data order together, which is the
    variance component the seed-level BCa bootstrap is supposed to be estimating.

    **Nesting is preserved**, which is what lets budgets stay comparable. The permutation is
    drawn over the WHOLE array once per (vocabulary, seed) and every budget reads a prefix
    of it, so a `1.1*C` run reads its `C` run's sequences plus more, in the same order --
    the preregistered "same tokens in the same order, differing only in how far they read",
    now holding per seed rather than globally.
    """

    def __init__(
        self, tokens: np.ndarray, block_size: int, order_seed: int | None = None
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a flat array")
        self.tokens = tokens
        self.block = block_size
        # -1 so the last sequence still has its lookahead token.
        self.n_sequences = (len(tokens) - 1) // block_size
        if self.n_sequences == 0:
            raise ValueError(
                f"array of {len(tokens)} tokens holds no complete sequence of {block_size}+1"
            )
        self.order_seed = order_seed
        # FULL content digest, streamed in chunks. An earlier version hashed a strided
        # SAMPLE for large arrays, which was rejected in review: a sample can miss exactly
        # the localised difference an audit is for, and a witness that can miss what it
        # exists to detect is not a witness. Streaming keeps peak memory flat instead of
        # materialising ~360 MB via .tobytes(); the cost is under a second against a
        # multi-hour run.
        h = hashlib.sha256()
        step = 1 << 24
        for s in range(0, len(tokens), step):
            h.update(np.ascontiguousarray(tokens[s : s + step]).tobytes())
        h.update(str(len(tokens)).encode())
        self.tokens_digest = h.hexdigest()[:16]
        if order_seed is None:
            self.order = np.arange(self.n_sequences, dtype=np.int64)
        else:
            self.order = np.random.default_rng(order_seed).permutation(self.n_sequences)
        self.cursor = 0

    @property
    def available(self) -> int:
        """Sequences not yet consumed."""
        return self.n_sequences - self.cursor

    @property
    def consumed_order(self) -> np.ndarray:
        """Sequence indices this run actually read, in consumption order.

        The unigram baseline in `L_u` must be fitted on THIS set. Under permutation the
        consumed set is scattered, not a prefix, so fitting on `train_tokens[:consumed]`
        would fit on text the model never read -- and on a set identical across seeds while
        the model's set is not, silently removing a real component of seed variance from
        the primary metric.
        """
        return self.order[: self.cursor]

    def order_digest(self, n: int | None = None) -> str:
        """Hash of the permutation. `n=None` means the WHOLE order, not the consumed part.

        The default is the whole order because that is what makes nesting checkable across
        runs of DIFFERENT lengths. A digest of each run's consumed prefix cannot do it: a
        `C` run and a `1.1*C` run necessarily consume different amounts, so their prefix
        digests differ even when perfectly nested, and the audit would have no way to tell
        correct nesting from a broken permutation.

        Over the whole order it is decidable from the artifacts alone. Two runs are nested
        iff they agree on `tokens_digest` and this digest -- identical arrays and identical
        permutations -- after which one having read further is the entire difference between
        them, which `sequences_consumed` records.
        """
        order = self.order if n is None else self.order[:n]
        return hashlib.sha256(
            np.ascontiguousarray(order, dtype=np.int64).tobytes()
        ).hexdigest()[:16]

    def next_batch(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.available < batch:
            raise RuntimeError(
                f"token stream exhausted: need {batch} sequences, have {self.available}. "
                f"The corpus slice is too small for the requested budget."
            )
        idx = self.order[self.cursor : self.cursor + batch]
        B = self.block
        xs = np.empty((batch, B), dtype=np.int64)
        ys = np.empty((batch, B), dtype=np.int64)
        for i, k in enumerate(idx):
            s = int(k) * B
            seq = np.asarray(self.tokens[s : s + B + 1], dtype=np.int64)
            xs[i] = seq[:-1]
            ys[i] = seq[1:]
        self.cursor += batch
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_run(
    model: M.Transformer,
    stream: TokenStream,
    cfg: TrainConfig,
    log_path: str | Path | None = None,
) -> TrainResult:
    if not cfg.unseeded_order_ok and stream.order_seed != cfg.seed:
        raise ValueError(
            f"data order is not tied to the seed: stream.order_seed={stream.order_seed}, "
            f"cfg.seed={cfg.seed}. A confirmatory run must vary initialisation AND data "
            f"order together, or the seed-level bootstrap estimates the wrong variance "
            f"component. Pass TokenStream(..., order_seed=cfg.seed), or set "
            f"unseeded_order_ok=True to declare this run non-confirmatory."
        )
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = model.to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    total_steps, warmup = cfg.total_steps, cfg.warmup_steps
    remaining_seq = cfg.target_sequences
    consumed = 0
    curve: list[tuple[int, float]] = []
    windows: list[float] = []
    t0 = time.perf_counter()
    t_win, win_tokens = t0, 0
    last_loss = float("nan")
    steps_taken = 0

    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, warmup, cfg.lr)

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        # The final step is trimmed to whatever is left of the budget, so the number of
        # micro-batches and their size are decided here rather than fixed. Gradients are
        # scaled by the micro-batches ACTUALLY taken, not by cfg.grad_accum, or a short
        # final step would silently carry less weight than a full one.
        micro = []
        while len(micro) < cfg.grad_accum and remaining_seq > 0:
            micro.append(min(cfg.micro_batch, remaining_seq))
            remaining_seq -= micro[-1]
        if not micro:
            break
        # Weight each micro-batch by its SHARE OF SEQUENCES, not by 1/len(micro).
        # `model.loss` returns a mean over the micro-batch, so equal weights are only
        # correct when the micro-batches are equal size. On a trimmed final step like
        # [4, 4, 4, 1] the one-sequence batch would get 1/4 of the gradient instead of
        # 1/13, over-weighting its tokens more than threefold in the step that ends the run.
        n_seq_step = sum(micro)
        for b in micro:
            x, y = stream.next_batch(b, dev)
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.loss(x, y, chunk_size=cfg.chunk_size)
            else:
                loss = model.loss(x, y, chunk_size=cfg.chunk_size)
            w = b / n_seq_step
            (loss * w).backward()
            step_loss += loss.item() * w
            consumed += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = step_loss
        steps_taken = step + 1
        # Tokens ACTUALLY processed. Charging `cfg.tokens_per_step` would credit a trimmed
        # final step at full width and inflate the last throughput window -- and these
        # windows are what the Stage B runtime projections are built from.
        win_tokens += sum(micro) * cfg.block_size

        now = time.perf_counter()
        if now - t_win >= cfg.log_every_s or step == total_steps - 1:
            tps = win_tokens / (now - t_win)
            windows.append(tps)
            curve.append((step, step_loss))
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{now - t0:.1f},{step_loss:.6f},{tps:.1f},"
                             f"{lr_at(step, warmup, cfg.lr):.3e}\n")
            t_win, win_tokens = now, 0

    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0

    return TrainResult(
        target_tokens=cfg.target_tokens,
        consumed_tokens=consumed,
        steps=steps_taken,
        warmup_steps=warmup,
        seconds=elapsed,
        tokens_per_sec=consumed / elapsed if elapsed else 0.0,
        final_train_loss=last_loss,
        loss_curve=curve,
        throughput_windows=windows,
        peak_mem_gb=peak,
        seed=cfg.seed,
        order_seed=stream.order_seed,
        stream_sequences=stream.n_sequences,
        sequences_consumed=stream.cursor,
        tokens_digest=stream.tokens_digest,
        order_digest=stream.order_digest(),
        consumed_order_digest=stream.order_digest(stream.cursor),
        numpy_version=np.__version__,
    )


@torch.no_grad()
def evaluate_nll(
    model: M.Transformer,
    tokens: np.ndarray,
    block_size: int,
    batch: int = 4,
    chunk_size: int = 4096,
    device: str = "cuda",
) -> tuple[float, int]:
    """Total NLL in nats over a token array, plus the number of scored positions.

    Returns the SUM, not a mean: BPB divides by an exact byte count and `L_u` divides by
    the token count, so both need the total rather than a pre-averaged figure.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    n_seq = (len(tokens) - 1) // block_size
    if n_seq == 0:
        raise ValueError("evaluation slice shorter than one block")

    total_nll = 0.0
    scored = 0
    for i in range(0, n_seq, batch):
        b = min(batch, n_seq - i)
        need = b * block_size + 1
        buf = tokens[i * block_size : i * block_size + need]
        if len(buf) < need:
            break
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(b, block_size).to(dev)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(b, block_size).to(dev)
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model.loss(x, y, chunk_size=chunk_size, reduction="sum")
        else:
            s = model.loss(x, y, chunk_size=chunk_size, reduction="sum")
        total_nll += float(s)
        scored += b * block_size
    return total_nll, scored


def save_result(res: TrainResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
