"""Training loop, matching Tao et al.'s recipe.

Hyperparameters transcribed from `reference/tinyllama_pretrain.py`, not chosen:

    learning_rate  4e-4        weight_decay 1e-1        betas (0.9, 0.95)
    grad_clip      1.0         block_size   2048        global_batch 512 sequences
    warmup         2000 of 25000 steps == 8% of training

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
WARMUP_FRACTION = 2000 / 25000       # 8%, expressed as a fraction so it scales with run length


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

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.block_size * self.grad_accum

    @property
    def total_steps(self) -> int:
        """Steps needed to consume the token budget. Ceil, then the last step is trimmed."""
        return max(1, math.ceil(self.target_tokens / self.tokens_per_step))

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


class TokenStream:
    """Sequential, non-repeating view over a flat token array.

    Sequences are packed contiguously with no shuffling: the plan holds the ordered
    source corpus fixed so that runs at different budgets see the same tokens in the same
    order, differing only in how far they read.
    """

    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a flat array")
        self.tokens = tokens
        self.block = block_size
        self.pos = 0

    @property
    def available(self) -> int:
        return len(self.tokens) - self.pos - 1        # -1 because targets are shifted

    def next_batch(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        need = batch * self.block + 1
        if self.available < batch * self.block:
            raise RuntimeError(
                f"token stream exhausted: need {need}, have {self.available + 1} left. "
                f"The corpus slice is too small for the requested budget."
            )
        buf = self.tokens[self.pos : self.pos + need]
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(batch, self.block)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(batch, self.block)
        self.pos += batch * self.block
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_run(
    model: M.Transformer,
    stream: TokenStream,
    cfg: TrainConfig,
    log_path: str | Path | None = None,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = model.to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    total_steps, warmup = cfg.total_steps, cfg.warmup_steps
    consumed = 0
    curve: list[tuple[int, float]] = []
    windows: list[float] = []
    t0 = time.perf_counter()
    t_win, win_tokens = t0, 0
    last_loss = float("nan")

    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, warmup, cfg.lr)

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = stream.next_batch(cfg.micro_batch, dev)
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.loss(x, y, chunk_size=cfg.chunk_size)
            else:
                loss = model.loss(x, y, chunk_size=cfg.chunk_size)
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum
            consumed += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = step_loss
        win_tokens += cfg.tokens_per_step

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
        steps=total_steps,
        warmup_steps=warmup,
        seconds=elapsed,
        tokens_per_sec=consumed / elapsed if elapsed else 0.0,
        final_train_loss=last_loss,
        loss_curve=curve,
        throughput_windows=windows,
        peak_mem_gb=peak,
        seed=cfg.seed,
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
