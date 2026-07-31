"""Training-loop tests.

Two things here are load-bearing for the study rather than for the code:

  1. the LR schedule is warmup-then-CONSTANT, matching `reference/tinyllama_pretrain.py`
     -- an assumed cosine would test their law under a different intervention;
  2. each run consumes the exact token budget, because that is what makes C exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import extension as ext, model as M, train as T  # noqa: E402


def tiny_model(vocab: int = 512, block: int = 64) -> M.Transformer:
    a = ext.arch_for(2_000_000)
    return M.build(M.ModelConfig(vocab_size=vocab, d=a.d, n_layer=2, n_head=a.n_head,
                                 d_ffn=a.d_ffn, block_size=block))


# --- the schedule ------------------------------------------------------------------


def test_schedule_is_warmup_then_constant_not_cosine() -> None:
    """Faithful to reference get_lr: no decay after warmup, ever."""
    warm, base = 100, 4e-4
    assert T.lr_at(0, warm, base) == 0.0
    assert T.lr_at(50, warm, base) == pytest.approx(base * 0.5)
    assert T.lr_at(100, warm, base) == pytest.approx(base)
    # constant from warmup onward -- a cosine would fall here
    for step in (100, 500, 5_000, 1_000_000):
        assert T.lr_at(step, warm, base) == pytest.approx(base)


def test_no_min_lr_floor_is_applied() -> None:
    """`min_lr = 4e-5` exists in their file and is referenced nowhere. Do not honour it."""
    assert T.lr_at(10_000, 100, 4e-4) == pytest.approx(4e-4)
    assert T.lr_at(10_000, 100, 4e-4) != pytest.approx(4e-5)


def test_warmup_is_eight_percent_of_the_run() -> None:
    cfg = T.TrainConfig(target_tokens=64 * 100 * 25, micro_batch=1, block_size=64)
    assert cfg.warmup_steps == pytest.approx(cfg.total_steps * (2000 / 25000), abs=1)


def test_recipe_constants_match_the_reference() -> None:
    assert (T.LEARNING_RATE, T.WEIGHT_DECAY, T.GRAD_CLIP) == (4e-4, 1e-1, 1.0)
    assert (T.BETA1, T.BETA2) == (0.9, 0.95)
    assert T.BLOCK_SIZE == 2048
    assert T.WARMUP_FRACTION == pytest.approx(0.08)


# --- token budget ------------------------------------------------------------------


def test_step_count_covers_the_token_budget() -> None:
    cfg = T.TrainConfig(target_tokens=10_000, micro_batch=2, block_size=64, grad_accum=3)
    assert cfg.tokens_per_step == 2 * 64 * 3
    assert cfg.total_steps * cfg.tokens_per_step >= cfg.target_tokens
    assert (cfg.total_steps - 1) * cfg.tokens_per_step < cfg.target_tokens


def test_run_consumes_the_expected_tokens() -> None:
    block, mb = 64, 2
    cfg = T.TrainConfig(target_tokens=block * mb * 8, micro_batch=mb, block_size=block,
                        device="cpu", log_every_s=1e9)
    toks = np.random.randint(0, 512, size=cfg.total_steps * cfg.tokens_per_step + 64,
                             dtype=np.uint16)
    res = T.train_run(tiny_model(block=block), T.TokenStream(toks, block), cfg)
    assert res.consumed_tokens == cfg.total_steps * cfg.tokens_per_step
    assert res.consumed_tokens >= cfg.target_tokens
    assert res.steps == cfg.total_steps


# --- the stream --------------------------------------------------------------------


def test_stream_is_sequential_and_non_repeating() -> None:
    toks = np.arange(1000, dtype=np.uint16)
    s = T.TokenStream(toks, block_size=10)
    a, _ = s.next_batch(2, torch.device("cpu"))
    b, _ = s.next_batch(2, torch.device("cpu"))
    assert a.flatten().tolist() == list(range(0, 20))
    assert b.flatten().tolist() == list(range(20, 40))


def test_targets_are_inputs_shifted_by_one() -> None:
    toks = np.arange(500, dtype=np.uint16)
    x, y = T.TokenStream(toks, block_size=8).next_batch(2, torch.device("cpu"))
    assert y.flatten().tolist() == [v + 1 for v in x.flatten().tolist()]


def test_exhausted_stream_raises_a_diagnosable_error() -> None:
    s = T.TokenStream(np.arange(50, dtype=np.uint16), block_size=10)
    s.next_batch(4, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="token stream exhausted"):
        s.next_batch(4, torch.device("cpu"))


# --- learning actually happens -----------------------------------------------------

def test_loss_decreases_on_a_learnable_pattern() -> None:
    """A repeating pattern must be learned; a flat curve means the loop is broken."""
    block, mb, V = 32, 4, 64
    period = np.arange(V, dtype=np.uint16)
    toks = np.tile(period, 4000)
    cfg = T.TrainConfig(target_tokens=block * mb * 60, micro_batch=mb, block_size=block,
                        device="cpu", log_every_s=0.0, seed=0)
    res = T.train_run(tiny_model(vocab=V, block=block), T.TokenStream(toks, block), cfg)
    losses = [l for _, l in res.loss_curve]
    assert len(losses) >= 5
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < 2.0, "should nearly solve a period-64 cycle"


def test_seed_controls_reproducibility() -> None:
    block, mb, V = 32, 2, 64
    toks = np.random.RandomState(0).randint(0, V, size=20_000).astype(np.uint16)
    def run(seed: int) -> float:
        cfg = T.TrainConfig(target_tokens=block * mb * 12, micro_batch=mb, block_size=block,
                            device="cpu", log_every_s=1e9, seed=seed)
        torch.manual_seed(seed)
        m = tiny_model(vocab=V, block=block)
        return T.train_run(m, T.TokenStream(toks, block), cfg).final_train_loss
    assert run(1) == pytest.approx(run(1), rel=1e-6)


# --- evaluation --------------------------------------------------------------------


def test_evaluate_returns_a_sum_not_a_mean() -> None:
    """L_u and BPB need totals; a pre-averaged value cannot be re-divided by bytes."""
    block, V = 32, 64
    toks = np.random.RandomState(1).randint(0, V, size=block * 8 + 1).astype(np.uint16)
    m = tiny_model(vocab=V, block=block)
    total, scored = T.evaluate_nll(m, toks, block, batch=2, device="cpu")
    assert scored > 0
    per_token = total / scored
    # untrained model sits near ln(V)
    assert abs(per_token - np.log(V)) < 0.6
    assert total > per_token          # a sum, not a mean
