"""Tests for the L_u evaluation harness.

The load-bearing test here is `test_unigram_and_model_score_identical_positions`. The whole
reason this module exists is that scoring the two metric terms over different token sets
produces a plausible wrong number rather than an error, so that property is asserted
directly rather than trusted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate as E  # noqa: E402
from src import model as M  # noqa: E402
from src import metrics as mx  # noqa: E402


# --- block plan -----------------------------------------------------------------------


def test_plan_rounds_down_to_whole_batches():
    # 10 sequences available, batch 4 -> only 8 are consumed by the model loop.
    plan = E.plan_blocks(n_tokens=10 * 8 + 1, block_size=8, batch=4)
    assert plan.n_seq == 8
    assert plan.n_scored == 64


def test_plan_keeps_partial_batch_when_fewer_sequences_than_one_batch():
    plan = E.plan_blocks(n_tokens=3 * 8 + 1, block_size=8, batch=4)
    assert plan.n_seq == 3


def test_plan_rejects_slice_shorter_than_a_block():
    with pytest.raises(ValueError, match="shorter than one block"):
        E.plan_blocks(n_tokens=5, block_size=8, batch=1)


def test_dropped_tokens_are_reported_not_absorbed():
    # 100 tokens, block 8 -> 12 sequences, which is already a whole number of batches.
    plan = E.plan_blocks(n_tokens=100, block_size=8, batch=4)
    assert plan.n_seq == 12
    assert plan.n_scored == 96
    assert plan.n_scored + plan.n_dropped + 1 == plan.n_tokens
    # And a case where rounding to whole batches really does drop a tail.
    p2 = E.plan_blocks(n_tokens=8 * 14 + 1, block_size=8, batch=4)
    assert p2.n_seq == 12 and p2.n_dropped == 8 * 2


# --- the property the module exists to guarantee --------------------------------------


def test_unigram_and_model_score_identical_positions():
    """Both terms must cover exactly the same targets, or L_u is silently wrong."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 16, size=101, dtype=np.uint16)
    plan = E.plan_blocks(len(toks), block_size=8, batch=4)
    targets = E.scored_targets(toks, plan)

    assert len(targets) == plan.n_scored
    # Positionally identical to what the model loop consumes as `y`.
    model_side = []
    for i in range(0, plan.n_seq, plan.batch):
        b = min(plan.batch, plan.n_seq - i)
        need = b * plan.block_size + 1
        buf = toks[i * plan.block_size : i * plan.block_size + need]
        model_side.append(np.asarray(buf[1:], dtype=np.int64))
    assert np.array_equal(np.concatenate(model_side), targets)


def test_scored_targets_excludes_first_token_only():
    toks = np.arange(1 + 8 * 4, dtype=np.uint16)
    plan = E.plan_blocks(len(toks), block_size=8, batch=4)
    targets = E.scored_targets(toks, plan)
    assert targets[0] == toks[1]
    assert targets[-1] == toks[plan.n_scored]


# --- unigram --------------------------------------------------------------------------


def test_unigram_is_a_normalised_distribution():
    toks = np.array([0, 1, 1, 2, 2, 2], dtype=np.uint16)
    logp = E.unigram_logp(toks, vocab_size=4)
    assert math.isclose(float(np.exp(logp).sum()), 1.0, rel_tol=1e-12)


def test_unigram_matches_metrics_reference_implementation():
    toks = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.uint16)
    logp = E.unigram_logp(toks, vocab_size=4, alpha=1.0)
    ref = mx.fit_unigram([toks.tolist()], vocab_size=4, alpha=1.0)
    for tid in range(4):
        assert math.isclose(float(logp[tid]), ref.logp[tid], rel_tol=1e-12)


def test_unigram_fits_only_the_consumed_prefix():
    """Fitting on more text than the run saw would give the baseline unearned information."""
    toks = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint16)
    full = E.unigram_logp(toks, vocab_size=2)
    half = E.unigram_logp(toks, vocab_size=2, consumed=4)
    assert math.isclose(float(np.exp(full[0])), 0.5, rel_tol=1e-12)
    # First half is all zeros, so token 0 must dominate under the prefix fit.
    assert float(np.exp(half[0])) > 0.8


def test_unigram_follows_the_permuted_order_not_the_prefix():
    """Under a seed permutation the consumed set is scattered, so a prefix fit is wrong.

    Built so the two answers cannot coincide: the array's first half is all zeros and its
    second half all ones, and the order picks sequences from the second half only. A prefix
    fit would report token 0 dominant; the correct fit reports token 1 dominant.
    """
    block = 4
    toks = np.array([0] * 12 + [1] * 13, dtype=np.uint16)
    order = np.array([3, 4, 5], dtype=np.int64)          # sequences at 12.., 16.., 20..
    consumed = 3 * block

    prefix_fit = E.unigram_logp(toks, vocab_size=2, consumed=consumed)
    permuted_fit = E.unigram_logp(toks, vocab_size=2, consumed=consumed,
                                  order=order, block_size=block)

    assert float(np.exp(prefix_fit[0])) > 0.8            # what the OLD code would report
    assert float(np.exp(permuted_fit[1])) > 0.8          # what the model actually read
    assert float(np.exp(permuted_fit[0])) < 0.2


def test_unigram_permuted_fit_requires_a_block_size():
    with pytest.raises(ValueError, match="block_size is required"):
        E.unigram_logp(np.zeros(8, dtype=np.uint16), vocab_size=2,
                       order=np.array([0, 1], dtype=np.int64))


def test_consumed_target_blocks_are_targets_not_inputs():
    """Sequence k spans [k*B, k*B+B+1); its TARGETS are [k*B+1, k*B+B+1)."""
    block = 4
    toks = np.arange(21, dtype=np.uint16)
    blocks = list(E.consumed_target_blocks(toks, block, np.array([0, 2]), 2))
    assert [b.tolist() for b in blocks] == [[1, 2, 3, 4], [9, 10, 11, 12]]


def test_evaluate_run_refuses_to_guess_the_training_order():
    """Silently defaulting to a prefix fit would rebuild the defect A6 exists to fix."""
    V, block = 8, 16
    rng = np.random.default_rng(0)
    ev = rng.integers(0, V, size=block * 4 + 1, dtype=np.uint16)
    train = rng.integers(0, V, size=block * 8 + 1, dtype=np.uint16)
    model = M.build(M.ModelConfig(vocab_size=V, d=32, n_layer=1, n_head=2, d_ffn=64,
                                  block_size=block))

    class _Tok:
        def decode(self, ids):
            return "x" * len(ids)

    with pytest.raises(ValueError, match="train_order is required"):
        E.evaluate_run(model, ev, train, V, _Tok(), block_size=block, batch=2, device="cpu")


def test_unigram_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="outside"):
        E.unigram_logp(np.array([0, 5], dtype=np.uint16), vocab_size=2)


def test_every_id_has_mass_so_zero_frequency_is_finite():
    """A3: zero-frequency eval tokens occur, so unsmoothed MLE would be undefined."""
    toks = np.array([0, 0, 0], dtype=np.uint16)
    logp = E.unigram_logp(toks, vocab_size=8)
    assert np.all(np.isfinite(logp))


# --- L_u sign and composition ---------------------------------------------------------


def _result(model_nll: float, uni_nll: float, n: int = 100, nb: int = 400) -> E.EvalResult:
    return E.EvalResult(
        n_scored=n, n_dropped=0, n_bytes=nb, model_nll_nats=model_nll,
        unigram_nll_nats=uni_nll, vocab_size=16, unigram_fit_tokens=1000, alpha=1.0,
    )


def test_l_u_is_negative_when_model_beats_unigram():
    """Sign convention is asserted, not assumed -- it is easy to invert by accident."""
    r = _result(model_nll=100.0, uni_nll=250.0)
    assert r.l_u < 0
    assert math.isclose(r.l_u, r.ce_model - r.h_unigram, rel_tol=1e-12)


def test_l_u_is_positive_when_model_loses_to_unigram():
    assert _result(model_nll=300.0, uni_nll=250.0).l_u > 0


def test_more_negative_l_u_is_better():
    better = _result(model_nll=80.0, uni_nll=250.0)
    worse = _result(model_nll=120.0, uni_nll=250.0)
    assert better.l_u < worse.l_u


def test_bpb_uses_the_byte_count_not_fertility():
    r = _result(model_nll=math.log(2.0) * 400, uni_nll=1.0, n=100, nb=400)
    assert math.isclose(r.bpb, 1.0, rel_tol=1e-12)


def test_l_u_agrees_with_metrics_module():
    r = _result(model_nll=100.0, uni_nll=250.0, n=100, nb=400)
    ev = mx.Evaluation(n_tokens=100, n_bytes=400, model_nll_nats=100.0, unigram_nll_nats=250.0)
    assert math.isclose(r.l_u, ev.l_u, rel_tol=1e-12)
    assert math.isclose(r.bpb, ev.bpb, rel_tol=1e-12)


# --- end to end against a real model --------------------------------------------------


def test_evaluate_run_matches_a_hand_computed_unigram_term():
    """The unigram term must equal -sum(logp[targets]) over exactly the scored ids."""
    torch.manual_seed(0)
    V = 32
    cfg = M.ModelConfig(vocab_size=V, d=64, n_layer=2, n_head=2, d_ffn=128,
                        block_size=16)
    model = M.Transformer(cfg)

    rng = np.random.default_rng(1)
    train = rng.integers(0, V, size=500, dtype=np.uint16)
    ev = rng.integers(0, V, size=16 * 4 + 1, dtype=np.uint16)

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    res = E.evaluate_run(model, ev, train, V, _Tok(), block_size=16, batch=2,
                         device="cpu", sequential_order_ok=True)

    plan = E.plan_blocks(len(ev), 16, 2)
    targets = E.scored_targets(ev, plan)
    logp = E.unigram_logp(train, V)
    assert math.isclose(res.unigram_nll_nats, float(-logp[targets].sum()), rel_tol=1e-10)
    assert res.n_scored == plan.n_scored
    assert res.n_bytes == plan.n_scored  # one "x" per scored token


def test_evaluate_run_model_term_matches_direct_computation():
    torch.manual_seed(0)
    V = 32
    cfg = M.ModelConfig(vocab_size=V, d=64, n_layer=2, n_head=2, d_ffn=128,
                        block_size=16)
    model = M.Transformer(cfg)
    rng = np.random.default_rng(2)
    train = rng.integers(0, V, size=200, dtype=np.uint16)
    ev = rng.integers(0, V, size=16 * 4 + 1, dtype=np.uint16)

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    res = E.evaluate_run(model, ev, train, V, _Tok(), block_size=16, batch=2,
                         device="cpu", sequential_order_ok=True)
    plan = E.plan_blocks(len(ev), 16, 2)
    direct = E.model_nll_nats(model, ev, plan, device="cpu")
    assert math.isclose(res.model_nll_nats, direct, rel_tol=1e-6)
