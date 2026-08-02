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


def test_warmup_is_ten_percent_from_the_run_script_not_the_module_defaults() -> None:
    """run.sh passes 5480/54800 = 10%; the 2000/25000 defaults were never the experiment."""
    assert T.WARMUP_FRACTION == pytest.approx(0.10)
    cfg = T.TrainConfig(target_tokens=64 * 100 * 25, micro_batch=1, block_size=64)
    assert cfg.warmup_steps == pytest.approx(cfg.total_steps * 0.10, abs=1)


def test_recipe_constants_match_the_reference() -> None:
    assert (T.LEARNING_RATE, T.WEIGHT_DECAY, T.GRAD_CLIP) == (4e-4, 1e-1, 1.0)
    assert (T.BETA1, T.BETA2) == (0.9, 0.95)
    assert T.BLOCK_SIZE == 2048
    assert T.WARMUP_FRACTION == pytest.approx(0.10)


# --- token budget ------------------------------------------------------------------


def test_budget_is_never_exceeded_and_shortfall_is_sub_sequence() -> None:
    """IsoFLOP is enforced on exact tokens, so the budget must never be rounded UP."""
    cfg = T.TrainConfig(target_tokens=10_000, micro_batch=2, block_size=64, grad_accum=3)
    assert cfg.tokens_per_step == 2 * 64 * 3
    assert cfg.target_sequences == 10_000 // 64            # 156 sequences = 9,984 tokens
    planned = cfg.target_sequences * cfg.block_size
    assert planned <= cfg.target_tokens
    assert cfg.target_tokens - planned < cfg.block_size    # shortfall under one sequence


def test_sub_block_budget_is_refused_rather_than_rounded_up() -> None:
    """The one case where the old floor could EXCEED the target it exists to bound."""
    cfg = T.TrainConfig(target_tokens=100, micro_batch=1, block_size=2048)
    with pytest.raises(ValueError, match="under one sequence"):
        _ = cfg.target_sequences


def test_stream_needs_one_id_beyond_the_last_block_for_its_final_target() -> None:
    """n*B ids yield n-1 sequences, not n: the last target is a lookahead token."""
    block = 8
    assert T.TokenStream(np.zeros(block * 4, dtype=np.uint16), block).n_sequences == 3
    assert T.TokenStream(np.zeros(block * 4 + 1, dtype=np.uint16), block).n_sequences == 4


def test_run_consumes_exactly_the_planned_tokens() -> None:
    block, mb = 64, 2
    # Deliberately NOT a whole multiple of a step, so the final step must be trimmed.
    cfg = T.TrainConfig(target_tokens=block * 17 + 30, micro_batch=mb, block_size=block,
                        grad_accum=3, device="cpu", log_every_s=1e9,
                        unseeded_order_ok=True)
    toks = np.random.randint(0, 512, size=block * 40 + 1, dtype=np.uint16)
    res = T.train_run(tiny_model(block=block), T.TokenStream(toks, block), cfg)
    assert res.consumed_tokens == cfg.target_sequences * block
    assert res.consumed_tokens <= cfg.target_tokens
    assert cfg.target_tokens - res.consumed_tokens < block


def test_sequences_are_self_contained_so_reordering_invents_no_targets() -> None:
    """Permuting order must never create a next-token pair absent from the corpus."""
    block = 8
    toks = np.arange(block * 6 + 1, dtype=np.uint16)
    dev = torch.device("cpu")
    plain = T.TokenStream(toks, block)
    shuf = T.TokenStream(toks, block, order_seed=7)
    assert not np.array_equal(plain.order, shuf.order)      # the order really did change
    for stream in (plain, shuf):
        for _ in range(3):
            x, y = stream.next_batch(2, dev)
            # Every (input, target) pair must be genuinely adjacent in the source array.
            assert torch.equal(y[:, :-1], x[:, 1:])
            for r in range(x.shape[0]):
                s = int(x[r, 0])
                assert torch.equal(x[r], torch.arange(s, s + block))
                assert torch.equal(y[r], torch.arange(s + 1, s + block + 1))


def test_permutation_is_nested_so_a_bigger_budget_extends_a_smaller_one() -> None:
    """A 1.1C run must read its C run's sequences plus more, in the same order.

    Comparing two whole streams would be tautological -- same array, same seed, same
    permutation. The property that can actually break is about CONSUMPTION: read a small
    budget and a large one from the same (array, seed) and the small one must be a strict
    prefix of the large one, sequence for sequence.
    """
    block, dev = 8, torch.device("cpu")
    toks = np.arange(block * 20 + 1, dtype=np.uint16)

    def read(n_seq: int) -> list[list[int]]:
        s = T.TokenStream(toks, block, order_seed=3)
        return [row.tolist() for _ in range(n_seq) for row in s.next_batch(1, dev)[0]]

    at_c, at_11c = read(6), read(9)
    assert at_11c[: len(at_c)] == at_c
    assert len(at_11c) > len(at_c)


def test_nesting_breaks_if_the_array_is_sliced_to_budget_and_the_result_records_it() -> None:
    """The realistic way to break M2 pairing, and the artifact field that catches it."""
    block = 8
    full = np.arange(block * 20 + 1, dtype=np.uint16)
    sliced = full[: block * 10 + 1]          # a caller "helpfully" trimming to budget
    wide = T.TokenStream(full, block, order_seed=3)
    narrow = T.TokenStream(sliced, block, order_seed=3)
    # Same seed, different permuted domain -> the prefixes diverge.
    assert wide.n_sequences != narrow.n_sequences
    assert not np.array_equal(wide.order[:5], narrow.order[:5])
    # Which is exactly what `stream_sequences` in TrainResult makes auditable.
    assert (wide.n_sequences, narrow.n_sequences) == (20, 10)


def test_confirmatory_run_refuses_a_stream_whose_order_is_not_tied_to_the_seed() -> None:
    """The original defect was an unenforced contract. It is now enforced."""
    block, mb = 8, 2
    cfg = T.TrainConfig(target_tokens=block * mb * 3, micro_batch=mb, block_size=block,
                        device="cpu", log_every_s=1e9, seed=5)
    toks = np.random.RandomState(0).randint(0, 64, size=block * 40 + 1).astype(np.uint16)
    with pytest.raises(ValueError, match="not tied to the seed"):
        T.train_run(tiny_model(vocab=64, block=block), T.TokenStream(toks, block), cfg)
    # A mismatched seed is refused too, not just a missing one.
    with pytest.raises(ValueError, match="not tied to the seed"):
        T.train_run(tiny_model(vocab=64, block=block),
                    T.TokenStream(toks, block, order_seed=4), cfg)
    # Correctly tied: runs, and records what it used.
    res = T.train_run(tiny_model(vocab=64, block=block),
                      T.TokenStream(toks, block, order_seed=5), cfg)
    assert (res.order_seed, res.stream_sequences) == (5, 40)


def test_trimmed_final_step_weights_micro_batches_by_size_not_count() -> None:
    """[4,4,4,1]: the 1-sequence batch must get 1/13 of the gradient, not 1/4.

    Checked against an explicit full-batch reference: accumulating the trimmed step must
    give the same gradient as one backward pass over all its sequences at once.
    """
    block, V = 8, 32
    torch.manual_seed(0)
    m = tiny_model(vocab=V, block=block)
    toks = np.random.RandomState(1).randint(0, V, size=block * 40 + 1).astype(np.uint16)
    dev = torch.device("cpu")

    def grads_from(micro: list[int]) -> torch.Tensor:
        m.zero_grad(set_to_none=True)
        s = T.TokenStream(toks, block)
        n = sum(micro)
        for b in micro:
            x, y = s.next_batch(b, dev)
            (m.loss(x, y) * (b / n)).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

    def grads_whole(n: int) -> torch.Tensor:
        m.zero_grad(set_to_none=True)
        s = T.TokenStream(toks, block)
        x, y = s.next_batch(n, dev)
        m.loss(x, y).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

    accumulated, whole = grads_from([4, 4, 4, 1]), grads_whole(13)
    assert torch.allclose(accumulated, whole, atol=1e-5), "trimmed step is mis-weighted"


def test_run_records_a_witness_that_distinguishes_equal_length_corpora() -> None:
    """stream_sequences alone cannot tell two different arrays of the same length apart."""
    block = 8
    a = np.arange(block * 20 + 1, dtype=np.uint16)
    b = (np.arange(block * 20 + 1, dtype=np.uint16) + 7) % 500
    sa, sb = T.TokenStream(a, block, order_seed=2), T.TokenStream(b, block, order_seed=2)
    assert sa.n_sequences == sb.n_sequences        # the weak witness cannot separate them
    assert sa.tokens_digest != sb.tokens_digest    # the strong one can


def test_tokens_digest_catches_a_single_changed_token_anywhere() -> None:
    """A strided sample could miss this; a full content digest cannot."""
    block, n = 8, 4000
    base = np.zeros(block * n + 1, dtype=np.uint16)
    d0 = T.TokenStream(base, block).tokens_digest
    for pos in (0, 1, 17, len(base) // 2, len(base) - 2, len(base) - 1):
        probe = base.copy()
        probe[pos] = 1
        assert T.TokenStream(probe, block).tokens_digest != d0, f"missed a change at {pos}"


def test_nesting_is_decidable_between_runs_of_different_lengths() -> None:
    """The audit rule, which a consumed-prefix digest could not support.

    A C run and a 1.1C run consume different amounts by construction, so their consumed
    prefixes always differ. Nesting has to be decidable anyway, from the whole-permutation
    digest plus how far each read.
    """
    block, dev = 8, torch.device("cpu")
    toks = np.arange(block * 30 + 1, dtype=np.uint16)

    short = T.TokenStream(toks, block, order_seed=11)
    long = T.TokenStream(toks, block, order_seed=11)
    short.next_batch(6, dev)
    long.next_batch(6, dev)
    long.next_batch(3, dev)                        # the 1.1C arm reads further

    # Consumed-prefix digests differ despite perfect nesting -- the rejected witness.
    assert short.order_digest(short.cursor) != long.order_digest(long.cursor)
    # The audit rule still decides it correctly.
    assert short.tokens_digest == long.tokens_digest
    assert short.order_digest() == long.order_digest()
    assert short.cursor < long.cursor
    # And a genuinely different permutation is rejected by the same rule.
    other = T.TokenStream(toks, block, order_seed=12)
    assert other.order_digest() != short.order_digest()


def test_seeds_differing_only_in_order_produce_different_runs() -> None:
    """If order were still fixed, these would be identical and the fix would be a no-op."""
    block, mb, V = 16, 2, 64
    toks = np.random.RandomState(3).randint(0, V, size=block * 200 + 1).astype(np.uint16)

    def run(order_seed: int) -> float:
        cfg = T.TrainConfig(target_tokens=block * mb * 20, micro_batch=mb, block_size=block,
                            device="cpu", log_every_s=1e9, seed=order_seed)
        torch.manual_seed(0)                      # SAME init, so only order can differ
        m = tiny_model(vocab=V, block=block)
        return T.train_run(m, T.TokenStream(toks, block, order_seed=order_seed), cfg
                           ).final_train_loss

    assert run(1) != pytest.approx(run(2), rel=1e-9)


def test_order_seed_none_reproduces_corpus_order() -> None:
    stream = T.TokenStream(np.arange(8 * 5 + 1, dtype=np.uint16), 8)
    assert np.array_equal(stream.order, np.arange(5))


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
                        device="cpu", log_every_s=0.0, seed=0, unseeded_order_ok=True)
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
        return T.train_run(m, T.TokenStream(toks, block, order_seed=seed), cfg
                           ).final_train_loss
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
