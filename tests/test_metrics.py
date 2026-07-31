"""Tests for L_u and BPB.

The sign convention on L_u is the one that would be easiest to invert and hardest to
notice: an inverted L_u would still produce smooth, plausible bowls -- with their minima
in the wrong place.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as mx  # noqa: E402


def stream(n: int, V: int, seed: int = 0, skew: bool = True) -> list[int]:
    rng = random.Random(seed)
    if not skew:
        return [rng.randrange(V) for _ in range(n)]
    # Zipf-ish: a few tokens dominate, like real text
    return [min(V - 1, int(rng.paretovariate(1.2)) - 1) for _ in range(n)]


# --- unigram fitting ---------------------------------------------------------------


def test_unigram_is_a_normalised_distribution() -> None:
    u = mx.fit_unigram([stream(5000, 100)], vocab_size=100)
    total = sum(math.exp(u.logp[t]) for t in range(100))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_every_vocabulary_id_has_mass_even_if_unseen() -> None:
    """Smoothing must be total: an unseen token in evaluation cannot be infinitely surprising."""
    u = mx.fit_unigram([[0, 1, 2]], vocab_size=50)
    assert u.n_unseen == 47
    for t in range(50):
        assert u.logp[t] > -math.inf
    assert u.nll_nats([49]) < math.inf


def test_out_of_vocabulary_id_raises_rather_than_silently_scoring() -> None:
    u = mx.fit_unigram([[0, 1]], vocab_size=10)
    with pytest.raises(KeyError):
        u.log_prob(10)


def test_more_frequent_tokens_get_higher_probability() -> None:
    u = mx.fit_unigram([[0] * 100 + [1] * 10 + [2]], vocab_size=5)
    assert u.logp[0] > u.logp[1] > u.logp[2] > u.logp[4]


# --- L_u sign and behaviour --------------------------------------------------------


def test_l_u_is_negative_when_the_model_beats_unigram() -> None:
    """The published rows are -1.35 / -2.10 / -2.50. More negative is better."""
    ids = stream(2000, 256)
    u = mx.fit_unigram([ids], vocab_size=256)
    h_uni = u.nll_nats(ids)
    ev = mx.evaluate(model_nll_nats=h_uni * 0.5, token_ids=ids, unigram=u, n_bytes=8000)
    assert ev.l_u < 0
    assert ev.l_u == pytest.approx(ev.ce_model - ev.h_unigram)


def test_l_u_is_zero_when_the_model_equals_unigram() -> None:
    ids = stream(1000, 128)
    u = mx.fit_unigram([ids], vocab_size=128)
    ev = mx.evaluate(u.nll_nats(ids), ids, u, n_bytes=4000)
    assert ev.l_u == pytest.approx(0.0, abs=1e-12)


def test_l_u_is_positive_when_the_model_is_worse_than_unigram() -> None:
    ids = stream(1000, 128)
    u = mx.fit_unigram([ids], vocab_size=128)
    ev = mx.evaluate(u.nll_nats(ids) * 1.5, ids, u, n_bytes=4000)
    assert ev.l_u > 0


def test_lower_l_u_means_a_better_model() -> None:
    """Monotonicity in the right direction -- an inverted sign would still look smooth."""
    ids = stream(2000, 256)
    u = mx.fit_unigram([ids], vocab_size=256)
    base = u.nll_nats(ids)
    better = mx.evaluate(base * 0.4, ids, u, 8000)
    worse = mx.evaluate(base * 0.8, ids, u, 8000)
    assert better.l_u < worse.l_u
    assert better.ce_model < worse.ce_model


# --- BPB ---------------------------------------------------------------------------


def test_bpb_uses_exact_bytes_not_tokens() -> None:
    ids = stream(1000, 256)
    u = mx.fit_unigram([ids], vocab_size=256)
    nll = 700.0
    ev = mx.evaluate(nll, ids, u, n_bytes=4000)
    assert ev.bpb == pytest.approx(nll / (4000 * math.log(2)))
    assert ev.bpb != pytest.approx(nll / (1000 * math.log(2)))


def test_bpb_is_invariant_to_tokenisation_at_equal_total_nll() -> None:
    """The whole point: same text, same total NLL, different segmentation -> same BPB.

    Per-token cross-entropy differs between the two; BPB does not. That is why BPB is the
    tokenizer-agnostic cross-check and raw per-token CE is never used for comparison.
    """
    nll, nbytes = 5000.0, 20_000
    coarse = stream(1000, 256, seed=1)
    fine = stream(2500, 256, seed=2)
    u_c = mx.fit_unigram([coarse], 256)
    u_f = mx.fit_unigram([fine], 256)
    a = mx.evaluate(nll, coarse, u_c, nbytes)
    b = mx.evaluate(nll, fine, u_f, nbytes)
    assert a.bpb == pytest.approx(b.bpb)
    assert a.ce_model != pytest.approx(b.ce_model)


def test_bpb_helper_matches_the_property() -> None:
    assert mx.bpb_from_nll(1234.0, 5000) == pytest.approx(1234.0 / (5000 * math.log(2)))


def test_zero_bytes_is_rejected() -> None:
    ids = [0, 1, 2]
    u = mx.fit_unigram([ids], 8)
    with pytest.raises(ValueError):
        mx.evaluate(1.0, ids, u, n_bytes=0)


def test_empty_token_stream_is_rejected() -> None:
    u = mx.fit_unigram([[0]], 8)
    with pytest.raises(ValueError):
        mx.evaluate(1.0, [], u, n_bytes=10)


# --- derived quantities ------------------------------------------------------------


def test_realized_fertility_and_perplexity() -> None:
    ids = stream(1500, 128)
    u = mx.fit_unigram([ids], 128)
    ev = mx.evaluate(1000.0, ids, u, n_bytes=6000)
    assert ev.realized_fertility == pytest.approx(1500 / 6000)
    assert mx.perplexity(ev.ce_model) == pytest.approx(math.exp(1000.0 / 1500))
    assert ev.bits_per_token == pytest.approx(ev.ce_model / math.log(2))
