"""Tokenizer invariant tests.

Three properties the study's arithmetic depends on:

  1. total size is EXACTLY V (not V+3) -- otherwise V_tok != V_head and every
     parameter/FLOP figure derived from V is wrong;
  2. round-trip is lossless -- BPB divides by an exact UTF-8 byte count;
  3. unknown-token rate is zero -- asserted by the plan.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import tokenizer as tk  # noqa: E402

_BASE = ("the quick brown fox jumps over lazy dogs while parsing tokens and bytes "
         "language model vocabulary scaling boundary experiment").split()

# BPE needs enough DISTINCT merge candidates to reach the target vocabulary. A handful of
# repeated words exhausts after ~350 merges -- which is what the size assertion caught on
# the first run. A varied lexicon keeps the tests exercising the invariants rather than
# the corpus.
_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _lexicon(seed: int = 0, n: int = 4000) -> list[str]:
    rng = random.Random(seed)
    words = list(_BASE)
    for _ in range(n):
        k = rng.randint(3, 9)
        words.append("".join(rng.choice(_ALPHA) for _ in range(k)))
    return words


def corpus(n: int = 400, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    lex = _lexicon(seed)
    return [" ".join(rng.choice(lex) for _ in range(120)) for _ in range(n)]


@pytest.fixture(scope="module")
def trained() -> tuple:
    tok, rep = tk.train(512, corpus())
    return tok, rep


# --- the V-3 rule ------------------------------------------------------------------


@pytest.mark.parametrize("V", [384, 512, 1024])
def test_realized_size_is_exactly_V(V: int) -> None:
    """Trained to V-3, plus 3 specials, equals V. Not V+3."""
    tok, rep = tk.train(V, corpus(200))
    assert tok.get_vocab_size() == V
    assert rep.realized_size == V
    assert rep.lexical_target == V - tk.N_SPECIAL
    assert rep.n_special == tk.N_SPECIAL


def test_specials_are_present_and_distinct() -> None:
    tok, _ = tk.train(512, corpus(200))
    ids = [tok.token_to_id(s) for s in tk.SPECIAL_TOKENS]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == tk.N_SPECIAL


def test_corpus_exhaustion_fails_loudly_with_a_diagnosable_message() -> None:
    """A corpus too small to reach V must raise, not silently yield a smaller vocabulary.

    This fired for real on the first run of these tests: a repetitive 400-document corpus
    exhausted at 358 tokens against a 509 target. Silently accepting that would have made
    V_tok != V_head, corrupting every parameter and FLOP figure derived from V.
    """
    tiny = ["hello world " * 20] * 5
    with pytest.raises(AssertionError, match="TRAINING CORPUS EXHAUSTED"):
        tk.train(4096, tiny)


def test_below_byte_alphabet_floor_is_refused() -> None:
    """256 bytes + specials cannot fit under 384; fail loudly rather than silently."""
    with pytest.raises(ValueError):
        tk.build_tokenizer(256)


# --- losslessness and UNK ----------------------------------------------------------


def test_roundtrip_is_lossless_on_training_text(trained) -> None:
    tok, rep = trained
    assert rep.roundtrip_ok


def test_roundtrip_survives_bytes_absent_from_training(trained) -> None:
    """The initial alphabet seeds all 256 bytes, so unseen input still round-trips.

    Without `initial_alphabet=ByteLevel.alphabet()` this is exactly where losslessness
    and the zero-UNK claim would quietly fail.
    """
    tok, _ = trained
    for s in ("日本語のテキスト", "emoji 🙂🚀", "\t\n weird \x0b control", "Ω≈ç√∫˜µ", "«»‹›"):
        assert tok.decode(tok.encode(s).ids) == s, f"round-trip failed for {s!r}"


def test_unknown_token_rate_is_zero(trained) -> None:
    tok, rep = trained
    assert rep.unk_rate == 0.0
    assert tok.token_to_id("<unk>") is None      # no UNK token exists at all


def test_every_byte_value_is_representable() -> None:
    tok, _ = tk.train(384, corpus(150))
    raw = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(raw).ids) == raw


# --- fertility ---------------------------------------------------------------------


def test_fertility_falls_as_vocabulary_grows() -> None:
    """More merges means fewer tokens per byte. This is the mechanism f(V) models."""
    docs = corpus(300)
    f = []
    for V in (384, 1024, 4096):
        tok, _ = tk.train(V, docs)
        f.append(tk.measure_fertility(tok, docs)["tokens_per_byte"])
    assert f[0] > f[1] > f[2], f"fertility should decrease with V, got {f}"


def test_measured_fertility_is_self_consistent(trained) -> None:
    tok, _ = trained
    docs = corpus(50, seed=9)
    m = tk.measure_fertility(tok, docs)
    assert m["tokens_per_byte"] == pytest.approx(m["tokens"] / m["bytes"])
    assert m["bytes_per_token"] == pytest.approx(1.0 / m["tokens_per_byte"])
    assert m["tokens_per_char"] >= m["tokens_per_byte"]   # bytes >= chars for UTF-8


# --- persistence -------------------------------------------------------------------


def test_save_and_load_preserves_encoding(tmp_path, trained) -> None:
    tok, rep = trained
    p = tk.save(tok, rep, tmp_path)
    again = tk.load(p)
    assert again.get_vocab_size() == rep.vocab_size
    s = "a round trip through disk must not change the encoding"
    assert again.encode(s).ids == tok.encode(s).ids


def test_training_is_deterministic() -> None:
    """Same corpus, same V, same merges -- required for a reproducible preregistration."""
    docs = corpus(200, seed=3)
    a, _ = tk.train(512, docs)
    b, _ = tk.train(512, docs)
    s = "determinism check over tokens and bytes"
    assert a.encode(s).ids == b.encode(s).ids
    assert a.get_vocab() == b.get_vocab()
