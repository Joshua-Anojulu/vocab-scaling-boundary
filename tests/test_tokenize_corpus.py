"""Tests for the tokenization stage."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ingest as I, tokenize_corpus as TC, tokenizer as tk  # noqa: E402

_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _lexicon(seed: int = 0, n: int = 3000) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(_ALPHA) for _ in range(rng.randint(3, 9))) for _ in range(n)]


def corpus(n: int = 400, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    lex = _lexicon(seed)
    return [" ".join(rng.choice(lex) for _ in range(120)) for _ in range(n)]


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    """Ingest a small corpus and train one tokenizer against it (expensive; shared)."""
    root = tmp_path_factory.mktemp("raw")
    tdir = tmp_path_factory.mktemp("toks")
    # 3000, not 500: selection_val and final_test are 1% of buckets each, and at 500
    # documents selection_val drew ZERO. That surfaced downstream as an empty token
    # array rather than as an obviously-too-small fixture, so the precondition is
    # asserted here instead of being discovered three tests later.
    docs = corpus(3000)
    man = I.ingest(source_key="slimpajama_6b", root=root, docs=docs)
    empty = [s for s, n in man.per_split_docs.items() if n == 0]
    assert not empty, f"fixture corpus too small; empty splits: {empty}"
    tok, rep = tk.train(512, docs)
    # NEVER write into the production tokenizers/ directory: an earlier version of this
    # fixture did, silently replacing the V=512 tokenizer trained on the real 150 MB
    # sample with one trained on 500 synthetic documents.
    tk.save(tok, rep, tdir)
    return root, tdir


@pytest.fixture
def staged(raw_root, tmp_path):
    """Per-test output directory.

    Sharing one output dir across tests deadlocks on Windows: a test that leaves an
    `mmap_mode='r'` handle open locks the path, and the next test's write to it fails
    with EINVAL. Isolating outputs keeps the tests independent of execution order.
    """
    root, tdir = raw_root
    return root, tmp_path, tdir


def test_roundtrip_tokens_decode_to_the_source(staged) -> None:
    """EOS-separated packing must still decode back to the original documents."""
    root, out, tdir = staged
    r = TC.tokenize_split(512, "selection_val", raw_root=root, out_dir=out, tokenizer_dir=tdir)
    arr = TC.load_tokens(512, "selection_val", out)
    assert arr.dtype == np.uint16
    assert len(arr) == r.n_tokens

    tok = tk.load(Path(tdir) / "bpe_v512.json")
    eos = tok.token_to_id("<eos>")
    docs = list(I.read_split("selection_val", root))
    # split on EOS and decode each segment
    ids = arr.tolist()
    segs, cur = [], []
    for i in ids:
        if i == eos:
            segs.append(cur); cur = []
        else:
            cur.append(i)
    assert len(segs) == len(docs)
    for seg, d in zip(segs[:20], docs[:20]):
        assert tok.decode(seg) == d


def test_one_eos_per_document(staged) -> None:
    root, out, tdir = staged
    r = TC.tokenize_split(512, "final_test", raw_root=root, out_dir=out, tokenizer_dir=tdir)
    arr = TC.load_tokens(512, "final_test", out)
    tok = tk.load(Path(tdir) / "bpe_v512.json")
    eos = tok.token_to_id("<eos>")
    assert int((arr == eos).sum()) == r.n_docs


def test_max_tokens_truncates_exactly(staged) -> None:
    """The token budget is what makes C exact -- it must not overshoot."""
    root, out, tdir = staged
    cap = 5_000
    r = TC.tokenize_split(512, "train", max_tokens=cap, raw_root=root, out_dir=out, tokenizer_dir=tdir)
    assert r.n_tokens == cap
    assert len(TC.load_tokens(512, "train", out)) == cap


def test_truncated_output_is_a_prefix_of_the_untruncated(staged) -> None:
    """A smaller budget must read a strict PREFIX of a larger one, in corpus order."""
    root, out, tdir = staged
    TC.tokenize_split(512, "train", max_tokens=8_000, raw_root=root, out_dir=out, tokenizer_dir=tdir)
    big = np.array(TC.load_tokens(512, "train", out))
    TC.tokenize_split(512, "train", max_tokens=3_000, raw_root=root, out_dir=out, tokenizer_dir=tdir)
    small = np.array(TC.load_tokens(512, "train", out))
    assert np.array_equal(small, big[: len(small)])


def test_all_ids_are_within_the_vocabulary(staged) -> None:
    root, out, tdir = staged
    TC.tokenize_split(512, "selection_val", raw_root=root, out_dir=out, tokenizer_dir=tdir)
    arr = TC.load_tokens(512, "selection_val", out)
    assert int(arr.max()) < 512
    assert int(arr.min()) >= 0


def test_reported_fertility_matches_the_array(staged) -> None:
    root, out, tdir = staged
    r = TC.tokenize_split(512, "final_test", raw_root=root, out_dir=out, tokenizer_dir=tdir)
    assert r.tokens_per_byte == pytest.approx(r.n_tokens / r.n_bytes)
    assert r.bytes_per_token == pytest.approx(1.0 / r.tokens_per_byte)


def test_oversized_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match="uint16"):
        TC.tokenize_split(70_000, "train")


def test_load_is_memory_mapped(staged) -> None:
    """A 1.4 GB token array must not be pulled into RAM to read a prefix."""
    root, out, tdir = staged
    TC.tokenize_split(512, "selection_val", raw_root=root, out_dir=out, tokenizer_dir=tdir)
    arr = TC.load_tokens(512, "selection_val", out)
    assert isinstance(arr, np.memmap)
