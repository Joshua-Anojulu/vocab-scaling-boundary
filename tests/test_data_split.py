"""Tests for the four-way document split.

The split is load-bearing for the study's validity, not just its plumbing: final-test is
touched exactly once by the frozen estimator, so any leakage between partitions
invalidates the reported estimand. These tests check the partition is total, disjoint,
content-addressed and order-independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402


def doc(i: int, n: int = 400) -> str:
    """A synthetic document long enough to pass the byte filter."""
    return f"document {i} " + ("lorem ipsum dolor sit amet " * (n // 26 + 1))


# --- the partition itself ----------------------------------------------------------


def test_split_bounds_are_total_and_disjoint() -> None:
    """Every bucket 0..999 belongs to exactly one split."""
    seen: dict[int, str] = {}
    for name, (lo, hi) in data.SPLIT_BOUNDS.items():
        for b in range(lo, hi + 1):
            assert b not in seen, f"bucket {b} claimed by {seen.get(b)} and {name}"
            seen[b] = name
    assert len(seen) == data.N_BUCKETS


def test_fractions_sum_to_one() -> None:
    assert sum(data.expected_fraction(s) for s in data.SPLIT_BOUNDS) == pytest.approx(1.0)


def test_final_test_is_one_percent() -> None:
    """Small on purpose: it is spent once, so it only needs enough for one evaluation."""
    assert data.expected_fraction("final_test") == 0.01
    assert data.expected_fraction("selection_val") == 0.01
    assert data.expected_fraction("tokenizer_fit") == 0.02
    assert data.expected_fraction("train") == 0.96


# --- content addressing ------------------------------------------------------------


def test_bucket_is_deterministic() -> None:
    t = doc(7)
    assert data.bucket_of(t) == data.bucket_of(t)


def test_bucket_is_content_addressed_not_position_addressed() -> None:
    """The same text must land in the same split no matter where it appears.

    This is why the corpus can be re-downloaded, resumed, or re-sharded without silently
    re-partitioning documents and leaking test data into training.
    """
    t = doc(999_999)                       # deliberately outside the surrounding ranges
    first = data.split_of(t)
    for prefix in (0, 50, 500):
        stream = [doc(i) for i in range(prefix)] + [t] + [doc(i) for i in range(prefix, prefix + 50)]
        assert stream.count(t) == 1
        idx = stream.index(t)
        assert data.split_of(stream[idx]) == first, "split moved with position"
    assert data.split_of(t) == first


def test_distribution_matches_expected_fractions() -> None:
    """With 20k documents the empirical split should track the bucket ranges closely."""
    n = 20_000
    counts = {k: 0 for k in data.SPLIT_BOUNDS}
    for i in range(n):
        counts[data.split_of(doc(i))] += 1
    for name in data.SPLIT_BOUNDS:
        exp = data.expected_fraction(name)
        obs = counts[name] / n
        assert abs(obs - exp) < 0.01, f"{name}: expected ~{exp:.3f}, got {obs:.3f}"


def test_splits_are_mutually_exclusive_over_a_stream() -> None:
    docs = [doc(i) for i in range(3000)]
    members = {k: set(data.iter_split(docs, k)) for k in data.SPLIT_BOUNDS}
    keys = list(members)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (members[a] & members[b]), f"{a} and {b} overlap"
    assert sum(len(v) for v in members.values()) == len(docs)


# --- admissibility -----------------------------------------------------------------


def test_short_documents_are_rejected() -> None:
    assert not data.is_admissible("")
    assert not data.is_admissible("x" * (data.MIN_DOC_BYTES - 1))
    assert data.is_admissible("x" * data.MIN_DOC_BYTES)


def test_byte_length_not_character_length_is_the_filter() -> None:
    """Multi-byte characters count as their UTF-8 length, not one each."""
    s = "é" * 100                       # 100 chars, 200 bytes
    assert len(s) < data.MIN_DOC_BYTES <= len(s.encode("utf-8"))
    assert data.is_admissible(s)


def test_surrogates_are_rejected_rather_than_crashing() -> None:
    """Lone surrogates cannot be encoded; they must be filtered, not raise."""
    bad = "valid text " * 20 + "\ud800"
    assert not data.is_admissible(bad)


# --- streaming behaviour -----------------------------------------------------------


def test_iter_split_preserves_corpus_order() -> None:
    """Two budgets must see the same documents in the same sequence."""
    docs = [doc(i) for i in range(2000)]
    out = list(data.iter_split(docs, "train"))
    assert out == [d for d in docs if data.split_of(d) == "train"]


def test_max_bytes_truncates_without_exceeding() -> None:
    docs = [doc(i) for i in range(2000)]
    cap = 50_000
    out = list(data.iter_split(docs, "train", max_bytes=cap))
    total = sum(len(d.encode("utf-8")) for d in out)
    assert total <= cap
    assert out                                    # non-empty
    # and it is a prefix of the untruncated stream
    assert out == list(data.iter_split(docs, "train"))[: len(out)]


def test_stats_account_for_every_document() -> None:
    docs = [doc(i) for i in range(500)] + ["", "tiny"]
    st = data.SplitStats()
    list(data.iter_split(docs, "train", stats=st))
    assert st.seen == len(docs)
    assert st.rejected_short == 2
    assert st.total_kept == 500
    assert sum(st.bytes_kept.values()) == sum(len(d.encode("utf-8")) for d in docs[:500])


def test_unknown_split_name_raises() -> None:
    with pytest.raises(ValueError):
        list(data.iter_split([doc(1)], "validation"))
