"""Ingestion tests, run against injected documents so no network is required."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D, ingest as I  # noqa: E402

_A = "abcdefghijklmnopqrstuvwxyz "


def docs(n: int, seed: int = 0, size: int = 500) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(_A) for _ in range(size)) for _ in range(n)]


def test_ingest_partitions_and_persists(tmp_path) -> None:
    ds = docs(600)
    man = I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=ds)
    assert man.docs_seen == 600
    assert man.docs_kept == 600
    assert sum(man.per_split_docs.values()) == 600
    for s in D.SPLIT_BOUNDS:
        assert (tmp_path / f"{s}.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()


def test_round_trip_preserves_text_and_order(tmp_path) -> None:
    ds = docs(400, seed=3)
    I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=ds)
    expected = [d for d in ds if D.split_of(d) == "train"]
    assert list(I.read_split("train", tmp_path)) == expected


def test_rejected_documents_are_counted_not_silently_dropped(tmp_path) -> None:
    ds = docs(100) + ["", "short", "x" * (D.MIN_DOC_BYTES - 1)]
    man = I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=ds)
    assert man.docs_seen == 103
    assert man.docs_kept == 100
    assert man.rejected_short == 3


def test_caps_stop_ingestion_once_every_split_is_full(tmp_path) -> None:
    """Caps are how a 627B-token corpus is turned into a bounded local slice."""
    caps = {s: 20_000 for s in D.SPLIT_BOUNDS}
    man = I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=docs(20_000), caps_bytes=caps)
    for s, c in caps.items():
        assert man.per_split_bytes[s] >= c, f"{s} under-filled"
        # overshoot is bounded by one document
        assert man.per_split_bytes[s] < c + 1000
    assert man.docs_seen < 20_000, "should stop early once caps are met"


def test_a_full_split_does_not_stop_the_others(tmp_path) -> None:
    """final_test is 1% of buckets; it must not throttle filling of train."""
    caps = {"final_test": 2_000, "train": 100_000}
    man = I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=docs(4000), caps_bytes=caps)
    assert man.per_split_bytes["final_test"] >= 2_000
    assert man.per_split_bytes["train"] > 50_000


def test_manifest_hash_is_stable_across_identical_ingests(tmp_path) -> None:
    """Re-ingest can be PROVEN identical rather than assumed."""
    ds = docs(300, seed=11)
    a = I.ingest(source_key="slimpajama_6b", root=tmp_path / "a", docs=ds)
    b = I.ingest(source_key="slimpajama_6b", root=tmp_path / "b", docs=ds)
    assert a.per_split_sha256 == b.per_split_sha256
    assert a.per_split_bytes == b.per_split_bytes


def test_manifest_hash_changes_when_the_corpus_changes(tmp_path) -> None:
    a = I.ingest(source_key="slimpajama_6b", root=tmp_path / "a", docs=docs(300, seed=1))
    b = I.ingest(source_key="slimpajama_6b", root=tmp_path / "b", docs=docs(300, seed=2))
    assert a.per_split_sha256 != b.per_split_sha256


def test_manifest_records_the_preregistered_split_bounds(tmp_path) -> None:
    man = I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=docs(200))
    loaded = I.load_manifest(tmp_path)
    assert loaded.split_bounds == dict(D.SPLIT_BOUNDS)
    assert loaded.min_doc_bytes == D.MIN_DOC_BYTES
    assert loaded.per_split_sha256 == man.per_split_sha256


def test_read_split_respects_max_bytes(tmp_path) -> None:
    I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=docs(800))
    out = list(I.read_split("train", tmp_path, max_bytes=5_000))
    assert sum(len(d.encode("utf-8")) for d in out) <= 5_000
    assert out == list(I.read_split("train", tmp_path))[: len(out)]


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        next(I.stream_source("not_a_real_source"))


def test_preregistered_source_is_recorded_as_unavailable() -> None:
    """The preregistered corpus is DELETED, not gated -- established with a valid token.

    Pinned as a test because the distinction determines whether a deviation was forced or
    chosen, and the anonymous 401 is actively misleading about which.
    """
    pre = I.SOURCES["cerebras_chunk1"]
    assert pre["available"] is False
    assert "PREREGISTERED" in pre["note"] and "DELETED" in pre["note"]
    assert "404" in pre["note"]


def test_rejected_mirrors_record_why() -> None:
    for alt in ("rokset3_chunk1", "ultraronin_chunk1"):
        assert "REJECTED" in I.SOURCES[alt]["note"]


def test_adopted_source_is_the_default() -> None:
    assert I.DEFAULT_SOURCE == "slimpajama_6b"
    assert "ADOPTED" in I.SOURCES[I.DEFAULT_SOURCE]["note"]


def test_jsonl_lines_are_valid_and_utf8_safe(tmp_path) -> None:
    ds = docs(50) + ["日本語のテキスト " * 40, "emoji 🙂🚀 " * 40]
    I.ingest(source_key="slimpajama_6b", root=tmp_path, docs=ds)
    for s in D.SPLIT_BOUNDS:
        for line in (tmp_path / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            assert isinstance(json.loads(line)["text"], str)
