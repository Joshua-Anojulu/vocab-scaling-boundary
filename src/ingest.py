"""Corpus ingestion: stream a source, filter, split, and persist with a manifest.

Stage 1 of the data path. Produces raw text per split; tokenization is deliberately a
separate later stage, because the tokenizers themselves must be trained on the
tokenizer-fit split before anything else can be tokenized.

Output layout:

    data/raw/tokenizer_fit.jsonl
    data/raw/train.jsonl
    data/raw/selection_val.jsonl
    data/raw/final_test.jsonl
    data/raw/manifest.json

One JSON object per line, `{"text": ...}`, preserving corpus order. Order matters: the
plan holds the ordered source fixed so runs at different budgets read the same documents
in the same sequence, differing only in how far they get.

The manifest records source identity, per-split counts and byte totals, rejection counts,
and a rolling SHA-256 over the emitted text of each split, so a re-ingest can be proven
identical rather than assumed to be.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import data as D

DEFAULT_ROOT = Path("data/raw")

#: The source actually used. See AMENDMENTS.md A1 for why it is not the preregistered one.
DEFAULT_SOURCE = "slimpajama_6b"

SOURCES: dict[str, dict] = {
    "cerebras_chunk1": {
        "repo": "cerebras/SlimPajama-627B",
        "split": "train",
        "available": False,
        "note": (
            "PREREGISTERED but DELETED from the Hub. Authenticated, dataset_info() "
            "returns 404 RepositoryNotFoundError; the anonymous 401 was the Hub masking "
            "existence, not a terms gate. The cerebras org hosts no SlimPajama dataset. "
            "See AMENDMENTS.md A1."
        ),
    },
    "rokset3_chunk1": {
        "repo": "rokset3/slim_pajama_chunk1",
        "split": "train",
        "available": True,
        "note": (
            "Third-party chunk1 mirror. REJECTED: disagrees with the UltraRonin mirror on "
            "0/200 sampled documents, so fidelity to the original ordering is "
            "unverifiable -- deviation with no provenance benefit."
        ),
    },
    "ultraronin_chunk1": {
        "repo": "UltraRonin/SlimPajama-chunk1",
        "split": "train",
        "available": True,
        "note": "Second chunk1 mirror. REJECTED for the same reason as rokset3_chunk1.",
    },
    "slimpajama_6b": {
        "repo": "DKYoon/SlimPajama-6B",
        "split": "train",
        "available": True,
        "note": (
            "ADOPTED (AMENDMENTS.md A1). Sampled 6B-token subset of SlimPajama, not "
            "chunk1. Chosen for provenance and reproducibility: 13,536 downloads vs 342 "
            "and 419 for the mirrors, and it stays within the corpus family the law under "
            "test was fitted on."
        ),
    },
}


@dataclass
class IngestManifest:
    source_key: str
    repo: str
    split: str
    started_utc: str
    finished_utc: str = ""
    docs_seen: int = 0
    docs_kept: int = 0
    rejected_short: int = 0
    rejected_undecodable: int = 0
    per_split_docs: dict[str, int] = field(default_factory=dict)
    per_split_bytes: dict[str, int] = field(default_factory=dict)
    per_split_sha256: dict[str, str] = field(default_factory=dict)
    caps_bytes: dict[str, int] = field(default_factory=dict)
    min_doc_bytes: int = D.MIN_DOC_BYTES
    split_bounds: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(D.SPLIT_BOUNDS))
    seconds: float = 0.0


def stream_source(source_key: str, limit: int | None = None) -> Iterator[str]:
    """Yield raw document texts from a Hub dataset, in dataset order."""
    from datasets import load_dataset  # imported lazily; ingestion is optional

    if source_key not in SOURCES:
        raise ValueError(f"unknown source {source_key!r}; known: {sorted(SOURCES)}")
    spec = SOURCES[source_key]
    ds = load_dataset(spec["repo"], split=spec["split"], streaming=True)
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            return
        text = rec.get("text")
        if isinstance(text, str):
            yield text


def ingest(
    source_key: str = DEFAULT_SOURCE,
    root: str | Path = DEFAULT_ROOT,
    caps_bytes: dict[str, int] | None = None,
    limit_docs: int | None = None,
    docs: Iterable[str] | None = None,
    progress_every: int = 100_000,
    on_progress: Callable[[IngestManifest], None] | None = None,
) -> IngestManifest:
    """Stream, filter, partition and persist. Stops once every cap is satisfied.

    `docs` overrides the Hub stream, which is what the tests use -- ingestion logic is
    then exercised without a network dependency.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    caps = dict(caps_bytes or {})
    spec = SOURCES.get(source_key, {"repo": "(injected)", "split": "(injected)"})

    man = IngestManifest(
        source_key=source_key,
        repo=spec["repo"],
        split=spec.get("split", ""),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        caps_bytes=caps,
    )
    for s in D.SPLIT_BOUNDS:
        man.per_split_docs[s] = 0
        man.per_split_bytes[s] = 0

    hashers = {s: hashlib.sha256() for s in D.SPLIT_BOUNDS}
    handles = {s: open(root / f"{s}.jsonl", "w", encoding="utf-8") for s in D.SPLIT_BOUNDS}
    t0 = time.perf_counter()

    def satisfied() -> bool:
        """All capped splits full. Uncapped splits never satisfy, so caps are required."""
        return bool(caps) and all(
            man.per_split_bytes.get(s, 0) >= c for s, c in caps.items()
        )

    try:
        source = docs if docs is not None else stream_source(source_key, limit=limit_docs)
        for text in source:
            man.docs_seen += 1
            if not text:
                man.rejected_short += 1
                continue
            try:
                nbytes = len(text.encode("utf-8"))
            except UnicodeEncodeError:
                man.rejected_undecodable += 1
                continue
            if not D.is_admissible(text):
                if nbytes < D.MIN_DOC_BYTES:
                    man.rejected_short += 1
                else:
                    man.rejected_undecodable += 1
                continue

            s = D.split_of(text)
            cap = caps.get(s)
            if cap is not None and man.per_split_bytes[s] >= cap:
                continue                       # this split is full; keep scanning others

            handles[s].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            hashers[s].update(text.encode("utf-8"))
            man.per_split_docs[s] += 1
            man.per_split_bytes[s] += nbytes
            man.docs_kept += 1

            if on_progress and man.docs_seen % progress_every == 0:
                on_progress(man)
            if satisfied():
                break
    finally:
        for h in handles.values():
            h.close()

    man.seconds = time.perf_counter() - t0
    man.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man.per_split_sha256 = {s: h.hexdigest() for s, h in hashers.items()}
    (root / "manifest.json").write_text(json.dumps(asdict(man), indent=2), encoding="utf-8")
    return man


def read_split(split: str, root: str | Path = DEFAULT_ROOT, max_bytes: int | None = None) -> Iterator[str]:
    """Stream persisted documents back, in ingest order."""
    p = Path(root) / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run ingest() first")
    used = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)["text"]
            n = len(t.encode("utf-8"))
            if max_bytes is not None and used + n > max_bytes:
                return
            used += n
            yield t


def load_manifest(root: str | Path = DEFAULT_ROOT) -> IngestManifest:
    d = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    d["split_bounds"] = {k: tuple(v) for k, v in d["split_bounds"].items()}
    return IngestManifest(**d)
