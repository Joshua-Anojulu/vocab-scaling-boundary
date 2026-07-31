"""Corpus ingestion and the four-way document split.

Every parameter here is preregistered in PLAN.md and must not drift, because the split
determines which documents can influence a decision and which can only be reported:

    tokenizer-fit      buckets   0- 19   tokenizers are trained ONLY on this
    train              buckets  20-979   model training; unigram probs for L_u
    selection-val      buckets 980-989   boundary-extension decision ONLY
    final-test         buckets 990-999   the frozen estimator runs ONCE, here

`bucket = int(sha256(document_text.utf8).hexdigest()[:8], 16) % 1000`

Content-hashed rather than index-hashed on purpose: the partition is then a property of
the document itself, reproducible from the text alone, and invariant to shard ordering,
resumption, or a corpus re-download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator

# --- preregistered split ------------------------------------------------------------

SPLIT_BOUNDS: dict[str, tuple[int, int]] = {
    "tokenizer_fit": (0, 19),
    "train": (20, 979),
    "selection_val": (980, 989),
    "final_test": (990, 999),
}

MIN_DOC_BYTES = 128
N_BUCKETS = 1000

#: Corpus identity, pinned. Recorded so a re-download is verifiably the same corpus.
CORPUS = {
    "dataset": "cerebras/SlimPajama-627B",
    "split": "train",
    "chunk": "chunk1",
    "shard_order": "ascending index",
    "rng_seed": 1234,
}


def bucket_of(text: str) -> int:
    """Deterministic partition key. Content-addressed, not position-addressed."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % N_BUCKETS


def split_of(text: str) -> str:
    b = bucket_of(text)
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        if lo <= b <= hi:
            return name
    raise AssertionError(f"bucket {b} outside every range -- SPLIT_BOUNDS is not total")


def is_admissible(text: str) -> bool:
    """Preregistered filter: >= MIN_DOC_BYTES of UTF-8, and round-trippable.

    The round-trip check matters more than it looks. The study asserts lossless
    tokenization and a zero unknown-token rate; a document that cannot survive
    encode/decode would break that invariant far downstream, where it would look like a
    tokenizer bug rather than a data problem.
    """
    if not text:
        return False
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(encoded) < MIN_DOC_BYTES:
        return False
    try:
        return encoded.decode("utf-8") == text
    except UnicodeDecodeError:
        return False


# --- streaming ----------------------------------------------------------------------


@dataclass
class SplitStats:
    seen: int = 0
    rejected_short: int = 0
    rejected_undecodable: int = 0
    kept: dict[str, int] = None          # type: ignore[assignment]
    bytes_kept: dict[str, int] = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kept is None:
            self.kept = {k: 0 for k in SPLIT_BOUNDS}
        if self.bytes_kept is None:
            self.bytes_kept = {k: 0 for k in SPLIT_BOUNDS}

    @property
    def total_kept(self) -> int:
        return sum(self.kept.values())


def iter_split(
    docs: Iterable[str],
    want: str,
    stats: SplitStats | None = None,
    max_bytes: int | None = None,
) -> Iterator[str]:
    """Yield admissible documents belonging to `want`, in corpus order.

    Corpus order is preserved deliberately: the plan holds the ordered source corpus
    fixed and lets each configuration consume as many BYTES as its FLOP budget allows, so
    two runs at different budgets must see the same documents in the same sequence.
    """
    if want not in SPLIT_BOUNDS:
        raise ValueError(f"unknown split {want!r}; expected one of {sorted(SPLIT_BOUNDS)}")
    used = 0
    for text in docs:
        if stats is not None:
            stats.seen += 1
        if not text:
            if stats is not None:
                stats.rejected_short += 1
            continue
        try:
            nbytes = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            if stats is not None:
                stats.rejected_undecodable += 1
            continue
        if not is_admissible(text):
            if stats is not None:
                if nbytes < MIN_DOC_BYTES:
                    stats.rejected_short += 1
                else:
                    stats.rejected_undecodable += 1
            continue
        s = split_of(text)
        if stats is not None:
            stats.kept[s] += 1
            stats.bytes_kept[s] += nbytes
        if s != want:
            continue
        if max_bytes is not None and used + nbytes > max_bytes:
            return
        used += nbytes
        yield text


def expected_fraction(split: str) -> float:
    lo, hi = SPLIT_BOUNDS[split]
    return (hi - lo + 1) / N_BUCKETS
