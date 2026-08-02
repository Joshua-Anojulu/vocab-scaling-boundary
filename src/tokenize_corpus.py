"""Stage 3 of the data path: tokenize each split under each vocabulary.

Produces flat `uint16` token arrays on disk, one per (vocabulary, split), plus a manifest
recording exact token counts, byte counts and realized fertility.

Two decisions worth stating rather than burying:

**Only the needed prefix of `train` is tokenized.** Each configuration consumes exactly
`T_target = C / (6·(N_nv + V·d))` tokens, so tokenizing the whole 6 GB split under all
twenty vocabularies would be ~120 GB of redundant work. Per-vocabulary requirements range
from 0.08 GB to 2.33 GB of text.

Two qualifications, both load-bearing:

**The array needs `T_target + 1` ids, not `T_target`.** Sequences are self-contained --
sequence `k` spans `tokens[k*B : k*B + B + 1]`, `B` inputs plus the ONE lookahead token
that supplies the final target. An array holding exactly `n*B` ids therefore yields only
`n - 1` complete sequences, not `n`. The margins here are ~2% so this fails loudly via
`TokenStream`'s "holds no complete sequence" check rather than silently shortening a run,
but the requirement belongs in writing.

**Corpus order is preserved in the ARRAY, which is no longer the order runs read it in.**
It was, when consumption was sequential. Under the seed-semantics amendment a seed permutes
sequence order, so "a smaller budget reads a strict prefix of a larger one" now holds
per `(vocabulary, seed)` -- over the seed's permuted order -- rather than globally over
corpus order. The plan's requirement that the ordered corpus be held fixed is met by
drawing one permutation per `(vocabulary, seed)` over the WHOLE array and reading prefixes
of it; slicing this array to budget before building the stream would break that.

**Documents are joined with EOS.** Packed pretraining needs a boundary marker or the model
learns to run one document into the next. The plan specifies three special tokens but not
their use, and Tao's released code does not expose the packing convention. EOS-separation
is the standard choice and is recorded in AMENDMENTS.md (P4) as an open convention to
reconcile, because it changes token counts and therefore `L_u`.

`uint16` is safe: the largest vocabulary is 17,792, well inside 65,535. Asserted anyway.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import ingest as I, tokenizer as tk

TOKEN_DIR = Path("data/tokens")
#: Injectable so tests never write into the production tokenizer directory.
TOKENIZER_DIR = Path("tokenizers")
UINT16_MAX = 65_535
BATCH_DOCS = 2_000


@dataclass
class TokenizeReport:
    vocab_size: int
    split: str
    n_docs: int
    n_bytes: int
    n_chars: int
    n_tokens: int
    n_eos_inserted: int
    tokens_per_byte: float
    tokens_per_char: float
    bytes_per_token: float
    seconds: float
    path: str
    dtype: str = "uint16"


def _encode_stream(
    tok: Tokenizer, docs, eos_id: int, max_tokens: int | None
) -> tuple[np.ndarray, int, int, int, int]:
    """Encode documents in batches, EOS-separated, stopping at `max_tokens`."""
    chunks: list[np.ndarray] = []
    total = n_docs = n_bytes = n_chars = 0
    batch: list[str] = []

    def flush() -> bool:
        """Returns True when the token budget is reached."""
        nonlocal total, chunks
        if not batch:
            return False
        for enc in tok.encode_batch(batch):
            ids = enc.ids + [eos_id]
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
            if max_tokens is not None and total >= max_tokens:
                return True
        return False

    for d in docs:
        batch.append(d)
        n_docs += 1
        n_bytes += len(d.encode("utf-8"))
        n_chars += len(d)
        if len(batch) >= BATCH_DOCS:
            done = flush()
            batch = []
            if done:
                break
    else:
        flush()

    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    if max_tokens is not None and len(arr) > max_tokens:
        arr = arr[:max_tokens]
    return arr, n_docs, n_bytes, n_chars, n_docs


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write via a temp file, then replace. Two reasons, both encountered:

    * A crashed or interrupted tokenization must not leave a truncated `.npy` that loads
      cleanly and silently shortens a run's token budget.
    * On Windows an open `mmap_mode='r'` handle locks the path, and writing to it fails
      with `OSError: [Errno 22] Invalid argument`. Writing elsewhere and replacing avoids
      contending with a reader that is still holding the old array.
    """
    # NB: np.save appends '.npy' unless the name already ends with it, so the temp name
    # must end in '.npy' or the written path differs from the requested one.
    tmp = path.parent / (path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def tokenize_split(
    vocab_size: int,
    split: str,
    max_tokens: int | None = None,
    raw_root: str | Path = I.DEFAULT_ROOT,
    out_dir: str | Path = TOKEN_DIR,
    tokenizer_dir: str | Path = TOKENIZER_DIR,
) -> TokenizeReport:
    if vocab_size > UINT16_MAX:
        raise ValueError(f"vocab {vocab_size} exceeds uint16; widen the dtype")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = tk.load(Path(tokenizer_dir) / f"bpe_v{vocab_size}.json")
    if tok.get_vocab_size() != vocab_size:
        raise AssertionError(
            f"tokenizer at V={vocab_size} reports {tok.get_vocab_size()}; "
            "V_tok == V_head is a preregistered invariant"
        )
    eos = tok.token_to_id("<eos>")
    if eos is None:
        raise AssertionError("<eos> missing; packing needs a document separator")

    t0 = time.perf_counter()
    arr, n_docs, n_bytes, n_chars, n_eos = _encode_stream(
        tok, I.read_split(split, raw_root), eos, max_tokens
    )
    if arr.size and int(arr.max()) >= vocab_size:
        raise AssertionError(f"token id {int(arr.max())} >= V={vocab_size}")

    path = out_dir / f"v{vocab_size}_{split}.npy"
    _atomic_save(path, arr)
    secs = time.perf_counter() - t0

    return TokenizeReport(
        vocab_size=vocab_size, split=split, n_docs=n_docs, n_bytes=n_bytes,
        n_chars=n_chars, n_tokens=int(arr.size), n_eos_inserted=n_eos,
        tokens_per_byte=arr.size / n_bytes if n_bytes else 0.0,
        tokens_per_char=arr.size / n_chars if n_chars else 0.0,
        bytes_per_token=n_bytes / arr.size if arr.size else 0.0,
        seconds=secs, path=str(path),
    )


def load_tokens(vocab_size: int, split: str, out_dir: str | Path = TOKEN_DIR) -> np.ndarray:
    """Memory-mapped so a 1.4 GB array is not pulled into RAM to read a prefix."""
    return np.load(Path(out_dir) / f"v{vocab_size}_{split}.npy", mmap_mode="r")


def save_manifest(reports: list[TokenizeReport], out_dir: str | Path = TOKEN_DIR) -> None:
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
