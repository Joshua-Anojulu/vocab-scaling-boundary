"""Byte-level BPE training, one tokenizer per vocabulary size.

Every setting is preregistered. The two that are easy to get wrong, and that the study's
invariants depend on:

* **Train the lexical BPE to `V - 3`, then add the 3 specials, so the total is EXACTLY V.**
  Training to `V` and adding specials afterwards yields `V + 3`, which silently breaks
  `V_tok == V_head == V` and every parameter/FLOP figure derived from it. Asserted, not
  trusted.
* **Seed the initial alphabet with all 256 byte values.** Without it, bytes absent from
  the training sample have no representation, round-trip stops being lossless, and the
  unknown-token rate stops being zero -- which the plan asserts and BPB depends on.

Normalization is deliberately NONE. Any Unicode normalization would make
`decode(encode(x)) == x` false for some inputs, breaking the exact-byte denominator that
BPB is computed over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

#: BOS, EOS, PAD -- added AFTER training, hence the `V - 3` target.
SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>"]
N_SPECIAL = len(SPECIAL_TOKENS)

#: GPT-2 byte-level pre-tokenization. `pre_tokenizers.ByteLevel(use_regex=True)` applies
#: exactly this pattern; it is named here so the preregistration is explicit rather than
#: implied by a library default that could change.
GPT2_PRETOKENIZER_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

MIN_VOCAB = 384          # 256 byte alphabet + specials, snapped to the 128 quantum


@dataclass
class TokenizerReport:
    vocab_size: int
    lexical_target: int
    realized_size: int
    n_special: int
    train_docs: int
    train_bytes: int
    roundtrip_ok: bool
    unk_rate: float
    fertility_tokens_per_byte: float
    fertility_tokens_per_char: float


def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    if vocab_size < MIN_VOCAB:
        raise ValueError(f"vocab_size {vocab_size} below the byte-alphabet floor {MIN_VOCAB}")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = None                                    # see module docstring
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - N_SPECIAL,                   # <- the V-3 rule
        special_tokens=[],                                   # added after training
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
        min_frequency=0,
    )
    return tok, trainer


def train(vocab_size: int, docs: Iterable[str]) -> tuple[Tokenizer, TokenizerReport]:
    """Train one tokenizer and verify its invariants before it can be used."""
    tok, trainer = build_tokenizer(vocab_size)

    n_docs = 0
    n_bytes = 0
    kept: list[str] = []

    def counting() -> Iterable[str]:
        nonlocal n_docs, n_bytes
        for d in docs:
            n_docs += 1
            n_bytes += len(d.encode("utf-8"))
            kept.append(d)
            yield d

    tok.train_from_iterator(counting(), trainer)
    added = tok.add_special_tokens(SPECIAL_TOKENS)
    realized = tok.get_vocab_size()

    if realized != vocab_size:
        shortfall = vocab_size - realized
        cause = (
            "TRAINING CORPUS EXHAUSTED: BPE ran out of merge candidates before reaching "
            f"the target. It found only {realized - added} lexical tokens against a "
            f"target of {vocab_size - N_SPECIAL}. Use more, or more varied, text."
            if shortfall > 0 else
            "OVERSHOOT: more tokens than requested -- check that specials are not also "
            "being added by the trainer."
        )
        raise AssertionError(
            f"tokenizer realized {realized} tokens, expected exactly {vocab_size} "
            f"(lexical target {vocab_size - N_SPECIAL} + {added} specials). "
            f"V_tok == V_head is a preregistered invariant. {cause}"
        )

    sample = kept[: min(len(kept), 200)]
    rt_ok, unk, n_tok, n_by, n_ch = _audit(tok, sample)
    return tok, TokenizerReport(
        vocab_size=vocab_size,
        lexical_target=vocab_size - N_SPECIAL,
        realized_size=realized,
        n_special=added,
        train_docs=n_docs,
        train_bytes=n_bytes,
        roundtrip_ok=rt_ok,
        unk_rate=unk,
        fertility_tokens_per_byte=(n_tok / n_by) if n_by else 0.0,
        fertility_tokens_per_char=(n_tok / n_ch) if n_ch else 0.0,
    )


def _audit(tok: Tokenizer, docs: list[str]) -> tuple[bool, float, int, int, int]:
    """Round-trip losslessness and unknown-token rate over a sample."""
    ok = True
    n_tok = n_by = n_ch = 0
    unk_id = tok.token_to_id("<unk>")
    unk_hits = 0
    for d in docs:
        enc = tok.encode(d)
        n_tok += len(enc.ids)
        n_by += len(d.encode("utf-8"))
        n_ch += len(d)
        if unk_id is not None:
            unk_hits += sum(1 for i in enc.ids if i == unk_id)
        if tok.decode(enc.ids) != d:
            ok = False
    return ok, (unk_hits / n_tok if n_tok else 0.0), n_tok, n_by, n_ch


def measure_fertility(tok: Tokenizer, docs: Iterable[str]) -> dict:
    """Realized fertility -- the quantity Tao's fitted f(V) is compared against.

    The plan enforces IsoFLOP on exact TOKEN counts precisely because this will differ
    from f(V); the divergence is a reported result, not an error.
    """
    n_tok = n_by = n_ch = n_doc = 0
    for d in docs:
        n_tok += len(tok.encode(d).ids)
        n_by += len(d.encode("utf-8"))
        n_ch += len(d)
        n_doc += 1
    return {
        "docs": n_doc, "tokens": n_tok, "bytes": n_by, "chars": n_ch,
        "tokens_per_byte": n_tok / n_by if n_by else 0.0,
        "tokens_per_char": n_tok / n_ch if n_ch else 0.0,
        "bytes_per_token": n_by / n_tok if n_tok else 0.0,
    }


def save(tok: Tokenizer, report: TokenizerReport, outdir: str | Path) -> Path:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"bpe_v{report.vocab_size}.json"
    tok.save(str(p))
    (d / f"bpe_v{report.vocab_size}.report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return p


def load(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
