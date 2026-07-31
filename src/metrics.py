"""L_u (primary) and BPB (secondary co-reported).

    L_u = -(1/T) Σ log [ p(wᵢ | w₁:ᵢ₋₁, V) / p(wᵢ | V) ]
        = CE_model  -  H_unigram          (both per token, in nats)

`L_u` is PRIMARY because Tao's law was selected under it and their released results are
in that column (`Lossu`). BPB is co-reported and carries a separately worded claim: BPB
rejecting while `L_u` agrees falsifies only the transfer of the recommendation to BPB,
not the law.

**Sign convention.** `L_u < 0` whenever the model beats the unigram baseline, and MORE
NEGATIVE IS BETTER. Their published rows are -1.35 / -2.10 / -2.50. This is easy to
invert by accident, so it is asserted in tests rather than assumed.

**Why BPB needs an exact byte count.** Per-token cross-entropy is not comparable across
tokenizers -- different segmentations give different denominators. BPB divides total NLL
by the exact number of UTF-8 bytes in the ORIGINAL text, which is tokenizer-independent.
It must never be derived from an average fertility: that is exact only when fertility and
NLL cover identical bytes, with no special tokens, padding, or boundary effects.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

LN2 = math.log(2.0)


# --- unigram baseline ---------------------------------------------------------------


@dataclass
class UnigramModel:
    """Token unigram distribution, estimated from the TRAINING split only.

    Fitting on validation or test text would leak the evaluation distribution into the
    normaliser and make `L_u` optimistic in a way no downstream check would catch.
    """

    logp: dict[int, float]
    vocab_size: int
    total_tokens: int
    alpha: float
    n_unseen: int

    def log_prob(self, token_id: int) -> float:
        try:
            return self.logp[token_id]
        except KeyError:
            raise KeyError(
                f"token {token_id} outside the fitted vocabulary of {self.vocab_size}; "
                "every id must have mass under add-alpha smoothing"
            ) from None

    def nll_nats(self, ids: Sequence[int]) -> float:
        return -sum(self.logp[i] for i in ids)


def fit_unigram(
    token_streams: Iterable[Sequence[int]], vocab_size: int, alpha: float = 1.0
) -> UnigramModel:
    """Add-alpha (Laplace at alpha=1) smoothing over the full vocabulary.

    Smoothing is not optional: any token with zero training count would otherwise carry
    infinite surprisal and make `L_u` undefined the moment it appears in evaluation.
    Every id in [0, V) receives mass, so `log_prob` is total.

    OPEN ITEM: the plan requires matching Tao's exact smoothing / special-token /
    boundary / zero-frequency conventions. Their released code does not expose the
    unigram construction, so add-1 over the full vocabulary is used as a documented,
    explicit default. It must be reconciled against their implementation before any
    confirmatory run, and the choice recorded in Preregistration I.
    """
    counts: Counter[int] = Counter()
    total = 0
    for ids in token_streams:
        counts.update(ids)
        total += len(ids)

    denom = total + alpha * vocab_size
    logp: dict[int, float] = {}
    for tid in range(vocab_size):
        logp[tid] = math.log((counts.get(tid, 0) + alpha) / denom)

    return UnigramModel(
        logp=logp,
        vocab_size=vocab_size,
        total_tokens=total,
        alpha=alpha,
        n_unseen=sum(1 for t in range(vocab_size) if counts.get(t, 0) == 0),
    )


# --- the metrics --------------------------------------------------------------------


@dataclass
class Evaluation:
    n_tokens: int
    n_bytes: int
    model_nll_nats: float
    unigram_nll_nats: float

    @property
    def ce_model(self) -> float:
        """Model cross-entropy, nats/token. NOT comparable across tokenizers."""
        return self.model_nll_nats / self.n_tokens

    @property
    def h_unigram(self) -> float:
        return self.unigram_nll_nats / self.n_tokens

    @property
    def l_u(self) -> float:
        """Primary metric. Negative when the model beats unigram; lower is better."""
        return self.ce_model - self.h_unigram

    @property
    def bpb(self) -> float:
        """Secondary metric. Total NLL over the EXACT original UTF-8 byte count."""
        return self.model_nll_nats / (self.n_bytes * LN2)

    @property
    def bits_per_token(self) -> float:
        return self.ce_model / LN2

    @property
    def realized_fertility(self) -> float:
        """tokens per byte -- compared against Tao's fitted f(V) as a reported result."""
        return self.n_tokens / self.n_bytes


def evaluate(
    model_nll_nats: float,
    token_ids: Sequence[int],
    unigram: UnigramModel,
    n_bytes: int,
) -> Evaluation:
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive; BPB divides by it")
    if len(token_ids) == 0:
        raise ValueError("no tokens to evaluate")
    return Evaluation(
        n_tokens=len(token_ids),
        n_bytes=n_bytes,
        model_nll_nats=model_nll_nats,
        unigram_nll_nats=unigram.nll_nats(token_ids),
    )


def bpb_from_nll(total_nll_nats: float, n_bytes: int) -> float:
    return total_nll_nats / (n_bytes * LN2)


def perplexity(ce_nats_per_token: float) -> float:
    return math.exp(ce_nats_per_token)
