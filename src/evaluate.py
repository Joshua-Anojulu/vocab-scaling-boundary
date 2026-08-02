"""Compute the primary metric `L_u` (and co-reported BPB) for a trained run.

`L_u = CE_model - H_unigram`, both per token in nats. `src/metrics.py` defines the metric;
this module is the harness that actually produces it for a run, and its whole job is to
guarantee that the two terms cover the SAME tokens.

**Why that needs enforcing rather than asserting.** `CE_model` is accumulated by iterating
fixed blocks and dropping any final partial batch: the first token of each block is an input
with no target, and the tail that does not fill a block is skipped. If the unigram term were
instead computed over the whole evaluation array -- the obvious thing to write -- the two
denominators would differ by a fraction of a percent, `L_u` would be quietly wrong, and
nothing downstream would catch it, because a plausible number would still appear. So both
terms consume ONE shared block plan and the SAME extracted target array. Divergence is made
impossible by construction, not forbidden by comment.

**Conventions fixed by amendment A3** (see `AMENDMENTS.md`):

* Unigram is add-1 (Laplace) over the full vocabulary, fitted on the TRAINING split only --
  specifically on the prefix that run actually consumed, not the whole train array. Fitting
  on more text than the model saw would give the baseline information the model did not have.
* Fitting on evaluation text at all would leak the evaluation distribution into the
  normaliser and make `L_u` optimistic in a way no downstream check would catch.

**BPB byte count.** BPB divides total NLL by the exact UTF-8 byte count of the ORIGINAL
text, never by a figure derived from average fertility. Here the bytes are obtained by
decoding exactly the scored target ids, so the numerator and denominator cover the same
tokens by construction. Note one asymmetry, inherited from the EOS packing convention (A4):
the NLL includes the model's cost of predicting `<eos>`, while `<eos>` contributes no text
bytes. That is standard for packed pretraining and is ~0.04-0.09% of tokens; it is recorded
here rather than hidden.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import model as M

LN2 = math.log(2.0)


# --- the shared block plan ------------------------------------------------------------


@dataclass(frozen=True)
class BlockPlan:
    """Exactly which positions get scored. Consumed by BOTH metric terms."""

    n_tokens: int
    block_size: int
    batch: int
    n_seq: int

    @property
    def n_scored(self) -> int:
        return self.n_seq * self.block_size

    @property
    def n_dropped(self) -> int:
        """Tokens in the array that no term scores -- reported, never silently absorbed."""
        return self.n_tokens - (self.n_scored + 1) if self.n_seq else self.n_tokens


def plan_blocks(n_tokens: int, block_size: int, batch: int) -> BlockPlan:
    n_seq = (n_tokens - 1) // block_size
    if n_seq == 0:
        raise ValueError(
            f"evaluation slice of {n_tokens} tokens is shorter than one block of {block_size}"
        )
    # A partial final batch is dropped, so the plan must round down to whole batches to
    # stay identical to what the model loop consumes.
    n_seq = (n_seq // batch) * batch if n_seq >= batch else n_seq
    return BlockPlan(n_tokens=n_tokens, block_size=block_size, batch=batch, n_seq=n_seq)


def scored_targets(tokens: np.ndarray, plan: BlockPlan) -> np.ndarray:
    """The exact target ids whose NLL is summed. Both metric terms use this array."""
    end = plan.n_seq * plan.block_size + 1
    return np.asarray(tokens[1:end], dtype=np.int64)


# --- the unigram baseline -------------------------------------------------------------


def _accumulate(counts: np.ndarray, block: np.ndarray, vocab_size: int) -> int:
    block = np.asarray(block, dtype=np.int64)
    if not block.size:
        return 0
    # Checked explicitly: an id >= vocab_size makes bincount return a LONGER array, so
    # without this the failure surfaces as an opaque numpy broadcast error instead of
    # naming the actual problem.
    if block.max() >= vocab_size or block.min() < 0:
        raise ValueError(
            f"training tokens outside [0,{vocab_size}): "
            f"saw [{int(block.min())}, {int(block.max())}]"
        )
    counts += np.bincount(block, minlength=vocab_size)
    return int(block.size)


def consumed_target_blocks(
    train_tokens: np.ndarray, block_size: int, order: np.ndarray, n_sequences: int
):
    """The TARGET ids of the sequences a run actually trained on, in consumption order.

    Sequence `k` spans `tokens[k*B : k*B + B + 1]`, so its targets are `[k*B+1, k*B+B+1)`.
    Targets rather than inputs, because the model's loss is over targets and the evaluation
    unigram term is over eval targets -- the two sides must count the same kind of thing.
    """
    B = block_size
    for k in np.asarray(order[:n_sequences], dtype=np.int64):
        s = int(k) * B
        yield train_tokens[s + 1 : s + B + 1]


def unigram_logp(
    train_tokens: np.ndarray,
    vocab_size: int,
    consumed: int | None = None,
    alpha: float = 1.0,
    order: np.ndarray | None = None,
    block_size: int | None = None,
) -> np.ndarray:
    """Add-alpha unigram log-probabilities over [0, V), fitted on what the run actually read.

    Two regimes, and picking the wrong one silently fits the baseline on text the model
    never saw:

    * `order=None` -- sequential consumption. The consumed set is the prefix
      `train_tokens[:consumed]`, which is what the original implementation assumed.
    * `order` given -- **seed-permuted consumption**, which is now the confirmatory case.
      A seed permutes sequence order, so the consumed set is a SCATTERED subset, not a
      prefix. Fitting on a prefix here would fit the baseline on text the model never read
      and, worse, on a set that is identical across seeds while the model's set is not --
      quietly removing a real component of seed variance from `L_u`, which is exactly the
      quantity Stage B.7 exists to estimate.

    `consumed` is a token count; with `order` it is converted to whole sequences.
    """
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0

    if order is None:
        end = len(train_tokens) if consumed is None else min(consumed, len(train_tokens))
        step = 1 << 26
        for s in range(0, end, step):
            total += _accumulate(counts, train_tokens[s : min(s + step, end)], vocab_size)
    else:
        if block_size is None:
            raise ValueError("block_size is required when fitting on a permuted order")
        n_seq = len(order) if consumed is None else min(consumed // block_size, len(order))
        for block in consumed_target_blocks(train_tokens, block_size, order, n_seq):
            total += _accumulate(counts, block, vocab_size)

    return np.log((counts + alpha) / (total + alpha * vocab_size))


# --- the model term -------------------------------------------------------------------


@torch.no_grad()
def model_nll_nats(
    model: M.Transformer,
    tokens: np.ndarray,
    plan: BlockPlan,
    chunk_size: int = 4096,
    device: str = "cuda",
) -> float:
    """Total (not mean) model NLL in nats over exactly `plan`'s scored positions."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    total = 0.0
    for i in range(0, plan.n_seq, plan.batch):
        b = min(plan.batch, plan.n_seq - i)
        need = b * plan.block_size + 1
        buf = tokens[i * plan.block_size : i * plan.block_size + need]
        x = torch.from_numpy(np.asarray(buf[:-1], dtype=np.int64)).view(b, plan.block_size).to(dev)
        y = torch.from_numpy(np.asarray(buf[1:], dtype=np.int64)).view(b, plan.block_size).to(dev)
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model.loss(x, y, chunk_size=chunk_size, reduction="sum")
        else:
            s = model.loss(x, y, chunk_size=chunk_size, reduction="sum")
        total += float(s)
    return total


# --- exact bytes ----------------------------------------------------------------------


def exact_bytes(target_ids: np.ndarray, tokenizer, cache: Path | None = None) -> int:
    """UTF-8 byte count of the text the scored tokens represent.

    Decoded from the ids themselves rather than read off the manifest, because the manifest
    covers the whole array while the plan scores a prefix of it. Cached, since it depends
    only on (vocabulary, eval split, n_scored) and not on the model.
    """
    if cache is not None and cache.exists():
        rec = json.loads(cache.read_text())
        if rec.get("n_scored") == int(len(target_ids)):
            return int(rec["n_bytes"])
    text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=True)
    n = len(text.encode("utf-8"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"n_scored": int(len(target_ids)), "n_bytes": n}))
    return n


# --- the result -----------------------------------------------------------------------


@dataclass
class EvalResult:
    n_scored: int
    n_dropped: int
    n_bytes: int
    model_nll_nats: float
    unigram_nll_nats: float
    vocab_size: int
    unigram_fit_tokens: int
    alpha: float

    @property
    def ce_model(self) -> float:
        return self.model_nll_nats / self.n_scored

    @property
    def h_unigram(self) -> float:
        return self.unigram_nll_nats / self.n_scored

    @property
    def l_u(self) -> float:
        """PRIMARY. Negative when the model beats unigram; MORE NEGATIVE IS BETTER."""
        return self.ce_model - self.h_unigram

    @property
    def bpb(self) -> float:
        return self.model_nll_nats / (self.n_bytes * LN2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(ce_model=self.ce_model, h_unigram=self.h_unigram, l_u=self.l_u, bpb=self.bpb)
        return d


def evaluate_run(
    model: M.Transformer,
    eval_tokens: np.ndarray,
    train_tokens: np.ndarray,
    vocab_size: int,
    tokenizer,
    consumed_train_tokens: int | None = None,
    block_size: int = 2048,
    batch: int = 4,
    chunk_size: int = 4096,
    device: str = "cuda",
    bytes_cache: Path | None = None,
    alpha: float = 1.0,
    train_order: np.ndarray | None = None,
) -> EvalResult:
    """Produce `L_u` and BPB for one trained model, both terms over identical positions.

    `train_order` is the run's consumed sequence order (`TokenStream.consumed_order`). It
    must be passed for any seed-permuted run, or the unigram baseline is fitted on a
    contiguous prefix the model never read. See `unigram_logp`.
    """
    plan = plan_blocks(len(eval_tokens), block_size, batch)
    targets = scored_targets(eval_tokens, plan)
    if len(targets) != plan.n_scored:
        raise AssertionError(
            f"target extraction disagrees with the plan: {len(targets)} vs {plan.n_scored}"
        )

    logp = unigram_logp(
        train_tokens, vocab_size, consumed_train_tokens, alpha,
        order=train_order, block_size=block_size,
    )
    uni_nll = float(-logp[targets].sum())
    mdl_nll = model_nll_nats(model, eval_tokens, plan, chunk_size, device)

    if train_order is None:
        fit_n = len(train_tokens) if consumed_train_tokens is None else min(
            consumed_train_tokens, len(train_tokens)
        )
    else:
        n_seq = (len(train_order) if consumed_train_tokens is None
                 else min(consumed_train_tokens // block_size, len(train_order)))
        fit_n = n_seq * block_size
    return EvalResult(
        n_scored=plan.n_scored,
        n_dropped=plan.n_dropped,
        n_bytes=exact_bytes(targets, tokenizer, bytes_cache),
        model_nll_nats=mdl_nll,
        unigram_nll_nats=uni_nll,
        vocab_size=vocab_size,
        unigram_fit_tokens=fit_n,
        alpha=alpha,
    )
