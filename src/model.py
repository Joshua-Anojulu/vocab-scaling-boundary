"""Llama-style decoder, built to match the preregistered parameter accounting exactly.

Design constraints come from the plan and are not free choices:

* **Untied embeddings.** `reference/lit_gpt/model.py` defines `lm_head` as a separate
  `nn.Linear` from `wte` with no weight assignment. Tying would halve the vocabulary
  parameter cost and therefore change the quantity under study.
* **N_nv excludes BOTH embedding tables.** Verified against all six published families
  via `N_nv = nominal_total - 2*16384*d`. So `n_params_non_vocab()` here must count the
  transformer body and the final norm only, and must equal
  `extension.per_layer_params(d, d_ffn) * n_layer + d` on the nose.
* **Chunked cross-entropy is mandatory, not an optimisation.** A single [B, T, V] logit
  tensor at T=2048, V=32k is ~128 MiB in bf16 per sequence; materialising it would force
  microbatch 1 on an 8 GB card. The reference ships `fused_cross_entropy.py` for the same
  reason.
* **bf16 autocast with fp32 master weights.** Ada (sm_89) supports bf16 natively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d: int
    n_layer: int
    n_head: int
    d_ffn: int
    block_size: int = 2048
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.d % self.n_head == 0, "d must be divisible by n_head"
        return self.d // self.n_head


# --- building blocks ---------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm with a learned gain and no bias -- exactly `d` parameters."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_head, T, head_dim]. Rotates dimension pairs by position-dependent angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, : x.shape[-2], :].to(x.dtype)
    s = sin[None, None, : x.shape[-2], :].to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class Attention(nn.Module):
    """Multi-head causal attention, no biases -> exactly 4*d^2 parameters."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.k_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.v_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.o_proj = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, hd = self.cfg.n_head, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # SDPA picks a fused/flash kernel where available; is_causal applies the mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, self.cfg.d))


class SwiGLU(nn.Module):
    """Gated FFN, no biases -> exactly 3*d*d_ffn parameters."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d, cfg.d_ffn, bias=False)
        self.up_proj = nn.Linear(cfg.d, cfg.d_ffn, bias=False)
        self.down_proj = nn.Linear(cfg.d_ffn, cfg.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm residual block: 4d^2 + 3*d*d_ffn + 2d parameters."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


# --- the model ---------------------------------------------------------------------


class _ChunkedCE(torch.autograd.Function):
    """Fused linear + cross-entropy with logit recomputation in backward.

    Forward keeps only `h`, `W` and the per-chunk summed loss. Backward walks the same
    chunks, recomputes `logits = h_chunk @ W.T`, forms `softmax - onehot`, and
    accumulates `dh` and `dW` -- so no chunk's logits outlive its own iteration.

    Returns the SUM over positions; the caller divides. Cross-entropy is evaluated in
    fp32 for stability even when the surrounding autocast region is bf16.
    """

    @staticmethod
    def _acc_dtype(h: torch.Tensor) -> torch.dtype:
        """Accumulate in fp32 under bf16/fp16 autocast, but never DOWNCAST fp64.

        Hardcoding `.float()` here silently demoted double-precision inputs, which both
        broke autograd's dtype check and inflated accumulation-order differences from
        ~1e-16 to ~1e-7 -- caught by the gradient-parity tests.
        """
        return torch.float64 if h.dtype == torch.float64 else torch.float32

    @staticmethod
    def forward(ctx, h: torch.Tensor, W: torch.Tensor, t: torch.Tensor, chunk: int):
        acc = _ChunkedCE._acc_dtype(h)
        total = torch.zeros((), dtype=acc, device=h.device)
        for i in range(0, h.shape[0], chunk):
            logits = (h[i : i + chunk] @ W.t()).to(acc)
            total += F.cross_entropy(logits, t[i : i + chunk], reduction="sum")
        ctx.save_for_backward(h, W, t)
        ctx.chunk = chunk
        return total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h, W, t = ctx.saved_tensors
        chunk = ctx.chunk
        acc = _ChunkedCE._acc_dtype(h)
        dh = torch.zeros(h.shape, dtype=acc, device=h.device)
        dW = torch.zeros(W.shape, dtype=acc, device=W.device)
        g = grad_out.to(acc)
        Wa = W.to(acc)
        for i in range(0, h.shape[0], chunk):
            hc = h[i : i + chunk].to(acc)
            p = torch.softmax(hc @ Wa.t(), dim=-1)
            p.scatter_add_(
                1,
                t[i : i + chunk, None],
                torch.full((min(chunk, p.shape[0]), 1), -1.0, dtype=acc, device=p.device),
            )
            p *= g                              # scalar upstream grad
            dh[i : i + chunk] = p @ Wa
            dW += p.t() @ hc
            del p, hc
        return dh.to(h.dtype), dW.to(W.dtype), None, None


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d)
        # UNTIED on purpose. Never assign self.lm_head.weight = self.wte.weight.
        self.lm_head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)

        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.apply(self._init_weights)
        # Scaled init on residual-output projections, per GPT-2/Llama practice.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope is None or self._rope[0].device != device:
            self._rope = build_rope_cache(
                self.cfg.block_size, self.cfg.head_dim, self.cfg.rope_base, device
            )
        return self._rope

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope_cache(idx.device)
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Full logits. Only for tiny inputs -- training uses `loss()` instead."""
        return self.lm_head(self.hidden_states(idx))

    def loss(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int = 4096,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Memory-efficient cross-entropy that never materialises [B*T, V] logits.

        A naive chunked loop does NOT save memory under autograd: every chunk's logits
        stay alive because backward needs them. `_ChunkedCE` therefore recomputes each
        chunk's logits during backward and frees them immediately, so peak logit memory
        is `chunk_size * V` rather than `B*T*V`. That distinction matters here -- at
        B=8, T=2048, V=17792 the full logit tensor is ~1.2 GB in fp32, plus as much
        again for its gradient, on an 8 GB card.

        Matches an un-chunked `F.cross_entropy(..., reduction='mean')` to floating-point
        tolerance, and the result is independent of `chunk_size` (both are tested).
        """
        h = self.hidden_states(idx).flatten(0, 1)      # [B*T, d]
        t = targets.flatten()                          # [B*T]
        total = _ChunkedCE.apply(h, self.lm_head.weight, t, chunk_size)
        return total / t.numel() if reduction == "mean" else total

    # --- parameter accounting ------------------------------------------------------

    def n_params_non_vocab(self) -> int:
        """N_nv: transformer body + final norm. EXCLUDES both embedding tables.

        Must equal extension.per_layer_params(d, d_ffn) * n_layer + d.
        """
        total = sum(p.numel() for p in self.parameters())
        vocab = self.wte.weight.numel() + self.lm_head.weight.numel()
        return total - vocab

    def n_params_vocab(self) -> int:
        """N_v as Tao defines it: the OUTPUT head only, V*d.

        Note the model physically holds 2*V*d vocabulary parameters because the
        embeddings are untied, so total != N_nv + N_v.
        """
        return self.lm_head.weight.numel()

    def n_params_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(cfg: ModelConfig) -> Transformer:
    return Transformer(cfg)
