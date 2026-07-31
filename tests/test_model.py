"""Instrument tests for the transformer.

The load-bearing one is parameter accounting: if `n_params_non_vocab()` does not equal
the preregistered `per_layer_params * n_layer + d`, then the realized N_nv differs from
the tabled N_nv, and V_pred = N_v/d -- the quantity under test -- is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import extension as ext, model as M  # noqa: E402

torch.manual_seed(0)


def cfg_for(target: int, vocab: int) -> tuple[M.ModelConfig, ext.Arch]:
    a = ext.arch_for(target)
    return (
        M.ModelConfig(
            vocab_size=vocab, d=a.d, n_layer=a.n_layer, n_head=a.n_head,
            d_ffn=a.d_ffn, block_size=128,
        ),
        a,
    )


# --- parameter accounting ----------------------------------------------------------


@pytest.mark.parametrize("target", ext.TARGET_NNV)
def test_non_vocab_params_match_the_preregistered_rule(target: int) -> None:
    c, a = cfg_for(target, vocab=1024)
    m = M.build(c)
    assert m.n_params_non_vocab() == a.nnv, (
        f"target {target}: model has {m.n_params_non_vocab():,} non-vocab params, "
        f"rule says {a.nnv:,}"
    )


@pytest.mark.parametrize("target", ext.TARGET_NNV)
def test_non_vocab_params_are_independent_of_vocabulary(target: int) -> None:
    """N_nv must not move when V changes -- the property proved on their released data."""
    counts = {M.build(cfg_for(target, v)[0]).n_params_non_vocab() for v in (384, 4096, 32768)}
    assert len(counts) == 1


def test_n_v_is_the_output_head_only() -> None:
    c, a = cfg_for(8_000_000, vocab=3200)
    m = M.build(c)
    assert m.n_params_vocab() == 3200 * a.d
    # Untied: the model physically holds 2*V*d, so total != N_nv + N_v.
    assert m.n_params_total() == a.nnv + 2 * 3200 * a.d
    assert m.n_params_total() != a.nnv + m.n_params_vocab()


def test_embeddings_are_untied() -> None:
    m = M.build(cfg_for(2_000_000, 512)[0])
    assert m.wte.weight is not m.lm_head.weight
    assert m.wte.weight.data_ptr() != m.lm_head.weight.data_ptr()


def test_per_layer_breakdown_is_exact() -> None:
    """4d^2 attention + 3*d*d_ffn FFN + 2d norms, with no stray biases."""
    c, a = cfg_for(4_000_000, 1024)
    blk = M.Block(c)
    attn = sum(p.numel() for p in blk.attn.parameters())
    ffn = sum(p.numel() for p in blk.ffn.parameters())
    norms = blk.attn_norm.weight.numel() + blk.ffn_norm.weight.numel()
    assert attn == 4 * c.d * c.d
    assert ffn == 3 * c.d * c.d_ffn
    assert norms == 2 * c.d
    assert attn + ffn + norms == a.per_layer
    assert all(p.ndim > 0 for p in blk.parameters())
    assert not any("bias" in n for n, _ in blk.named_parameters())


# --- numerics ----------------------------------------------------------------------


def test_chunked_loss_matches_unchunked() -> None:
    """Chunking must be an exact refactor, not an approximation."""
    c, _ = cfg_for(2_000_000, vocab=512)
    m = M.build(c).eval()
    idx = torch.randint(0, 512, (2, 64))
    tgt = torch.randint(0, 512, (2, 64))
    with torch.no_grad():
        chunked = m.loss(idx, tgt, chunk_size=7)   # deliberately not a divisor
        logits = m(idx)
        direct = F.cross_entropy(logits.flatten(0, 1).float(), tgt.flatten())
    assert torch.allclose(chunked, direct, atol=1e-5)


def test_chunk_size_does_not_change_the_loss() -> None:
    c, _ = cfg_for(2_000_000, vocab=512)
    m = M.build(c).eval()
    idx = torch.randint(0, 512, (2, 64))
    tgt = torch.randint(0, 512, (2, 64))
    with torch.no_grad():
        losses = [m.loss(idx, tgt, chunk_size=k) for k in (1, 13, 128, 4096)]
    for l in losses[1:]:
        assert torch.allclose(losses[0], l, atol=1e-5)


def test_chunked_ce_gradients_match_autograd() -> None:
    """The hand-written backward must equal what autograd computes for the naive path.

    This is the test that matters for `_ChunkedCE`: recomputing logits in backward is a
    memory optimisation, and a wrong gradient there would corrupt every trained model
    while still producing a plausible-looking loss curve.
    """
    torch.manual_seed(1)
    N, d, V = 37, 16, 23          # deliberately not divisible by the chunk size
    h = torch.randn(N, d, dtype=torch.double, requires_grad=True)
    W = torch.randn(V, d, dtype=torch.double, requires_grad=True)
    t = torch.randint(0, V, (N,))

    fused = M._ChunkedCE.apply(h, W, t, 5)
    fused.backward()
    dh_fused, dW_fused = h.grad.clone(), W.grad.clone()

    h.grad = None
    W.grad = None
    # NB: no `.float()` here. Casting the reference path to fp32 while the fused path
    # runs in fp64 makes the comparison measure the cast, not the implementation.
    naive = F.cross_entropy(h @ W.t(), t, reduction="sum")
    naive.backward()

    assert torch.allclose(fused, naive, atol=1e-8)
    assert torch.allclose(dh_fused, h.grad, atol=1e-8), "dh mismatch"
    assert torch.allclose(dW_fused, W.grad, atol=1e-8), "dW mismatch"


def test_chunked_ce_gradient_is_chunk_size_invariant() -> None:
    torch.manual_seed(2)
    N, d, V = 40, 8, 11
    t = torch.randint(0, V, (N,))
    grads = []
    for chunk in (1, 7, 40, 4096):
        h = torch.randn(N, d, dtype=torch.double, requires_grad=True)
        torch.manual_seed(3)
        W = torch.randn(V, d, dtype=torch.double, requires_grad=True)
        torch.manual_seed(4)
        h.data = torch.randn(N, d, dtype=torch.double)
        M._ChunkedCE.apply(h, W, t, chunk).backward()
        grads.append((h.grad.clone(), W.grad.clone()))
    for dh, dW in grads[1:]:
        assert torch.allclose(grads[0][0], dh, atol=1e-10)
        assert torch.allclose(grads[0][1], dW, atol=1e-10)


def test_attention_is_causal() -> None:
    """Perturbing a future token must not change an earlier position's output."""
    c, _ = cfg_for(2_000_000, vocab=64)
    m = M.build(c).eval()
    a = torch.randint(0, 64, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 64
    with torch.no_grad():
        ha, hb = m.hidden_states(a), m.hidden_states(b)
    assert torch.allclose(ha[:, :-1], hb[:, :-1], atol=1e-5)
    assert not torch.allclose(ha[:, -1], hb[:, -1], atol=1e-5)


def test_rope_preserves_norm_and_is_relative() -> None:
    """Rotation is norm-preserving, and q.k depends on relative position only."""
    hd, T = 64, 8
    cos, sin = M.build_rope_cache(T, hd, 10000.0, torch.device("cpu"))
    x = torch.randn(1, 1, T, hd)
    r = M.apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), r.norm(dim=-1), atol=1e-5)

    q = torch.randn(1, 1, T, hd)
    k = torch.randn(1, 1, T, hd)
    rq, rk = M.apply_rope(q, cos, sin), M.apply_rope(k, cos, sin)
    # positions (2,5) and (3,6) share offset 3 -> equal inner products
    same_q = q.clone(); same_k = k.clone()
    same_q[0, 0, 3] = q[0, 0, 2]; same_k[0, 0, 6] = k[0, 0, 5]
    sq, sk = M.apply_rope(same_q, cos, sin), M.apply_rope(same_k, cos, sin)
    assert torch.allclose(
        (rq[0, 0, 2] * rk[0, 0, 5]).sum(), (sq[0, 0, 3] * sk[0, 0, 6]).sum(), atol=1e-4
    )


def test_forward_shapes_and_finiteness() -> None:
    c, _ = cfg_for(2_000_000, vocab=384)
    m = M.build(c)
    idx = torch.randint(0, 384, (3, 32))
    out = m(idx)
    assert out.shape == (3, 32, 384)
    assert torch.isfinite(out).all()


def test_loss_at_init_is_near_uniform() -> None:
    """An untrained model should sit near ln(V); a wild value means broken init."""
    V = 512
    c, _ = cfg_for(2_000_000, vocab=V)
    m = M.build(c).eval()
    idx = torch.randint(0, V, (4, 64))
    tgt = torch.randint(0, V, (4, 64))
    with torch.no_grad():
        loss = m.loss(idx, tgt)
    assert abs(loss.item() - torch.log(torch.tensor(float(V))).item()) < 0.5


def test_gradients_flow_to_every_parameter() -> None:
    c, _ = cfg_for(2_000_000, vocab=256)
    m = M.build(c)
    idx = torch.randint(0, 256, (2, 32))
    tgt = torch.randint(0, 256, (2, 32))
    m.loss(idx, tgt, chunk_size=17).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, f"no/!finite grad for: {missing}"
