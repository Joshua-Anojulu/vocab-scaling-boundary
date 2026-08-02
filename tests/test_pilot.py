"""Stage B.7 runner tests.

The load-bearing property is not that the runner works -- it is that a run violating
amendment A6 is not reachable through it. The defect A6 exists to fix was a contract held
by accident, so a runner that merely *happens* to pass the right arguments would rebuild it.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pilot as P  # noqa: E402
from src import train as T  # noqa: E402


# --- the plan ------------------------------------------------------------------------


def test_pilot_is_six_cells_five_at_C_and_one_at_1p1C() -> None:
    cells = P.pilot_cells()
    assert len(cells) == 6
    assert sum(c.arm == "C" for c in cells) == 5
    assert sum(c.arm == "1.1C" for c in cells) == 1


def test_the_m2_arm_pairs_with_a_C_arm_at_the_same_vocabulary() -> None:
    """An unpaired 1.1C arm would make D undefined."""
    cells = P.pilot_cells()
    m2 = next(c for c in cells if c.arm == "1.1C")
    partners = [c for c in cells if c.arm == "C" and c.vocab_size == m2.vocab_size]
    assert len(partners) == 1
    assert m2.budget_flops == pytest.approx(1.1 * partners[0].budget_flops)


def test_budgets_are_derived_not_transcribed() -> None:
    """A hardcoded C is how a 4.6e-06 rounding error entered this project once already."""
    src = inspect.getsource(P)
    assert "1.03084e16" not in src
    assert "ref.Nnvopt_to_flops" in src


def test_every_cell_stays_within_its_isoflop_budget() -> None:
    for c in P.pilot_cells():
        planned = (c.target_tokens // T.BLOCK_SIZE) * T.BLOCK_SIZE
        assert planned <= c.target_tokens
        assert c.target_tokens - planned < T.BLOCK_SIZE


# --- the contract is unreachable to violate ------------------------------------------


def test_runner_never_exposes_either_escape_hatch() -> None:
    """Both flags are legitimate for smoke runs and fatal for study runs."""
    params = inspect.signature(P.run_cell).parameters
    assert "unseeded_order_ok" not in params
    assert "sequential_order_ok" not in params
    # The docstring names both flags to explain their absence, which is the point; what
    # must not appear is either being SET.
    body = inspect.getsource(P.run_cell).replace(P.run_cell.__doc__ or "", "")
    assert "unseeded_order_ok" not in body
    assert "sequential_order_ok" not in body


def test_runner_builds_the_stream_over_the_whole_array_not_a_budget_slice() -> None:
    """Slicing to budget shrinks the permutation domain and silently breaks nesting."""
    src = inspect.getsource(P.run_cell)
    assert "T.TokenStream(np.asarray(train_tokens), block_size, order_seed=seed)" in src


def test_runner_passes_the_consumed_order_to_the_evaluator() -> None:
    """Without it H_unigram is fitted on a prefix the model never read."""
    assert "train_order=stream.consumed_order" in inspect.getsource(P.run_cell)


# --- the audit rule -------------------------------------------------------------------


def _rec(v: int, seed: int, arm: str, consumed: int, order_digest: str = "aa",
         tokens_digest: str = "tt", stream_sequences: int = 1000) -> dict:
    return {
        "vocab_size": v, "seed": seed, "arm": arm,
        "train_order_seed": seed, "train_stream_sequences": stream_sequences,
        "train_tokens_digest": tokens_digest, "train_order_digest": order_digest,
        "train_sequences_consumed": consumed,
    }


def test_audit_accepts_a_correctly_nested_pair() -> None:
    out = P.audit_nesting([_rec(3200, 0, "C", 500), _rec(3200, 0, "1.1C", 550)])
    assert len(out) == 1 and out[0]["nested"]
    assert out[0]["sequences_consumed"] == [500, 550]


def test_audit_rejects_a_shrunken_permutation_domain() -> None:
    """The realistic break: a caller sliced the array before building the stream."""
    out = P.audit_nesting([
        _rec(3200, 0, "C", 500, order_digest="aa", stream_sequences=1000),
        _rec(3200, 0, "1.1C", 550, order_digest="bb", stream_sequences=900),
    ])
    assert not out[0]["nested"]
    assert not out[0]["train_order_digest"]
    assert not out[0]["train_stream_sequences"]


def test_audit_rejects_a_different_corpus_of_the_same_length() -> None:
    out = P.audit_nesting([
        _rec(3200, 0, "C", 500, tokens_digest="tt"),
        _rec(3200, 0, "1.1C", 550, tokens_digest="zz"),
    ])
    assert not out[0]["nested"]


def test_audit_skips_vocabularies_with_only_one_arm() -> None:
    """Only V_run has both; the other four cannot be checked for pairing and are not."""
    assert P.audit_nesting([_rec(768, 0, "C", 500)]) == []


# --- the whole path, scaled down ------------------------------------------------------


def test_run_cell_executes_end_to_end_and_records_the_witness(monkeypatch, tmp_path) -> None:
    """Planning tests cannot catch a wiring mismatch; only running the path can.

    Scaled to a toy vocabulary and block size so it runs on CPU in seconds, with the token
    splits and tokenizer stubbed. Everything else is the real code path.
    """
    import numpy as np
    from src import evaluate as E

    V, block = 32, 8
    rng = np.random.default_rng(0)
    train = rng.integers(0, V, size=block * 200 + 1, dtype=np.uint16)
    val = rng.integers(0, V, size=block * 20 + 1, dtype=np.uint16)

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    monkeypatch.setattr(P, "_load_split",
                        lambda v, split: train if split == "train" else val)
    monkeypatch.setattr(P.tk, "load", lambda path: _Tok())
    monkeypatch.setattr(T, "BLOCK_SIZE", block)

    cell = P.Cell(vocab_size=V, arm="C", budget_flops=1.0,
                  target_tokens=block * 40, d=32, n_layer=1, n_head=2, d_ffn=64, nnv=1000)

    rec = P.run_cell(cell, seed=3, micro_batch=2, grad_accum=2, device="cpu",
                     block_size=block, eval_batch=2, log_dir=tmp_path)

    # The contract travelled into the record, which is what the audit reads.
    assert rec["train_order_seed"] == 3
    assert rec["train_stream_sequences"] == 200          # WHOLE array, not the 40-seq budget
    assert rec["train_sequences_consumed"] == 40
    assert rec["train_tokens_digest"] and rec["train_order_digest"]
    assert rec["train_consumed_tokens"] == 40 * block
    assert np.isfinite(rec["eval_l_u"]) and np.isfinite(rec["eval_bpb"])


def test_run_cell_fits_the_baseline_on_the_consumed_set_not_a_prefix(monkeypatch, tmp_path):
    """The round-2 defect, pinned at the runner level rather than the function level."""
    import numpy as np
    from src import evaluate as E

    V, block = 4, 8
    # First half all token 0, second half all token 1: a prefix fit and a permuted fit
    # cannot agree, so a regression here changes the number rather than hiding.
    train = np.array([0] * (block * 100) + [1] * (block * 100 + 1), dtype=np.uint16)
    val = np.array([1] * (block * 10 + 1), dtype=np.uint16)

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    monkeypatch.setattr(P, "_load_split",
                        lambda v, split: train if split == "train" else val)
    monkeypatch.setattr(P.tk, "load", lambda path: _Tok())

    cell = P.Cell(vocab_size=V, arm="C", budget_flops=1.0,
                  target_tokens=block * 60, d=32, n_layer=1, n_head=2, d_ffn=64, nnv=1000)
    rec = P.run_cell(cell, seed=1, micro_batch=2, grad_accum=2, device="cpu",
                     block_size=block, eval_batch=2, log_dir=tmp_path)

    # A permuted 60-sequence sample of a half-0/half-1 corpus sees both tokens, so the
    # baseline on an all-1 eval set is far from the prefix fit's near-certainty.
    prefix_logp = E.unigram_logp(train, V, consumed=block * 60)
    prefix_h = float(-prefix_logp[val[1:]].sum() / (len(val) - 1))
    assert abs(rec["eval_h_unigram"] - prefix_h) > 1.0, (
        "baseline matches the prefix fit; the consumed order is not reaching the evaluator"
    )


# --- the effective batch is a recipe parameter, not a hardware one --------------------


def test_global_batch_matches_the_reference() -> None:
    """reference/tinyllama_pretrain.py:37 -- and learning_rate 4e-4 is tuned to it."""
    assert P.GLOBAL_BATCH_SEQUENCES == 512


def test_grad_accum_is_derived_so_the_effective_batch_is_exactly_the_recipe() -> None:
    for mb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        assert mb * P.grad_accum_for(mb) == P.GLOBAL_BATCH_SEQUENCES


def test_a_micro_batch_that_would_change_the_effective_batch_is_refused() -> None:
    """Silently landing NEAR 512 would be a different recipe with the same label."""
    for mb in (3, 5, 6, 7, 100, 513, 0, -4):
        with pytest.raises(ValueError, match="does not divide the global batch"):
            P.grad_accum_for(mb)


def test_pilot_step_counts_sit_inside_taos_own_range() -> None:
    """The faithfulness check that decided this: Tao at 33M nnv ran 57-1144 steps.

    Recomputed from their released data rather than quoted, since a step count outside
    their range would mean "the same recipe" was a label rather than a fact.
    """
    import pandas as pd
    from src import reference as ref

    data = Path(__file__).resolve().parents[1] / "reference" / "exp_data.csv"
    if not data.exists():
        pytest.skip("reference/exp_data.csv not present (regenerable, gitignored)")
    d = pd.read_csv(data)
    d["steps"] = (d.num_characters * d.vocab_size.map(ref.fertility)
                  / (P.GLOBAL_BATCH_SEQUENCES * T.BLOCK_SIZE))
    smallest = d[d.Non_vocab_parameters < 3.4e7]
    lo, hi = smallest.steps.min(), smallest.steps.max()

    ours = [c.target_tokens // T.BLOCK_SIZE // P.GLOBAL_BATCH_SEQUENCES
            for c in P.pilot_cells()]
    assert all(lo <= s <= hi for s in ours), f"ours {sorted(ours)} outside Tao's [{lo:.0f},{hi:.0f}]"
