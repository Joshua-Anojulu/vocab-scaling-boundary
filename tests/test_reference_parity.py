"""Parity tests against Tao et al.'s published artifacts.

These are the gate described in the plan as "reference cross-validation". If any of
these fail, the instrument is not measuring what the hypothesis is about and no
downstream number can be trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import reference as ref  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "reference" / "exp_data.csv"

# All SIX families present in exp_data.csv (1200 rows = 6 x 10 vocabs x 20 budgets).
# An earlier draft listed only five and omitted 682M; the family identity below
# predicted 631,668,352 before that row was known to be present, which is the reason
# to trust the identity rather than a transcribed list.
PUBLISHED_NNV = {
    33_222_784.0,
    84_834_176.0,
    150_834_176.0,
    316_445_568.0,
    631_668_352.0,
    1_129_891_136.0,
}


@pytest.fixture(scope="module")
def exp_data() -> pd.DataFrame:
    if not CSV.exists():
        pytest.skip(f"{CSV} not downloaded")
    return pd.read_csv(CSV)


# --- the load-bearing test ---------------------------------------------------------


def test_func_flops_reproduces_every_published_row(exp_data: pd.DataFrame) -> None:
    """C = 6*(Nnv + V*d)*H*f(V) must reproduce the published FLOPs column exactly.

    This is the single most important parity check: it validates the fertility
    polynomial, the width lookup, and the FLOP convention simultaneously.
    """
    computed = np.array(
        [
            ref.func_flops(r.Non_vocab_parameters, r.num_characters, r.vocab_size)
            for r in exp_data.itertuples()
        ]
    )
    published = exp_data["FLOPs"].to_numpy()
    rel = np.abs(computed - published) / published
    assert rel.max() < 1e-12, (
        f"max relative error {rel.max():.3e} over {len(rel)} rows; "
        f"worst row {int(np.argmax(rel))}"
    )


def test_csv_shape_and_columns(exp_data: pd.DataFrame) -> None:
    assert list(exp_data.columns) == [
        "vocab_size", "embed_dim", "num_characters",
        "Non_vocab_parameters", "FLOPs", "Lossu",
    ]
    assert len(exp_data) == 1200


def test_lossu_is_negative(exp_data: pd.DataFrame) -> None:
    """L_u < 0 whenever the model beats the unigram baseline; more negative is better.

    The plan asserts this sign convention because it is easy to invert by accident.
    """
    assert (exp_data["Lossu"] < 0).all()


# --- accounting identity -----------------------------------------------------------


def test_family_nnv_identity_reproduces_published_values() -> None:
    """Nnv = nominal_total - 2*16384*d, for every family present in the released data."""
    derived = {
        ref.family_Nnv(total, d)
        for name, (total, d) in ref.REFERENCE_FAMILIES.items()
        # all six families appear in exp_data.csv
    }
    assert PUBLISHED_NNV <= derived


def test_nnv_excludes_both_embedding_tables(exp_data: pd.DataFrame) -> None:
    """Nnv must not vary with vocab_size.

    Empirical proof that Nnv excludes the embedding tables entirely rather than
    including the input side: every distinct Nnv is paired with the FULL vocabulary
    grid, so Nnv cannot be a function of V.

    Deliberately NOT grouped by embed_dim: the reference families 110M and 176M share
    d=768 with different Nnv, so "one Nnv per width" is false even though the
    accounting identity holds. That mistake failed this test on the first run.
    """
    all_vocabs = set(exp_data["vocab_size"].unique())
    assert len(all_vocabs) == 10

    for nnv, grp in exp_data.groupby("Non_vocab_parameters"):
        assert set(grp["vocab_size"].unique()) == all_vocabs, (
            f"Nnv={nnv:,.0f} is not paired with the full vocabulary grid"
        )
        assert grp["embed_dim"].nunique() == 1, f"Nnv={nnv:,.0f} spans multiple widths"

    # Six families, six distinct Nnv -- not 6 x 10.
    assert exp_data["Non_vocab_parameters"].nunique() == 6
    assert set(exp_data["Non_vocab_parameters"].unique()) == PUBLISHED_NNV


def test_two_families_share_a_width(exp_data: pd.DataFrame) -> None:
    """d=768 carries two distinct Nnv (the 110M and 176M families).

    Recorded as a test because it is the concrete evidence that the reference family
    is not a function of width, and therefore that no sub-50M width rule can be
    "recovered" from it -- ours is an admitted new extrapolation.
    """
    at_768 = exp_data[exp_data["embed_dim"] == 768]["Non_vocab_parameters"].unique()
    assert len(at_768) == 2
    assert set(at_768) == {84_834_176.0, 150_834_176.0}


# --- predictors --------------------------------------------------------------------


def test_a2_anchor_is_exact_at_the_reference_point() -> None:
    """A2 is anchored at Nnv = 33e6 exactly, NOT at the realized 33,222,784."""
    assert ref.approach2_derivative(ref.A2_ANCHOR_NNV) == ref.A2_ANCHOR_NV
    # At the realized anchor it is close but not equal -- the plan says so explicitly.
    realized = ref.approach2_derivative(33_222_784)
    assert realized != ref.A2_ANCHOR_NV
    assert abs(realized / ref.A2_ANCHOR_NV - 1) < 0.01


def test_a2_uses_truncation_not_rounding() -> None:
    """Faithful to `int(...)` in the reference source."""
    raw = ref.A2_ANCHOR_NV * (40_000_000 / ref.A2_ANCHOR_NNV) ** ref.A2_ALPHA
    assert ref.approach2_derivative(40_000_000) == int(raw)


def test_flops_nnv_roundtrip() -> None:
    for nnv in (2e6, 33_222_784, 1e9):
        assert ref.flops_to_Nnvopt(ref.Nnvopt_to_flops(nnv)) == pytest.approx(nnv, rel=1e-9)


def test_all_three_predictors_agree_within_an_order_of_magnitude() -> None:
    """Sanity, not parity: the three approaches should not wildly disagree."""
    for nnv in (33_222_784, 1e8, 7e9):
        preds = [
            ref.approach1_isoflops(nnv),
            ref.approach2_derivative(nnv),
            ref.approach3_isoloss(nnv),
        ]
        assert min(preds) > 0
        assert max(preds) / min(preds) < 10, f"Nnv={nnv}: {preds}"


# --- the infeasibility that forces a new architecture rule -------------------------


def test_reference_width_lookup_is_flat_below_50M() -> None:
    for nnv in (1e6, 2e6, 8e6, 16e6, 33_222_784, 50_000_000):
        assert ref.Nnv_to_d(nnv) == 512.0


def test_d512_block_exceeds_a_2M_budget() -> None:
    """One d=512 transformer block costs more than the entire 2M Nnv target.

    This is why the study cannot use Tao's architecture lookup below ~33M, and why
    the sub-50M width rule is an admitted new extrapolation rather than theirs.
    """
    one_block = 12 * 512**2
    assert one_block == 3_145_728
    assert one_block > 2e6
