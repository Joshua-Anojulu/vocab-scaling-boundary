"""Stage-B reference cross-validation gate.

The A1 IsoFLOP port must recover the three constants the reference itself ships in
`optimal_Nv_predict.py`, from `exp_data.csv` alone. If this fails, the estimator is not
Tao's estimator and no confirmatory inference may be built on it.

This is the strongest available validation of the analysis pipeline: it is an exact
numerical target published by the authors, not a tolerance we chose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import isoflop, reference as ref  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "reference" / "exp_data.csv"

#: Tolerance is set by L-BFGS-B convergence, not by methodology. The observed residuals
#: are ~1e-10; 1e-7 leaves headroom for platform/BLAS variation without admitting a
#: genuinely different estimator (the in-house alternative was off by ~8e-2).
ATOL = 1e-7


@pytest.fixture(scope="module")
def fit() -> isoflop.IsoFlopFit:
    if not CSV.exists():
        pytest.skip(f"{CSV} not downloaded")
    return isoflop.run(pd.read_csv(CSV))


@pytest.fixture(scope="module")
def exp_data() -> pd.DataFrame:
    if not CSV.exists():
        pytest.skip(f"{CSV} not downloaded")
    return pd.read_csv(CSV)


def test_reshape_ordering_is_model_vocab_eval(exp_data: pd.DataFrame) -> None:
    """utils.interpolate reshapes to (6, 10, 20); the port depends on that layout."""
    import numpy as np

    v = np.reshape(exp_data["vocab_size"].to_numpy(), (6, 10, 20))
    nnv = np.reshape(exp_data["Non_vocab_parameters"].to_numpy(), (6, 10, 20))
    for m in range(6):
        assert len(set(nnv[m].ravel())) == 1, "Nnv must be constant within a family"
        for k in range(10):
            assert len(set(v[m, k, :])) == 1, "vocab must be constant along the eval axis"


def test_recovers_published_K_nnv(fit: isoflop.IsoFlopFit) -> None:
    assert fit.K_nnv == pytest.approx(ref.K_NNV_FLOPS, abs=ATOL)


def test_recovers_published_K_nv(fit: isoflop.IsoFlopFit) -> None:
    assert fit.K_nv == pytest.approx(ref.A1_LOG_COEF, abs=ATOL)


def test_recovers_published_alpha(fit: isoflop.IsoFlopFit) -> None:
    assert fit.alpha_nv == pytest.approx(ref.A1_EXPONENT, abs=ATOL)


def test_implied_nv_vs_nnv_exponent(fit: isoflop.IsoFlopFit) -> None:
    """alpha_nv / 0.5 is the exponent relating N_v to N_nv -- A1's analogue of A2's 0.8354."""
    implied = fit.alpha_nv / 0.5
    assert implied == pytest.approx(ref.A1_EXPONENT / 0.5, abs=ATOL)
    # Close to, but deliberately not equal to, A2's separately-derived exponent.
    assert abs(implied - ref.A2_ALPHA) < 0.01
    assert implied != ref.A2_ALPHA


def test_frontier_is_monotone_and_nonempty(fit: isoflop.IsoFlopFit) -> None:
    assert fit.n_frontier > 50


def test_fit_quality(fit: isoflop.IsoFlopFit) -> None:
    for m in (fit.mse_nnv, fit.mse_nv, fit.mse_h):
        assert m < 1e-2


def test_pareto_frontier_is_strictly_decreasing() -> None:
    """The selection rule keeps a point only if it beats every point kept so far."""
    import numpy as np

    flops = np.arange(10, dtype=float)
    loss = np.array([5.0, 4.0, 6.0, 3.0, 3.5, 2.0, 2.5, 1.0, 9.0, 0.5])
    sel = isoflop.pareto_frontier(flops, loss)
    kept = loss[sel]
    assert list(kept) == [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    assert all(kept[i] > kept[i + 1] for i in range(len(kept) - 1))
