"""Faithful port of Tao et al.'s Approach-1 IsoFLOP pipeline.

Ported from `reference/approach1_isoflops.py` + `reference/utils.py`. This exists to
close the Stage-B reference-cross-validation gate: a correct port must recover the three
constants that the reference itself ships in `optimal_Nv_predict.py`:

    K1 (Nnv) = -2.4846510161625193
    K2 (Nv)  = -1.589031299255507
    alpha2   =  0.4163622634135234

An earlier in-house estimator (per-family bowl argmin, then OLS on log-log) recovered
alpha ~= 0.766 against a published 0.8354 -- an 8.3% gap. The cause was structural, not
numerical: their estimator is a POOLED RUNNING-MINIMUM PARETO FRONTIER over original plus
interpolated points, fit with Huber loss in log space and with the Nnv/H exponents held
FIXED at 0.5. It is not a per-budget argmin. Reproducing the published numbers therefore
requires reproducing that procedure, not a reasonable alternative to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize
from scipy.special import huber

from . import reference as ref

# Shape of the released grid: 6 families x 10 vocabularies x 20 budgets = 1200 rows.
NUM_MODEL, NUM_V, NUM_EVAL = 6, 10, 20

#: max_D_list from reference/utils.py, ordered to match the family axis of the reshape.
MAX_D_LIST = [1.2e9, 3.0e9, 5.3e9, 11.7e9, 27.7e9, 54.8e9]

HUBER_DELTA = 0.001


def generate_interpolation_log(values: np.ndarray, num: int = 8, var: float = 0.0) -> np.ndarray:
    return np.logspace(
        np.log10(values.min() * (1 - var)), np.log10(values.max() * (1 + var)), num
    )


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    errors = actual - predicted
    return float(np.mean(np.square(errors)) / np.mean(actual) ** 2)


def interpolate(
    Nnv_data: np.ndarray,
    H_data: np.ndarray,
    V_data: np.ndarray,
    L_values: np.ndarray,
    num_model: int = NUM_MODEL,
    num_v: int = NUM_V,
    num_eval: int = NUM_EVAL,
):
    """Exact port of utils.interpolate.

    Two passes, and the first one is easy to misread:

    Pass 1 interpolates each vocabulary's loss surface over (Nnv, H) with cubic
    `griddata`, but evaluates it along a 50-point DIAGONAL -- `zip(new_Nnv, new_H)` --
    not on a mesh. That diagonal is the compute-optimal ray, since Nnv and H both scale
    as sqrt(C).

    Pass 2 interpolates across V, quadratically, using ONLY the last evaluation of each
    (family, vocabulary) -- i.e. the maximum-H point -- and re-derives H from
    `max_D_list` via D_to_H.
    """
    r_Nnv = np.reshape(np.asarray(Nnv_data), (num_model, num_v, num_eval))
    r_H = np.reshape(np.asarray(H_data), (num_model, num_v, num_eval))
    r_V = np.reshape(np.asarray(V_data), (num_model, num_v, num_eval))
    r_L = np.reshape(np.asarray(L_values), (num_model, num_v, num_eval))

    i_Nnv: list[float] = []
    i_H: list[float] = []
    i_V: list[float] = []
    i_flops: list[float] = []
    i_loss: list[float] = []

    new_Nnv = generate_interpolation_log(np.unique(Nnv_data), 50)
    new_H = generate_interpolation_log(np.unique(H_data), 50)

    for vid in range(num_v):
        cur_V = r_V[0, vid, 0]
        pts = np.array(list(zip(r_Nnv[:, vid, :].ravel(), r_H[:, vid, :].ravel())))
        vals = r_L[:, vid, :].ravel()
        new_pts = np.array(list(zip(new_Nnv, new_H)))
        new_L = griddata(pts, vals, new_pts, method="cubic")
        for nnv, h, l in zip(new_Nnv, new_H, new_L):
            if np.isnan(l):
                continue
            i_Nnv.append(nnv)
            i_H.append(h)
            i_V.append(cur_V)
            i_flops.append(ref.func_flops(nnv, h, cur_V))
            i_loss.append(l)

    new_V = generate_interpolation_log(np.unique(V_data), 20)
    for mid in range(num_model):
        cur_Nnv = r_Nnv[mid, 0, 0]
        f = interp1d(
            r_V[mid, :, 0], r_L[mid, :, -1], kind="quadratic", fill_value="extrapolate"
        )
        new_L = f(new_V)
        H_of_V = np.array([ref.D_to_H(MAX_D_LIST[mid], V=v) for v in new_V])
        for v, h, l in zip(new_V, H_of_V, new_L):
            if np.isnan(l):
                continue
            i_Nnv.append(cur_Nnv)
            i_H.append(h)
            i_V.append(v)
            i_flops.append(ref.func_flops(cur_Nnv, h, v))
            i_loss.append(l)

    return (
        np.array(i_Nnv), np.array(i_H), np.array(i_V),
        np.array(i_flops), np.array(i_loss),
    )


def pareto_frontier(flops: np.ndarray, loss: np.ndarray, length_bin: int = 1):
    """Running-minimum selection over FLOPs-sorted points.

    With `length_bin = 1` this is exactly: walk up the FLOPs axis and keep a point only
    if its loss is strictly below every point kept so far. It is a monotone frontier,
    NOT a per-budget argmin -- which is the difference that mattered.
    """
    selected: list[int] = []
    pivot: float | None = None
    for i in range(0, len(flops), length_bin):
        j = i + int(np.argsort(loss[i : i + length_bin])[0])
        if pivot is None or loss[j] < pivot:
            pivot = loss[j]
            selected.append(j)
    return np.array(selected)


@dataclass
class IsoFlopFit:
    K_nnv: float
    K_nv: float
    alpha_nv: float
    K_h: float
    mse_nnv: float
    mse_nv: float
    mse_h: float
    n_frontier: int


def _fit_fixed_half(y_log: np.ndarray, flops: np.ndarray) -> tuple[float, float]:
    """Fit y = exp(K) * C^0.5 (exponent FIXED), Huber loss in log space."""
    def obj(params):
        pred = params[0] + 0.5 * np.log(flops)
        return np.sum(huber(HUBER_DELTA, pred - y_log))

    best_mse, best = float("inf"), None
    for init_K in np.linspace(-20, 15, 20):
        res = minimize(obj, [init_K], method="L-BFGS-B")
        pred = res.x[0] + 0.5 * np.log(flops)
        m = relative_mse(y_log, pred)
        if m < best_mse:
            best_mse, best = m, float(res.x[0])
    return best, best_mse


def _fit_free_alpha(y_log: np.ndarray, flops: np.ndarray) -> tuple[float, float, float]:
    """Fit y = exp(K) * C^alpha (both free), Huber loss in log space."""
    def obj(params):
        K, alpha = params
        pred = K + alpha * np.log(flops)
        return np.sum(huber(HUBER_DELTA, pred - y_log))

    best_mse, best = float("inf"), None
    for init_K in np.linspace(-20, 15, 20):
        for init_alpha in np.linspace(0, 1, 20):
            res = minimize(obj, [init_K, init_alpha], method="L-BFGS-B")
            pred = res.x[0] + res.x[1] * np.log(flops)
            m = relative_mse(y_log, pred)
            if m < best_mse:
                best_mse, best = m, (float(res.x[0]), float(res.x[1]))
    return best[0], best[1], best_mse


def run(df: pd.DataFrame, use_interpolation: bool = True) -> IsoFlopFit:
    V = df["vocab_size"].to_numpy(float)
    H = df["num_characters"].to_numpy(float)
    Nnv = df["Non_vocab_parameters"].to_numpy(float)
    flops = df["FLOPs"].to_numpy(float)
    L = df["Lossu"].to_numpy(float)

    if use_interpolation:
        iN, iH, iV, iF, iL = interpolate(Nnv, H, V, L)
        V = np.concatenate([V, iV])
        H = np.concatenate([H, iH])
        Nnv = np.concatenate([Nnv, iN])
        flops = np.concatenate([flops, iF])
        L = np.concatenate([L, iL])

    order = np.argsort(flops)
    flops, V, H, Nnv, L = flops[order], V[order], H[order], Nnv[order], L[order]

    sel = pareto_frontier(flops, L)
    f_opt, V_opt, H_opt, Nnv_opt = flops[sel], V[sel], H[sel], Nnv[sel]
    Nv_opt = np.array([v * ref.Nnv_to_d(n) for v, n in zip(V_opt, Nnv_opt)])

    K_nnv, mse_nnv = _fit_fixed_half(np.log(Nnv_opt), f_opt)
    K_h, mse_h = _fit_fixed_half(np.log(H_opt), f_opt)
    K_nv, alpha_nv, mse_nv = _fit_free_alpha(np.log(Nv_opt), f_opt)

    return IsoFlopFit(
        K_nnv=K_nnv, K_nv=K_nv, alpha_nv=alpha_nv, K_h=K_h,
        mse_nnv=mse_nnv, mse_nv=mse_nv, mse_h=mse_h, n_frontier=len(sel),
    )
