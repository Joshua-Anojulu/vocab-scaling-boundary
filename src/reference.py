"""Exact ports of Tao et al. (NeurIPS 2024, arXiv:2407.13623) reference code.

Source: github.com/sail-sg/scaling-with-vocab -- `optimal_Nv_predict.py`, `utils.py`.
Vendored copies live in `reference/` alongside `exp_data.csv`.

Nothing in this module may be "improved". It exists so that the primary falsification
test is measured against Tao's own ruler, predictors and conventions. Deviations belong
in `src/extension.py`, never here. In particular the `int()` truncations below are
faithful to the reference and are NOT rounding.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import fsolve

# --- pinned constants, transcribed from the reference source -----------------------

# Nnvopt_to_flops / flops_to_Nnvopt
K_NNV_FLOPS = -2.4846510161625193

# approach1_isoflops
A1_LOG_COEF = -1.589031299255507
A1_EXPONENT = 0.4163622634135234

# approach2_derivative
A2_ANCHOR_NV = 3145728          # value at the REFERENCE point Nnv = 33_000_000 exactly,
A2_ANCHOR_NNV = 33_000_000      # NOT at the realized 33_222_784
A2_ALPHA = 0.8353974035228025

# approach3_isoloss parametric fit
A3_A1, A3_A2 = 1.8313851559554126, 0.19584238398665638
A3_B, A3_E = 2.1241123120064955, 5.5327846803337435
A3_ALPHA1, A3_ALPHA2, A3_BETA = (
    0.44660634152009615,
    0.6707374679896795,
    0.44660634152009615,
)

# fertility polynomial, used by func_flops / D_to_H / H_to_D
FERT_A, FERT_B, FERT_C = 0.00639222, -0.15811069, 1.20470122
FERT_V_CAP = 200_000

# The vocabulary at which the reference prices embedding tables when naming a model
# family. See the header comment of reference/utils.py.
V_REF_FOR_NAMING = 16384


# --- architecture lookup -----------------------------------------------------------

_D_LADDER = [
    (50_000_000, 512),
    (200_000_000, 768),
    (500_000_000, 1024),
    (1_000_000_000, 1536),
    (2_000_000_000, 2048),
    (5_000_000_000, 3200),
    (10_000_000_000, 4096),
    (20_000_000_000, 5120),
    (50_000_000_000, 6048),
    (100_000_000_000, 8192),
    (200_000_000_000, 12288),
    (500_000_000_000, 16384),
    (1_000_000_000_000, 20480),
]


def Nnv_to_d(Nnv: float) -> float:
    """Reference width lookup. Returns 512 for EVERY Nnv <= 50M.

    This is why the study needs its own sub-50M rule: a single d=512 block costs
    12*512**2 = 3.15M parameters, which already exceeds a 2M Nnv budget.
    """
    for upper, d in _D_LADDER:
        if Nnv <= upper:
            return float(d)
    return 24576.0


def fertility(V: float) -> float:
    """Tao's fitted tokens-per-character term f(V). NOT a realized fertility."""
    logv = math.log(min(V, FERT_V_CAP))
    return FERT_A * logv**2 + FERT_B * logv + FERT_C


def func_flops(Nnv: float, H: float, V: float) -> float:
    """C = 6*(Nnv + V*d)*H*f(V), with H in CHARACTERS and d from the reference ladder."""
    d = Nnv_to_d(Nnv)
    return 6 * (Nnv + V * d) * H * fertility(V)


def D_to_H(D: float, V: float = V_REF_FOR_NAMING) -> float:
    return D / fertility(V)


def H_to_D(H: float, V: float = V_REF_FOR_NAMING) -> float:
    return H * fertility(V)


# --- optimal-budget relations ------------------------------------------------------


def Nnvopt_to_flops(Nnv: float) -> float:
    """Training-optimal FLOPs budget C for a given Nnv. Exponent is (1/0.5) == 2."""
    return (Nnv / np.exp(K_NNV_FLOPS)) ** (1 / 0.5)


def flops_to_Nnvopt(flops: float) -> float:
    return np.exp(K_NNV_FLOPS) * flops**0.5


# --- the three predictors ----------------------------------------------------------


def approach1_isoflops(Nnv: float) -> int:
    flops = (Nnv / np.exp(K_NNV_FLOPS)) ** (1 / 0.5)
    return int(np.exp(A1_LOG_COEF) * flops**A1_EXPONENT)


def approach2_derivative(Nnv: float) -> int:
    """PRIMARY predictor for this study. Note int() truncation, faithful to source."""
    return int(A2_ANCHOR_NV * (Nnv / A2_ANCHOR_NNV) ** A2_ALPHA)


def approach3_isoloss(Nnv: float, flops: float | None = None) -> int:
    def dl_dv(V, Nnv_n, d_n, F):
        term3 = -A3_ALPHA2 * A3_A2 * d_n / (V * d_n) ** (A3_ALPHA2 + 1)
        u = F / (6 * (Nnv_n + V * d_n))
        du_dV = F * d_n / (6 * (Nnv_n + V * d_n) ** 2)
        term4 = A3_BETA * A3_B * du_dV / (u ** (A3_BETA + 1))
        return term3 + term4

    d = Nnv_to_d(Nnv)
    if flops is None:
        flops = Nnvopt_to_flops(Nnv)
    # reference normalization
    Nnv_n, d_n, F_n = Nnv / 1e6, d / 1e3, flops / 1e15
    V = fsolve(dl_dv, 1, args=(Nnv_n, d_n, F_n))[0]
    # The reference DE-normalizes d (d = d * 1_000) before its `return int(V*1000*d)`,
    # so the multiplier is the real width, not the normalized one. Using d_n here is a
    # silent 1000x error -- it was caught by the cross-predictor sanity test.
    return int(V * 1000 * d)


# --- family accounting identity ----------------------------------------------------


def family_Nnv(nominal_total: float, d: float, V_ref: int = V_REF_FOR_NAMING) -> float:
    """Nnv = nominal_total - 2*V_ref*d.

    Verified to reproduce all five published Non_vocab_parameters values exactly.
    The factor of 2 is the load-bearing part: Nnv excludes BOTH embedding tables,
    so total(untied) = Nnv + 2*V*d and total != Nnv + Nv.
    """
    return nominal_total - 2 * V_ref * d


# Reference model families, from utils.py's dicts. Note 110M and 176M SHARE d=768:
# the reference family varies depth at fixed width, so it is not a function of d alone.
REFERENCE_FAMILIES = {
    "50M": (50e6, 512),
    "110M": (110e6, 768),
    "176M": (176e6, 768),
    "335M": (350e6, 1024),   # model_size_dict says 350e6 despite the '335M' key
    "682M": (682e6, 1536),
    "1197M": (1197e6, 2048),
}

# Max training tokens D per family, from max_D_dict.
REFERENCE_MAX_D = {
    "50M": 1.2e9, "110M": 3.0e9, "176M": 5.3e9, "335M": 11.7e9,
    "682M": 27.7e9, "1197M": 54.8e9, "2975M": 165.5e9,
}
