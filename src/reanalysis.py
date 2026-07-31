"""Stage B.3 -- zero-GPU re-analysis of Tao et al.'s released results.

Two jobs, both before any GPU-hour is spent:

1. Validate the analysis pipeline by recovering the published A2 exponent from
   `exp_data.csv` alone. If we cannot reproduce their fit from their own data, the
   pipeline is wrong and nothing downstream is trustworthy.

2. Estimate the LOSS-BOWL CURVATURE at the 33M anchor. Curvature plus seed variance is
   what determines the uncertainty in a fitted optimum, so this feeds the Stage-B power
   calculation. Per the plan it is a CONSERVATIVE CROSS-CHECK only -- never a plug-in
   for below-range scales, because curvature need not transfer downward.

Note on IsoFLOP slicing: rows at a fixed (family, vocabulary) span 20 budgets, and the
budget grid is not aligned across vocabularies (H = D/f(V) differs per V). A bowl at a
common C therefore requires interpolating each vocabulary's loss-vs-C curve and
evaluating them all at the same C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import reference as ref

CSV = Path(__file__).resolve().parents[1] / "reference" / "exp_data.csv"


def load() -> pd.DataFrame:
    return pd.read_csv(CSV)


# --- IsoFLOP slicing ---------------------------------------------------------------


def bowl_at_budget(df: pd.DataFrame, nnv: float, target_C: float) -> pd.DataFrame:
    """Loss vs vocabulary at a common compute budget, for one family.

    Each vocabulary's L_u(C) is interpolated in log-C space; vocabularies whose budget
    range does not cover `target_C` are dropped rather than extrapolated.
    """
    fam = df[df["Non_vocab_parameters"] == nnv]
    out = []
    for V, grp in fam.groupby("vocab_size"):
        grp = grp.sort_values("FLOPs")
        lo, hi = grp["FLOPs"].iloc[0], grp["FLOPs"].iloc[-1]
        if not (lo <= target_C <= hi):
            continue
        loss = np.interp(np.log(target_C), np.log(grp["FLOPs"]), grp["Lossu"])
        out.append({"V": V, "Lossu": float(loss)})
    return pd.DataFrame(out).sort_values("V").reset_index(drop=True)


@dataclass
class BowlFit:
    a: float          # quadratic coefficient in ln V -- the CURVATURE
    b: float
    c: float
    ln_v_star: float
    v_star: float
    n_points: int
    convex: bool
    rmse: float


def fit_bowl(bowl: pd.DataFrame) -> BowlFit:
    """Weighted-quadratic bowl fit in ln V. Unweighted here (published data has no seeds)."""
    x = np.log(bowl["V"].to_numpy())
    y = bowl["Lossu"].to_numpy()
    a, b, c = np.polyfit(x, y, 2)
    ln_v_star = -b / (2 * a)
    resid = y - np.polyval([a, b, c], x)
    return BowlFit(
        a=a, b=b, c=c,
        ln_v_star=ln_v_star,
        v_star=float(np.exp(ln_v_star)),
        n_points=len(x),
        convex=a > 0,
        rmse=float(np.sqrt(np.mean(resid**2))),
    )


# --- A2 refit ----------------------------------------------------------------------


def refit_a2(df: pd.DataFrame, n_budgets: int = 12) -> dict:
    """Recover the A2 exponent from the released data.

    For a ladder of common budgets, locate each family's loss-minimising vocabulary,
    convert to N_v = V*d, and regress ln N_v on ln N_nv. Tao's published alpha is
    0.8353974035228025.
    """
    families = sorted(df["Non_vocab_parameters"].unique())
    pairs: list[tuple[float, float]] = []

    for nnv in families:
        fam = df[df["Non_vocab_parameters"] == nnv]
        d = float(fam["embed_dim"].iloc[0])
        # budgets covered by EVERY vocabulary in this family
        lo = fam.groupby("vocab_size")["FLOPs"].min().max()
        hi = fam.groupby("vocab_size")["FLOPs"].max().min()
        if not np.isfinite(lo) or lo >= hi:
            continue
        for C in np.geomspace(lo, hi, n_budgets):
            bowl = bowl_at_budget(df, nnv, C)
            if len(bowl) < 5:
                continue
            fit = fit_bowl(bowl)
            if not fit.convex:
                continue
            v = fit.v_star
            if not (bowl["V"].min() <= v <= bowl["V"].max()):
                continue  # boundary-censored; excluded rather than clipped
            pairs.append((nnv, v * d))

    arr = np.array(pairs)
    slope, intercept = np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)
    return {
        "alpha_refit": float(slope),
        "alpha_published": ref.A2_ALPHA,
        "rel_error": float(abs(slope - ref.A2_ALPHA) / ref.A2_ALPHA),
        "intercept": float(intercept),
        "n_points": len(arr),
        "n_families": len(set(arr[:, 0])),
    }


def anchor_curvature(df: pd.DataFrame) -> dict:
    """Bowl curvature at the 33M anchor, across its covered budget range."""
    nnv = 33_222_784.0
    fam = df[df["Non_vocab_parameters"] == nnv]
    lo = fam.groupby("vocab_size")["FLOPs"].min().max()
    hi = fam.groupby("vocab_size")["FLOPs"].max().min()

    fits = []
    for C in np.geomspace(lo, hi, 8):
        bowl = bowl_at_budget(df, nnv, C)
        if len(bowl) < 5:
            continue
        f = fit_bowl(bowl)
        fits.append({"C": C, "a": f.a, "v_star": f.v_star,
                     "convex": f.convex, "rmse": f.rmse, "n": f.n_points})
    fdf = pd.DataFrame(fits)
    convex = fdf[fdf["convex"]]
    return {
        "n_budgets": len(fdf),
        "n_convex": int(fdf["convex"].sum()),
        "curvature_median": float(convex["a"].median()) if len(convex) else float("nan"),
        "curvature_min": float(convex["a"].min()) if len(convex) else float("nan"),
        "curvature_max": float(convex["a"].max()) if len(convex) else float("nan"),
        "v_star_median": float(convex["v_star"].median()) if len(convex) else float("nan"),
        "rmse_median": float(convex["rmse"].median()) if len(convex) else float("nan"),
        "table": fdf,
    }
