"""Convert the P2/P4 convention shifts (in nats) into the units the decision is made in.

`scripts/p2_smoothing_sensitivity.py` measures how far each convention moves `H_unigram`,
and therefore `L_u`, across vocabularies. That is a number in nats. The decisions are not
made in nats:

  M1 is a test on `theta = ln(N_v*/N_v_pred)`, where `N_v*` comes from `argmin_V L_u`.
  M2 is a SIGN test on `D`, with no equivalence margin at all.

So a shift in `L_u` matters only through how far it can move an argmin, and that depends
entirely on how sharply `L_u` curves around its minimum. This script measures that
curvature from Tao's own released measurements and converts.

Method. `exp_data.csv` has no exact IsoFLOP groups -- FLOPs varies continuously with V,
which is why their approach2 interpolates. IsoFLOP slices are rebuilt here the same way
they do it: quadratic `interp1d` of Lossu against FLOPs, per vocabulary, evaluated at a
common budget. A quadratic in `ln V` is then fitted to the slice and its curvature taken.

The comparison is a worst case in the shape of the perturbation: a spread `S` in the
convention-induced shift is treated as a monotone tilt of slope `S / range(ln V)`, which
displaces the argmin of a quadratic by `slope / curvature`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]

LN_1_5 = math.log(1.5)
# The vocabulary grid this study actually runs over.
OUR_V_LO, OUR_V_HI = 384, 17792
OUR_LNV_RANGE = math.log(OUR_V_HI) - math.log(OUR_V_LO)


def isoflop_slice(fam: pd.DataFrame, budget: float) -> tuple[np.ndarray, np.ndarray]:
    """Lossu at a common FLOP budget for every vocabulary in one N_nv family."""
    vs, ls = [], []
    for V, g in fam.groupby("vocab_size"):
        g = g.sort_values("FLOPs")
        f, y = g["FLOPs"].to_numpy(float), g["Lossu"].to_numpy(float)
        # Interpolate only; never extrapolate, or the fitted minimum is an artifact.
        if not (f.min() <= budget <= f.max()):
            continue
        vs.append(float(V))
        ls.append(float(interp1d(f, y, kind="quadratic")(budget)))
    return np.asarray(vs), np.asarray(ls)


def main() -> None:
    df = pd.read_csv(ROOT / "reference" / "exp_data.csv")

    results = []
    for nnv, fam in df.groupby("Non_vocab_parameters"):
        # Budgets where every vocabulary in the family has coverage.
        lo = fam.groupby("vocab_size")["FLOPs"].min().max()
        hi = fam.groupby("vocab_size")["FLOPs"].max().min()
        for frac in (0.15, 0.35, 0.55, 0.75, 0.95):
            budget = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
            vs, ls = isoflop_slice(fam, budget)
            if len(vs) < 5:
                continue
            x = np.log(vs)
            c = np.polyfit(x, ls, 2)
            if c[0] <= 0:  # not convex in ln V; no interior minimum to displace
                continue
            curv = 2 * c[0]
            vstar = math.exp(-c[1] / (2 * c[0]))
            results.append(
                {"Nnv": float(nnv), "budget": budget, "n": len(vs),
                 "curvature": curv, "V_star": vstar}
            )

    if not results:
        raise SystemExit("no convex IsoFLOP slices recovered")

    curvs = np.array([r["curvature"] for r in results])
    print(f"{'N_nv':>12} {'budget':>11} {'n':>3} {'curv d2L/dlnV2':>15} {'V*':>9}")
    for r in results:
        print(f"{r['Nnv']:>12.4g} {r['budget']:>11.4g} {r['n']:>3} "
              f"{r['curvature']:>15.5f} {r['V_star']:>9.0f}")

    a_min = curvs.min()
    print(f"\ncurvature d2L_u/d(lnV)^2 : min {a_min:.5f}  median {np.median(curvs):.5f}  "
          f"max {curvs.max():.5f}   (n={len(curvs)} slices)")

    # A tilt of this slope displaces the argmin by exactly the M1 margin.
    tilt_needed = a_min * LN_1_5
    print(f"\ntilt in dL_u/d(lnV) that displaces argmin by the M1 margin ln1.5={LN_1_5:.4f}:")
    print(f"  {tilt_needed:.6f} nats per unit lnV   (using the WEAKEST curvature observed)")

    sens = json.loads((ROOT / "results" / "p2_smoothing_sensitivity.json").read_text())
    print(f"\nconvention-induced tilt, spread / lnV-range ({OUR_LNV_RANGE:.3f}):")
    print(f"{'convention':>14} {'spread (nats)':>15} {'tilt':>12} {'margin factor':>15}")
    out = {}
    for key, d in sens["delta_summary"].items():
        if d["spread"] == 0.0:
            continue
        tilt = d["spread"] / OUR_LNV_RANGE
        factor = tilt_needed / tilt
        out[key] = {"spread": d["spread"], "tilt": tilt, "margin_factor": factor}
        print(f"{key:>14} {d['spread']:>15.3e} {tilt:>12.3e} {factor:>15,.0f}x")

    print("\nInterpretation: 'margin factor' is how many times larger the convention effect")
    print("would have to be before it could move theta by the M1 equivalence margin.")

    (ROOT / "results" / "p2p4_decision_relevance.json").write_text(
        json.dumps({"slices": results, "weakest_curvature": float(a_min),
                    "tilt_needed_for_M1_margin": tilt_needed,
                    "conventions": out}, indent=2)
    )
    print(f"\nwrote results/p2p4_decision_relevance.json")


if __name__ == "__main__":
    main()
