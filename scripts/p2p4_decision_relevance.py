"""Convert the P2/P4 convention shifts (in nats) into the units the decision is made in.

`scripts/p2_smoothing_sensitivity.py` measures how far each convention moves `H_unigram`,
and therefore `L_u`, across vocabularies. That is a number in nats. The decisions are not
made in nats:

  M1 is a test on `theta = ln(N_v*/N_v_pred)`, where `N_v*` comes from `argmin_V L_u`.
  M2 is a SIGN test on `D`, with no equivalence margin at all.

So a shift in `L_u` matters only through how far it can move an argmin, which depends on
(a) the LOCAL SLOPE of the perturbation in `ln V` and (b) the curvature of `L_u` about its
minimum. For a quadratic with curvature `a` perturbed by `eps`, the argmin moves to where
`a*(x - x*) + eps'(x) = 0`, so a slope `s` displaces it by `s / a`.

Two traps, both of which an earlier version of this script fell into:

1.  **Spread is not slope.** Treating a perturbation's cross-V spread `S` as a monotone
    tilt of slope `S / range(lnV)` is NOT a worst case: a bounded-range perturbation can
    have arbitrarily large derivative. The measured shift is used directly here and its
    largest slope between adjacent grid vocabularies is taken. That is up to 39x larger
    than the tilt heuristic gave.

2.  **The curvature floor depends on how it is fitted.** A quadratic in `ln V` fitted
    globally over Tao's grid [4096, 96256] is not a local model of the minimum, and one
    global slice returns `V* = 788`, far outside the grid, which shows the global fit is
    not always trustworthy. Curvature is therefore computed under three fitting windows
    and the SMALLEST value across all of them is used, which is the conservative direction:
    weaker curvature means a given slope displaces the argmin further.
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
WINDOWS = {"global": None, "local7": 7, "local5": 5}


def isoflop_slices(fam: pd.DataFrame):
    """Lossu at common FLOP budgets for every vocabulary in one N_nv family.

    Rebuilt the way their approach2 does it: quadratic `interp1d` of Lossu against FLOPs.
    Interpolation only -- extrapolating would manufacture minima that were never measured.
    """
    lo = fam.groupby("vocab_size")["FLOPs"].min().max()
    hi = fam.groupby("vocab_size")["FLOPs"].max().min()
    for frac in (0.15, 0.35, 0.55, 0.75, 0.95):
        budget = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        vs, ls = [], []
        for V, g in fam.groupby("vocab_size"):
            g = g.sort_values("FLOPs")
            f, y = g["FLOPs"].to_numpy(float), g["Lossu"].to_numpy(float)
            if f.min() <= budget <= f.max():
                vs.append(float(V))
                ls.append(float(interp1d(f, y, kind="quadratic")(budget)))
        if len(vs) >= 5:
            yield budget, np.log(np.asarray(vs)), np.asarray(ls)


def curvature(x: np.ndarray, y: np.ndarray, k: int | None):
    """Second derivative of a quadratic fitted in ln V, optionally windowed at the min."""
    if k is not None:
        j = int(np.argmin(y))
        lo = max(0, min(j - k // 2, len(x) - k))
        x, y = x[lo : lo + k], y[lo : lo + k]
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return None, None  # not convex; no interior minimum to displace
    return 2 * c[0], math.exp(-c[1] / (2 * c[0]))


def main() -> None:
    df = pd.read_csv(ROOT / "reference" / "exp_data.csv")

    per_window: dict[str, list[float]] = {}
    outliers = []
    for name, k in WINDOWS.items():
        curvs = []
        for nnv, fam in df.groupby("Non_vocab_parameters"):
            for budget, x, y in isoflop_slices(fam):
                a, vstar = curvature(x, y, k)
                if a is None:
                    continue
                curvs.append(a)
                if not (math.exp(x.min()) <= vstar <= math.exp(x.max())):
                    outliers.append((name, float(nnv), budget, a, vstar))
        per_window[name] = curvs

    print(f"{'window':>8} {'n':>4} {'min curv':>10} {'median':>10} {'max':>10}")
    for name, cs in per_window.items():
        a = np.asarray(cs)
        print(f"{name:>8} {len(a):>4} {a.min():>10.5f} {np.median(a):>10.5f} {a.max():>10.5f}")

    if outliers:
        print("\nslices whose fitted V* falls outside the measured grid (fit not trustworthy):")
        for name, nnv, b, a, v in outliers:
            print(f"  {name:>7} N_nv={nnv:.4g} budget={b:.4g} curv={a:.5f} V*={v:.0f}")

    a_min = min(min(cs) for cs in per_window.values())
    src = [n for n, cs in per_window.items() if min(cs) == a_min][0]
    slope_needed = a_min * LN_1_5
    print(f"\nmost conservative curvature across all windows: {a_min:.5f}  (from {src})")
    print(f"slope in dL_u/d(lnV) that displaces the argmin by the M1 margin "
          f"ln1.5={LN_1_5:.4f}: {slope_needed:.6f} nats per unit lnV")

    # --- the measured perturbations, used directly rather than assumed monotone ---------
    sens = json.loads((ROOT / "results" / "p2_smoothing_sensitivity.json").read_text())
    rows = sorted(sens["rows"], key=lambda r: r["V"])
    x = np.log(np.array([r["V"] for r in rows], dtype=float))
    base = np.array([r["H"]["add1"] for r in rows])

    print(f"\n{'convention':>14} {'spread':>11} {'max local slope':>17} {'margin factor':>15}")
    out = {}
    for key in list(rows[0]["H"].keys()) + ["no_eos_add1"]:
        if key == "no_eos_add1":
            d = np.array([r["H_no_eos_add1"] for r in rows]) - base
        else:
            d = np.array([r["H"][key] for r in rows]) - base
        if np.allclose(d, 0.0):
            continue
        slope = float(np.abs(np.diff(d) / np.diff(x)).max())
        factor = slope_needed / slope
        out[key] = {"spread": float(d.max() - d.min()), "max_slope": slope,
                    "margin_factor": factor}
        print(f"{key:>14} {d.max() - d.min():>11.3e} {slope:>17.3e} {factor:>14,.0f}x")

    print("\n'margin factor' is how many times larger the convention's local slope would")
    print("have to be before it could displace theta by the M1 equivalence margin.")

    (ROOT / "results" / "p2p4_decision_relevance.json").write_text(json.dumps({
        "curvature_by_window": {n: {"n": len(c), "min": min(c), "median": float(np.median(c)),
                                    "max": max(c)} for n, c in per_window.items()},
        "outlier_slices": [{"window": n, "Nnv": v, "budget": b, "curvature": a, "V_star": s}
                           for n, v, b, a, s in outliers],
        "conservative_curvature": a_min,
        "slope_needed_for_M1_margin": slope_needed,
        "conventions": out,
    }, indent=2))
    print("\nwrote results/p2p4_decision_relevance.json")


if __name__ == "__main__":
    main()
