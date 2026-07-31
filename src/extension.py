"""The study's own below-range extension of Tao et al.

Everything here is an ADMITTED EXTRAPOLATION beyond the reference's fitted range, and
the primary claim is scoped to the architecture family this module defines. Kept
strictly separate from `src/reference.py`, which may not be modified.

Two things in particular are ours, not Tao's:

1. The sub-50M width rule. `reference.Nnv_to_d` returns 512 for every Nnv <= 50M, which
   is infeasible below ~3.15M (one d=512 block). Note also that the reference family is
   not even a function of d -- its 110M and 176M members share d=768 -- so no rule can
   be "recovered" from it. Ours is new, and the claim is scoped accordingly.

2. The FLOP ruler's below-range form. Tao's helper hardcodes d=512 via the lookup; we
   substitute the actual preregistered width. This is a faithful algebraic extension,
   not verbatim execution of their architecture lookup, and their unmodified helper is
   used at the 33M anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from . import reference as ref

# --- constants ---------------------------------------------------------------------

D_REF, NNV_REF = 512, 33_222_784.0   # the anchor the width rule is calibrated through
HEAD_DIM = 64
PAD_QUANTUM = 128
N_SPECIAL_TOKENS = 3                 # BOS, EOS, PAD
BYTE_ALPHABET = 256

#: Smallest admissible vocabulary: byte alphabet + specials, snapped UP to the quantum.
V_MIN = PAD_QUANTUM * math.ceil((BYTE_ALPHABET + N_SPECIAL_TOKENS) / PAD_QUANTUM)  # 384

GRID_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Hard floor on confirmatory seeds. BCa inference from three seed clusters has highly
#: discrete, unstable coverage, so a small computed n is not accepted at face value.
SEED_FLOOR = 5

TARGET_NNV = (2_000_000, 4_000_000, 8_000_000, 16_000_000)


def snap(x: float, quantum: int) -> int:
    return quantum * int(round(x / quantum))


def snap_vocab(x: float) -> int:
    """Nearest-`PAD_QUANTUM` snap with the tokenizer floor applied."""
    return max(V_MIN, snap(x, PAD_QUANTUM))


# --- architecture rule -------------------------------------------------------------


@dataclass(frozen=True)
class Arch:
    target_nnv: int
    d: int
    d_ffn: int
    n_head: int
    n_layer: int
    nnv: int          # realized, reported exactly -- never the target
    per_layer: int

    @property
    def aspect(self) -> float:
        return self.d / self.n_layer


def per_layer_params(d: int, d_ffn: int) -> int:
    """4d^2 attention (q,k,v,o) + 3*d*d_ffn SwiGLU + 2d RMSNorm gains."""
    return 4 * d * d + 3 * d * d_ffn + 2 * d


def arch_for(target_nnv: float) -> Arch:
    """Constant-aspect-ratio rule: d ∝ Nnv^(1/3), calibrated to d=512 at the anchor.

    Deterministic and preregistered. `n_layer` rounds half up; the realized `nnv` is
    reported exactly rather than the target, because `V_pred = N_v/d` depends on it.
    """
    d = snap(D_REF * (target_nnv / NNV_REF) ** (1 / 3), 64)
    d_ffn = snap(8 * d / 3, 64)
    per = per_layer_params(d, d_ffn)
    n_layer = int(math.floor(target_nnv / per + 0.5))
    return Arch(
        target_nnv=int(target_nnv),
        d=d,
        d_ffn=d_ffn,
        n_head=d // HEAD_DIM,
        n_layer=n_layer,
        nnv=n_layer * per + d,
        per_layer=per,
    )


# --- predictions and grids ---------------------------------------------------------


def v_pred(arch: Arch) -> float:
    """Continuous predicted optimal vocabulary. M1 compares against this, unsnapped."""
    return ref.approach2_derivative(arch.nnv) / arch.d


def v_run(arch: Arch) -> int:
    """The IMPLEMENTED central vocabulary -- a model cannot use a continuous V.

    Used by the M2 runs at 1.1*C. By construction this equals the central grid point.
    """
    return snap_vocab(v_pred(arch))


def vocab_grid(arch: Arch) -> list[int]:
    """Five local points at V_pred * {1/4, 1/2, 1, 2, 4}, snapped, floor applied.

    If snapping collides two points, the whole set is regenerated as five distinct
    log-spaced feasible points retaining the one nearest V_pred -- never clipped
    individually, which would silently produce duplicates.
    """
    vp = v_pred(arch)
    grid = [snap_vocab(vp * m) for m in GRID_MULTIPLIERS]
    if len(set(grid)) == 5:
        return grid

    lo, hi = max(V_MIN, vp * GRID_MULTIPLIERS[0]), vp * GRID_MULTIPLIERS[-1]
    regenerated: list[int] = []
    for i in range(5):
        x = lo * (hi / lo) ** (i / 4)
        cand = snap_vocab(x)
        while cand in regenerated:
            cand += PAD_QUANTUM
        regenerated.append(cand)
    return sorted(regenerated)


# --- the ruler, extended -----------------------------------------------------------


def flops_extended(nnv: float, d: int, H: float, V: float) -> float:
    """C = 6*(Nnv + V*d_actual)*H*f(V) -- Tao's algebra, our preregistered width."""
    return 6 * (nnv + V * d) * H * ref.fertility(V)


def target_tokens(C: float, nnv: float, d: int, V: float) -> float:
    """T_target = C / (6*(Nnv + V*d)). IsoFLOP is enforced on TOKENS, not characters.

    f(V) is Tao's FITTED fertility; a newly trained tokenizer's realized fertility will
    differ, so consuming H characters would miss the target budget. Consuming exactly
    T_target tokens makes C exact by construction and turns the fertility mismatch into
    a reported observable instead of a silent error.
    """
    return C / (6 * (nnv + V * d))


def h_tao(C: float, nnv: float, d: int, V: float) -> float:
    """Tao-equivalent CHARACTER count, reported alongside H_actual for comparison."""
    return target_tokens(C, nnv, d, V) / ref.fertility(V)


# --- budget ------------------------------------------------------------------------


def budget(n_seed: int = SEED_FLOOR, boundary_extensions: int = 0) -> dict:
    """total = 6.1*n*SumC + pilot + anchor, plus n*C per boundary extension.

    6.1 = 5 sweep vocabularies + 1.1 for the M2 run at 1.1*C.
    The pilot is 6 configurations x 3 seeds at the 8M scale, and is EXCLUDED from
    confirmatory inference -- it exists to make D_pilot computable and to size n.
    """
    archs = [arch_for(t) for t in TARGET_NNV]
    Cs = {a.target_nnv: ref.Nnvopt_to_flops(a.nnv) for a in archs}
    sum_c = sum(Cs.values())
    c_8m = Cs[8_000_000]

    sweep = 5 * n_seed * sum_c
    m2 = 1.1 * n_seed * sum_c
    pilot = 6.1 * 3 * c_8m
    anchor = ref.Nnvopt_to_flops(NNV_REF)
    ext = boundary_extensions * n_seed * (sum_c / len(archs)) if boundary_extensions else 0.0
    # a boundary extension adds one vocabulary point at ONE scale; all four scales => n*SumC
    ext = n_seed * sum_c * (boundary_extensions / len(archs))

    total = sweep + m2 + pilot + anchor + ext
    runs = 20 * n_seed + 4 * n_seed + 18 + 1 + boundary_extensions * n_seed
    return {
        "n_seed": n_seed,
        "sum_C": sum_c,
        "C_per_scale": Cs,
        "sweep": sweep,
        "m2": m2,
        "pilot": pilot,
        "anchor": anchor,
        "boundary_extension": ext,
        "total": total,
        "runs": runs,
        "ideal_hours": {tf: total / (tf * 1e12) / 3600 for tf in (10, 15, 25)},
    }


def config_table(n_seed: int = SEED_FLOOR) -> list[dict]:
    """One row per architecture x vocabulary x seed -- the Stage C preregistration artifact."""
    rows: list[dict] = []
    for target in TARGET_NNV:
        a = arch_for(target)
        C = ref.Nnvopt_to_flops(a.nnv)
        vp, vr = v_pred(a), v_run(a)
        for V in vocab_grid(a):
            for seed in range(n_seed):
                T = target_tokens(C, a.nnv, a.d, V)
                rows.append(
                    {
                        **asdict(a),
                        "V": V,
                        "is_v_run": V == vr,
                        "V_pred_continuous": vp,
                        "seed": seed,
                        "budget_C": C,
                        "T_target_tokens": T,
                        "H_tao_chars": T / ref.fertility(V),
                        "fertility_fitted": ref.fertility(V),
                        "arm": "sweep",
                    }
                )
        # M2 arm: V_run at 1.1*C
        for seed in range(n_seed):
            T = target_tokens(1.1 * C, a.nnv, a.d, vr)
            rows.append(
                {
                    **asdict(a),
                    "V": vr,
                    "is_v_run": True,
                    "V_pred_continuous": vp,
                    "seed": seed,
                    "budget_C": 1.1 * C,
                    "T_target_tokens": T,
                    "H_tao_chars": T / ref.fertility(vr),
                    "fertility_fitted": ref.fertility(vr),
                    "arm": "m2_1.1C",
                }
            )
    return rows
