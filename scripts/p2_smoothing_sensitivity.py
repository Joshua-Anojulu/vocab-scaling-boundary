"""How much does the unresolved P2 smoothing convention actually move the primary metric?

`L_u = CE_model - H_unigram`. The smoothing convention enters ONLY through `H_unigram`,
which depends on token counts and not on any trained model. So the entire sensitivity of
the primary metric to P2 is measurable now, on the corpus already tokenized, with no GPU.

What matters is NOT the absolute shift. Both M1 and M2 compare `L_u` ACROSS vocabularies:

  M1 takes `argmin_V L_u(V)` at fixed compute, so a shift that is common to all V cannot
     move the argmin, while a shift that varies with V can.
  M2 is `D = L_u(V_run, 1.1C) - L_u(V*, C)`, a difference between two DIFFERENT
     vocabularies, so a V-dependent shift does not cancel there either.

Therefore the decision-relevant quantity is the SPREAD ACROSS V of the convention-induced
shift, `delta(V) = H_uni^conv(V) - H_uni^add1(V)`, compared against the M1/M2 margins.
A large common shift is harmless; a small V-dependent one is not.

Also measured here: the P4 EOS packing convention, which changes both the unigram
distribution and the evaluation token count, and so interacts with P2 directly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "tokens"
OUT = ROOT / "results" / "p2_smoothing_sensitivity.json"

CHUNK = 1 << 26  # 64M ids at a time; bounds peak RSS regardless of array size

# Special-token layout is fixed by the tokenizer builder: BPE is trained to V-3 and the
# three specials are appended, so they occupy the TOP three ids.
N_SPECIAL = 3

ALPHAS = [1.0, 0.5, 0.1, 0.01, 1e-4, 1e-6]

# Smoothing matters more when the unigram is fitted on less text, so the smallest training
# budget in the grid is the worst case. That is the 2M scale at V=6912: 33.2M tokens. Every
# vocabulary is refitted on a common prefix of this length as a robustness check, because
# the full-array numbers alone overstate the headroom.
WORST_CASE_PREFIX = 33_000_000


def counts_of(path: Path, vocab_size: int, limit: int | None = None) -> tuple[np.ndarray, int]:
    """Exact token-id histogram over an array (or its first `limit` ids), in bounded chunks."""
    arr = np.load(path, mmap_mode="r")
    end = arr.shape[0] if limit is None else min(limit, arr.shape[0])
    total = np.zeros(vocab_size, dtype=np.int64)
    n = 0
    for start in range(0, end, CHUNK):
        block = np.asarray(arr[start : min(start + CHUNK, end)], dtype=np.int64)
        total += np.bincount(block, minlength=vocab_size)
        n += block.shape[0]
    if total.sum() != n:
        raise SystemExit(f"{path.name}: ids outside [0,{vocab_size}) present")
    return total, n


def h_unigram(train_counts: np.ndarray, eval_counts: np.ndarray, alpha: float) -> float:
    """Cross-entropy of the eval stream under the add-alpha unigram, nats/token."""
    V = train_counts.shape[0]
    denom = train_counts.sum() + alpha * V
    logp = np.log((train_counts + alpha) / denom)
    n_eval = eval_counts.sum()
    return float(-(eval_counts * logp).sum() / n_eval)


def main() -> None:
    manifest = json.loads((TOKENS / "manifest.json").read_text())
    vocabs = sorted({e["vocab_size"] for e in manifest})
    by = {(e["vocab_size"], e["split"]): e for e in manifest}

    rows = []
    for V in vocabs:
        tr, n_tr = counts_of(TOKENS / f"v{V}_train.npy", V)
        ev, n_ev = counts_of(TOKENS / f"v{V}_selection_val.npy", V)

        n_unseen = int((tr == 0).sum())
        # The EOS id is read from the tokenizer rather than inferred from counts. Inferring
        # it by matching the stored count against `n_eos_inserted` does NOT work: arrays are
        # truncated at their token target, so the last document's EOS can be cut off.
        eos_id = Tokenizer.from_file(str(ROOT / "tokenizers" / f"bpe_v{V}.json")).token_to_id("<eos>")
        if eos_id is None or not (V - N_SPECIAL <= eos_id < V):
            raise SystemExit(f"V={V}: unexpected EOS id {eos_id}")

        entry = {
            "V": V,
            "n_train_tokens": n_tr,
            "n_eval_tokens": n_ev,
            "n_unseen": n_unseen,
            "n_singleton": int((tr == 1).sum()),
            "eos_id": eos_id,
            "eos_share_train": float(tr[eos_id] / n_tr),
            "H": {f"add{a:g}": h_unigram(tr, ev, a) for a in ALPHAS},
        }

        # P4: drop EOS from both the fitted unigram and the evaluation stream, which is
        # what "no packing separator" would have produced for this metric.
        tr_ne, ev_ne = tr.copy(), ev.copy()
        tr_ne[eos_id] = 0
        ev_ne[eos_id] = 0
        entry["H_no_eos_add1"] = h_unigram(tr_ne, ev_ne, 1.0)

        rows.append(entry)
        print(
            f"V={V:>6}  unseen={n_unseen:>5}  singleton={entry['n_singleton']:>5}  "
            f"H(add1)={entry['H']['add1']:.6f}  "
            f"H(add1e-06)={entry['H']['add1e-06']:.6f}  "
            f"H(no-eos)={entry['H_no_eos_add1']:.6f}",
            flush=True,
        )

    # --- the decision-relevant summary ------------------------------------------------
    base = {r["V"]: r["H"]["add1"] for r in rows}
    print("\n=== convention-induced shift delta(V) = H^conv(V) - H^add1(V), nats/token ===")
    print(f"{'convention':>14} {'min delta':>12} {'max delta':>12} {'SPREAD across V':>17}")
    summary = {}
    for key in list(rows[0]["H"].keys()) + ["no_eos_add1"]:
        if key == "no_eos_add1":
            deltas = {r["V"]: r["H_no_eos_add1"] - base[r["V"]] for r in rows}
        else:
            deltas = {r["V"]: r["H"][key] - base[r["V"]] for r in rows}
        lo, hi = min(deltas.values()), max(deltas.values())
        summary[key] = {"min": lo, "max": hi, "spread": hi - lo}
        print(f"{key:>14} {lo:>12.3e} {hi:>12.3e} {hi - lo:>17.3e}")

    m1_margin = math.log(1.5)
    print(f"\nM1 equivalence margin on theta = ln(N_v*/N_v_pred): +/-{m1_margin:.4f}")
    print("Spread is NOT the decision-relevant quantity -- a bounded-range perturbation can")
    print("have arbitrarily large slope, and it is slope against curvature that moves an")
    print("argmin. See scripts/p2p4_decision_relevance.py for the conversion.")

    # --- worst-case robustness: refit every vocabulary on the SMALLEST training budget ----
    print(f"\n=== refit on a common {WORST_CASE_PREFIX:,}-token prefix (smallest T_target) ===")
    worst = {}
    for V in vocabs:
        tr, _ = counts_of(TOKENS / f"v{V}_train.npy", V, limit=WORST_CASE_PREFIX)
        ev, _ = counts_of(TOKENS / f"v{V}_selection_val.npy", V)
        worst[V] = {f"add{a:g}": h_unigram(tr, ev, a) for a in (1.0, 1e-6)}
    xw = np.log(np.array(vocabs, dtype=float))
    dw = np.array([worst[V]["add1e-06"] - worst[V]["add1"] for V in vocabs])
    slope_w = float(np.abs(np.diff(dw) / np.diff(xw)).max())
    dfull = np.array([r["H"]["add1e-06"] - r["H"]["add1"] for r in rows])
    slope_f = float(np.abs(np.diff(dfull) / np.diff(np.log(np.array([r["V"] for r in rows], dtype=float)))).max())
    print(f"add1 vs add1e-6 max local slope: {slope_f:.3e} (full arrays) "
          f"-> {slope_w:.3e} (worst-case prefix), {slope_w / slope_f:.2f}x worse")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows,
        "delta_summary": summary,
        "worst_case_prefix": {
            "n_tokens": WORST_CASE_PREFIX,
            "H": {str(V): worst[V] for V in vocabs},
            "max_slope_full": slope_f,
            "max_slope_prefix": slope_w,
        },
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
