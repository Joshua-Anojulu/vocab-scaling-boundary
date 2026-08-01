"""Redo the fertility-vs-f(V) comparison on population counts, not a 200-doc sample.

The earlier table in `results/fertility_vs_fitted.md` estimated tokens-per-character from
200 sampled documents per vocabulary. `data/tokens/manifest.json` now carries exact counts
for every array, so the estimate can be replaced with the population value.

Two choices matter and are made explicitly here:

1.  **Which split.** The `train` arrays cover a DIFFERENT number of documents per
    vocabulary (22,000 to 570,000), because each vocabulary needed its own token target.
    Comparing fertility across V on those arrays would confound V with the text each
    tokenizer happened to see. The `selection_val` arrays are the same 17,749 documents
    for every V, so the cross-V curve is measured on identical text. That is the split
    used for the comparison; train is reported alongside only as a magnitude check.

2.  **EOS.** Per amendment P4 the packer inserts one `<eos>` per document, which inflates
    the token count by exactly `n_docs`. Tao's `f(V)` is a statement about how text
    segments, not about packing, so the headline number excludes EOS. Both are printed
    so the size of the choice is visible rather than assumed harmless.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reference import fertility  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "tokens" / "manifest.json"

# The range over which Tao et al. fitted f(V); outside it the curve is extrapolation.
FIT_LO, FIT_HI = 4096, 96256


def main() -> None:
    entries = json.loads(MANIFEST.read_text())
    val = sorted(
        (e for e in entries if e["split"] == "selection_val"),
        key=lambda e: e["vocab_size"],
    )
    train = {e["vocab_size"]: e for e in entries if e["split"] == "train"}

    # The comparison is only meaningful if every vocabulary really did see the same text.
    chars = {e["n_chars"] for e in val}
    docs = {e["n_docs"] for e in val}
    if len(chars) != 1 or len(docs) != 1:
        raise SystemExit(
            f"selection_val is not a common text set across V: "
            f"{len(chars)} distinct char counts, {len(docs)} distinct doc counts"
        )
    n_chars = chars.pop()
    n_docs = docs.pop()

    rows = []
    for e in val:
        V = e["vocab_size"]
        tok_with_eos = e["n_tokens"]
        tok_no_eos = tok_with_eos - e["n_eos_inserted"]
        measured = tok_no_eos / n_chars
        f_v = fertility(V)
        t = train[V]
        rows.append(
            {
                "V": V,
                "measured": measured,
                "measured_with_eos": tok_with_eos / n_chars,
                "f_v": f_v,
                "ratio": measured / f_v,
                "train_tok_per_char": (t["n_tokens"] - t["n_eos_inserted"]) / t["n_chars"],
                "in_range": FIT_LO <= V <= FIT_HI,
            }
        )

    print(f"selection_val: {n_docs:,} documents, {n_chars:,} characters, identical for all V\n")
    print(f"{'V':>7} {'tok/char':>9} {'f(V)':>8} {'ratio':>7} {'+EOS':>9} {'train':>9}  region")
    for r in rows:
        region = "inside" if r["in_range"] else "below"
        print(
            f"{r['V']:>7} {r['measured']:>9.5f} {r['f_v']:>8.5f} {r['ratio']:>7.4f} "
            f"{r['measured_with_eos']:>9.5f} {r['train_tok_per_char']:>9.5f}  {region}"
        )

    ratios = [r["ratio"] for r in rows]
    inside = [r["ratio"] for r in rows if r["in_range"]]
    below = [r["ratio"] for r in rows if not r["in_range"]]
    print(
        f"\nratio  overall [{min(ratios):.4f}, {max(ratios):.4f}]"
        f"   inside fitted range [{min(inside):.4f}, {max(inside):.4f}]"
        f"   below [{min(below):.4f}, {max(below):.4f}]"
    )

    eos_infl = [r["measured_with_eos"] / r["measured"] - 1 for r in rows]
    print(f"EOS inflates tokens/char by {min(eos_infl)*100:.4f}%–{max(eos_infl)*100:.4f}%")

    # The earlier 200-doc table is superseded; quantify how far off it was.
    print("\nsampling error of the superseded 200-doc estimates:")
    old = {
        384: 0.5970, 512: 0.5212, 768: 0.4578, 896: 0.4398, 1024: 0.4254,
        1152: 0.4133, 1536: 0.3879, 1664: 0.3818, 2176: 0.3609, 3200: 0.3350,
        3456: 0.3295, 4224: 0.3176, 4480: 0.3143, 6144: 0.2972, 6272: 0.2960,
        6912: 0.2908, 8448: 0.2810, 8960: 0.2784,
    }
    devs = []
    for r in rows:
        if r["V"] in old:
            dev = old[r["V"]] / r["measured"] - 1
            devs.append((abs(dev), r["V"], dev))
    devs.sort(reverse=True)
    for _, V, dev in devs[:5]:
        print(f"  V={V:>6}: 200-doc estimate was {dev*100:+.2f}% off the population value")
    print(f"  max |error| {devs[0][0]*100:.2f}%, median {sorted(d[0] for d in devs)[len(devs)//2]*100:.2f}%")


if __name__ == "__main__":
    main()
