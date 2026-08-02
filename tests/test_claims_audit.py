"""Mechanical audit of the claims this project has already had to withdraw.

Three failure modes have recurred across review tracks, and each was caught by a reviewer
rather than by the author:

1. **Incomplete sweeps.** A claim is withdrawn and corrected where the author happens to be
   looking, and left standing in three or four other files. This happened with the
   LR-coupling claim (fixed in `AMENDMENTS.md`, left in four files), the checkpoint
   LR-history equivalence (left in `lr_at` and `reference_architecture_findings.md`), the
   `+0.0134%` figures (fixed in prose, left in a docstring), and the 8% warmup (fixed in the
   constant, left in the module header and a test docstring).

2. **Evidence a reader cannot open.** A numeric claim citing an artifact that is gitignored
   or absent. This happened twice: the step-count test that silently skipped without
   `reference/exp_data.csv`, and the warmup stability table written to `runs/diagnostics/`.

3. **Numbers in prose that no artifact supports**, which is how `+0.0129%` became
   `+0.0134%` and `23.0x` became an unstable point estimate.

Reviewers should not be the mechanism that catches these. This file is the mechanism.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Phrases this project has withdrawn. A match is a failure UNLESS the surrounding lines
# mark it as withdrawn -- the record of what was claimed is deliberately preserved.
WITHDRAWN = [
    (r"tuned (?:to|for) (?:a |this )?batch",
     "LR-batch tuning was asserted without evidence; the released recipe only PAIRS them"),
    (r"learning rate is coupled to it",
     "same withdrawn coupling claim"),
    (r"actually produced the released IsoFLOP data",
     "run.sh provenance is not established; exp_data.csv predates it"),
    (r"exactly the learning-rate history of a run trained to",
     "false once warmup scales with run length; exposure differs by up to 1.69x"),
    (r"checkpoint reuse is\s+therefore legitimate here",
     "rests on the refuted LR-history equivalence"),
    (r"initialisation-only variance is a lower bound",
     "withdrawn in the seed-semantics track; the claim is 'wrong variance component'"),
    (r"2000 of 25000 steps == 8%",
     "the study uses 10%; and warmup is a study choice, not a transcription"),
]

WITHDRAWAL_MARKERS = re.compile(
    r"withdraw|WITHDRAWN|is false|was false|overclaim|earlier (?:version|draft)|"
    r"no longer|not established|refut|corrected|NOT transcribed|does not claim|"
    r"an earlier|superseded",
    re.IGNORECASE,
)


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    keep = {".py", ".md", ".json", ".txt", ".toml", ".cfg"}
    return [ROOT / f for f in out
            if Path(f).suffix in keep and not f.startswith("reference/")]


@pytest.mark.parametrize("pattern,why", [(p, w) for p, w in WITHDRAWN])
def test_withdrawn_claims_do_not_survive_anywhere(pattern: str, why: str) -> None:
    """A withdrawn claim must be gone or explicitly marked withdrawn -- in EVERY file."""
    rx = re.compile(pattern, re.IGNORECASE)
    offenders = []
    for path in _tracked_text_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            window = "\n".join(lines[max(0, i - 6): i + 7])
            if not WITHDRAWAL_MARKERS.search(window):
                offenders.append(f"{path.relative_to(ROOT)}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, (
        f"withdrawn claim survives ({why}):\n  " + "\n  ".join(offenders))


def test_every_artifact_cited_by_the_amendments_is_readable() -> None:
    """A numeric claim citing a gitignored or missing artifact is not evidence."""
    text = (ROOT / "AMENDMENTS.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    # A path may be NAMED in a withdrawal note precisely because citing it was the mistake;
    # that is the record working as intended, not a live citation.
    cited = set()
    for i, line in enumerate(lines):
        window = "\n".join(lines[max(0, i - 6): i + 7])
        if WITHDRAWAL_MARKERS.search(window):
            continue
        cited.update(re.findall(r"`((?:results|runs|data|scripts|src|tests)/[\w./-]+)`", line))
    missing, ignored = [], []
    for rel in sorted(cited):
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        if r.returncode == 0:                      # 0 means IS ignored
            ignored.append(rel)
    assert not missing, f"AMENDMENTS.md cites paths that do not exist: {missing}"
    assert not ignored, (
        f"AMENDMENTS.md cites gitignored artifacts, so its numbers cannot be checked by a "
        f"reader: {ignored}")


def test_headline_figures_match_their_artifacts() -> None:
    """The specific numbers that have already been corrected once, re-derived from JSON."""
    budget = json.loads((ROOT / "results" / "budget_convention_error.json").read_text())
    assert budget["worst_by_batch_shape"]["mb4_ga4"]["worst_old_overshoot_pct"] == \
        pytest.approx(0.0129, abs=5e-5), "the +0.0129% figure drifted from its artifact"
    assert budget["worst_new_undershoot_pct"] == pytest.approx(-0.0011, abs=5e-5)

    stability = json.loads((ROOT / "results" / "warmup_stability.json").read_text())
    for row in stability:
        # The bug that shipped once: the probe must exercise the REAL warmup length.
        assert row["real_warmup_steps"] >= 13, row
        assert row["sustained_below_chance"], row
        assert not row["spike_after_warmup"], row


def test_plan_is_still_untouched() -> None:
    """The binding this whole amendment mechanism exists to protect."""
    log = subprocess.run(["git", "log", "--all", "--oneline", "--", "PLAN.md"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert len(log.strip().splitlines()) == 1, f"PLAN.md has been edited:\n{log}"
