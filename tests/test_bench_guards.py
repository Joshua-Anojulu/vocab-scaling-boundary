"""Guards on the VRAM ceiling search.

These encode a platform behaviour that cost real measurement time to discover: on
Windows/WDDM, exceeding VRAM does NOT raise. The driver backs the allocation with host
memory over PCIe and the run continues at a fraction of the rate, still reporting
success. Measured directly on the RTX 4060 Laptop (8.59 GB):

    16M V=4480 b=16:  47,716 tok/s   5.05 TFLOP/s   peak  6.92 GB   <- usable
    16M V=4480 b=32:   2,388 tok/s   0.25 TFLOP/s   peak 13.45 GB   <- spilled, ok=True

A search that only catches OOM would have selected b=32 and run the entire sweep ~20x
slower than necessary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import bench  # noqa: E402

VRAM = 8.59  # RTX 4060 Laptop


def test_usable_probe_is_not_rejected() -> None:
    assert bench.reject_reason(6.92, 47_716, VRAM, best_tps=48_183) is None


def test_spill_is_rejected_even_though_the_run_succeeded() -> None:
    """The real b=32 observation: no exception, but 13.45 GB on an 8.59 GB card."""
    why = bench.reject_reason(13.45, 2_388, VRAM, best_tps=48_183)
    assert why is not None and "spill" in why


def test_spill_is_caught_before_throughput_collapse() -> None:
    """Spill is the cause; report it rather than the symptom, even if both trip."""
    why = bench.reject_reason(13.45, 2_388, VRAM, best_tps=48_183)
    assert "spill" in why and "collapse" not in why


def test_throughput_collapse_is_rejected_without_spill() -> None:
    """Covers non-memory regressions -- e.g. a kernel falling off a fast path."""
    why = bench.reject_reason(3.0, 10_000, VRAM, best_tps=48_183)
    assert why is not None and "collapse" in why


def test_first_probe_has_no_baseline_and_is_accepted() -> None:
    assert bench.reject_reason(1.04, 100_624, VRAM, best_tps=0.0) is None


def test_mild_throughput_dip_is_tolerated() -> None:
    """b=16 vs b=8 was a ~1% dip; a plateau must not be mistaken for a failure."""
    assert bench.reject_reason(6.92, 47_716, VRAM, best_tps=48_183) is None
    # 75% of best is still above the 70% floor
    assert bench.reject_reason(3.0, 36_137, VRAM, best_tps=48_183) is None


def test_thresholds_are_the_documented_values() -> None:
    assert bench.SPILL_FRACTION == 0.95
    assert bench.COLLAPSE_FRACTION == 0.70
