"""Stage B.5/B.6 -- sustained throughput benchmark and VRAM pilot.

Answers the only question that determines whether the study is affordable: what
throughput does this GPU actually SUSTAIN, as opposed to reach in a burst?

Two things this measures that a short benchmark cannot:

* **Thermal throttling.** A laptop RTX 4060 will clock down under sustained load. The
  plan requires >= 20 minutes precisely so the steady-state rate is measured rather than
  the first-minute rate, and requires per-run throughput logging so a throttled run is
  never mistaken for a slower configuration.
* **The real VRAM ceiling**, with the exact optimizer, context and chunked cross-entropy
  that training will use -- not "the weights fit".

Throughput is reported in the SAME FLOP convention as the budget
(`C_per_token = 6*(N_nv + V*d)`), so ideal-hours convert to wall-clock directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from . import extension as ext, model as M

#: PyTorch raises several distinct types for exhaustion. `torch.OutOfMemoryError` alone
#: is NOT enough: a driver-level failure surfaces as `torch.AcceleratorError`
#: ("CUDA error: out of memory"), which escaped an earlier narrower handler and aborted
#: the whole batch search instead of ending it.
_OOM_ERRORS: tuple[type[BaseException], ...] = tuple(
    t for t in (
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch, "AcceleratorError", None),
        torch.cuda.OutOfMemoryError,
        RuntimeError,          # broad on purpose -- but see classify_failure()
    ) if isinstance(t, type)
)

_OOM_MARKERS = ("out of memory", "outofmemory", "cuda_error_out_of_memory",
                "cudaerrormemoryallocation")


def classify_failure(exc: BaseException) -> str:
    """Distinguish genuine memory exhaustion from any other runtime failure.

    `RuntimeError` has to be caught broadly so a probe cannot take the search down, but
    labelling everything it catches as OOM is actively misleading: `torch.compile`
    failing with `TritonMissing` on Windows was reported as "OOM in warmup", which
    points at a VRAM problem that does not exist. Classify from the message, not the
    class.
    """
    name = type(exc).__name__
    text = f"{name} {exc}".lower()
    if any(mark in text for mark in _OOM_MARKERS) or "outofmemoryerror" in name.lower():
        return f"OOM: {name}"
    return f"ERROR: {name}: {str(exc)[:200]}"


@dataclass
class BenchResult:
    target_nnv: int
    d: int
    n_layer: int
    vocab: int
    batch: int
    block: int
    grad_accum: int
    tokens_per_step: int
    steps: int
    seconds: float
    tokens_per_sec: float
    achieved_tflops: float
    peak_mem_gb: float
    throttle_ratio: float          # last-quartile rate / first-quartile rate
    window_tps: list[float]
    ok: bool
    note: str = ""


def flops_per_token(nnv: int, vocab: int, d: int) -> float:
    """Budget convention: C = 6*(N_nv + V*d) per token."""
    return 6.0 * (nnv + vocab * d)


def make(target: int, vocab: int, block: int) -> tuple[M.Transformer, ext.Arch]:
    a = ext.arch_for(target)
    cfg = M.ModelConfig(
        vocab_size=vocab, d=a.d, n_layer=a.n_layer, n_head=a.n_head,
        d_ffn=a.d_ffn, block_size=block,
    )
    return M.build(cfg), a


def run_bench(
    target: int = 16_000_000,
    vocab: int = 4480,
    block: int = 2048,
    batch: int = 8,
    grad_accum: int = 1,
    seconds: float = 60.0,
    chunk: int = 4096,
    device: str = "cuda",
    warmup_steps: int = 5,
    compile_model: bool = False,
    window_log: str | None = None,
) -> BenchResult:
    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)
    m, a = make(target, vocab, block)
    m = m.to(dev)
    if compile_model:
        # Only the transformer body is compiled. The loss path contains a custom
        # autograd Function with a Python chunk loop, which inductor cannot trace.
        m.hidden_states = torch.compile(m.hidden_states, dynamic=False)  # type: ignore[method-assign]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    fpt = flops_per_token(a.nnv, vocab, a.d)
    tokens_per_step = batch * block * grad_accum

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            idx = torch.randint(0, vocab, (batch, block), device=dev)
            tgt = torch.randint(0, vocab, (batch, block), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m.loss(idx, tgt, chunk_size=chunk)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    try:
        for _ in range(warmup_steps):
            one_step()
        torch.cuda.synchronize(dev)
    except _OOM_ERRORS as e:  # pragma: no cover - hardware dependent
        torch.cuda.empty_cache()
        return BenchResult(
            target, a.d, a.n_layer, vocab, batch, block, grad_accum, tokens_per_step,
            0, 0.0, 0.0, 0.0, 0.0, 0.0, [], ok=False, note=f"warmup {classify_failure(e)}",
        )

    window_tps: list[float] = []
    steps = 0
    t0 = time.perf_counter()
    t_win = t0
    win_steps = 0
    try:
        while time.perf_counter() - t0 < seconds:
            one_step()
            steps += 1
            win_steps += 1
            now = time.perf_counter()
            if now - t_win >= 15.0:                 # 15 s reporting windows
                torch.cuda.synchronize(dev)
                w = win_steps * tokens_per_step / (now - t_win)
                window_tps.append(w)
                # Append IMMEDIATELY. A long sustained run that is killed at a harness
                # timeout must still leave its throttling curve behind; buffering the
                # whole run until the end once cost a full 40-minute measurement.
                if window_log:
                    with open(window_log, "a", encoding="utf-8") as fh:
                        fh.write(f"{now - t0:.1f},{w:.1f},"
                                 f"{w * fpt / 1e12:.4f},"
                                 f"{torch.cuda.max_memory_allocated(dev) / 1e9:.3f}\n")
                t_win, win_steps = now, 0
        torch.cuda.synchronize(dev)
    except _OOM_ERRORS as e:  # pragma: no cover - hardware dependent
        del m, opt
        torch.cuda.empty_cache()
        return BenchResult(
            target, a.d, a.n_layer, vocab, batch, block, grad_accum, tokens_per_step,
            steps, 0.0, 0.0, 0.0, 0.0, 0.0, [], ok=False,
            note=f"measurement {classify_failure(e)}",
        )
    elapsed = time.perf_counter() - t0

    tps = steps * tokens_per_step / elapsed
    q = max(1, len(window_tps) // 4)
    throttle = (
        sum(window_tps[-q:]) / q / (sum(window_tps[:q]) / q) if len(window_tps) >= 2 else 1.0
    )
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    del m, opt
    torch.cuda.empty_cache()

    return BenchResult(
        target_nnv=target, d=a.d, n_layer=a.n_layer, vocab=vocab, batch=batch,
        block=block, grad_accum=grad_accum, tokens_per_step=tokens_per_step,
        steps=steps, seconds=elapsed, tokens_per_sec=tps,
        achieved_tflops=tps * fpt / 1e12, peak_mem_gb=peak,
        throttle_ratio=throttle, window_tps=window_tps, ok=True,
    )


def probe_subprocess(
    target: int, vocab: int, block: int, batch: int,
    seconds: float = 0.1, chunk: int = 4096, grad_accum: int = 1,
    timeout: float = 900.0, compile_model: bool = False,
    window_log: str | None = None,
) -> BenchResult | None:
    """Run one configuration in a FRESH process and return its result, or None if it died.

    Subprocess isolation is not defensive programming here, it is required. Two distinct
    failures wear the same name:

      * `torch.OutOfMemoryError`   -- the caching allocator could not satisfy a request.
                                      Recoverable: empty_cache() and carry on.
      * `torch.AcceleratorError`   -- the CUDA DRIVER returned cudaErrorMemoryAllocation.
                                      The context is poisoned; even `empty_cache()` in
                                      the handler raises again. NOT recoverable in-process.

    An in-process batch search hit the second kind and took the whole run down with it.
    """
    payload = json.dumps(
        dict(target=target, vocab=vocab, block=block, batch=batch,
             seconds=seconds, chunk=chunk, grad_accum=grad_accum,
             compile_model=compile_model, window_log=window_log)
    )
    proc = subprocess.run(
        [sys.executable, "-m", "src.bench", payload],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__BENCH__"):
            return BenchResult(**json.loads(line[len("__BENCH__"):]))
    return None


#: Fraction of total VRAM above which a run is treated as having SPILLED to host memory.
SPILL_FRACTION = 0.95

#: A batch is rejected if its throughput falls below this multiple of the best seen.
COLLAPSE_FRACTION = 0.70


def total_vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def reject_reason(
    peak_mem_gb: float, tokens_per_sec: float, vram_gb: float, best_tps: float
) -> str | None:
    """Why this probe must not be used, or None if it is usable.

    Pure so it can be tested without a GPU. `ok=True` is NOT sufficient evidence that a
    configuration is usable -- see `find_max_batch`.
    """
    if peak_mem_gb > SPILL_FRACTION * vram_gb:
        return (f"REJECTED spill: peak {peak_mem_gb:.2f} GB > "
                f"{SPILL_FRACTION:.0%} of {vram_gb:.2f} GB")
    if best_tps and tokens_per_sec < COLLAPSE_FRACTION * best_tps:
        return (f"REJECTED collapse: {tokens_per_sec:,.0f} tok/s < "
                f"{COLLAPSE_FRACTION:.0%} of {best_tps:,.0f}")
    return None


def find_max_batch(
    target: int, vocab: int, block: int, chunk: int = 4096, max_batch: int = 64,
    vram_gb: float | None = None,
) -> tuple[int, list[BenchResult]]:
    """Largest batch that is genuinely usable -- NOT merely the largest that avoids OOM.

    On Windows/WDDM, exceeding VRAM does not raise. The driver silently backs the
    allocation with host memory over PCIe and the run keeps going roughly 20x slower.
    Observed directly: batch=32 at 16M/V=4480 reported `peak 13.45 GB` on an 8.59 GB
    card and fell from 5.05 to 0.25 TFLOP/s -- while reporting `ok=True`.

    An OOM-only search would therefore have selected that configuration and run the
    whole sweep at a twentieth of the achievable rate. Two additional rejection rules
    are required:

      * SPILL     -- peak allocation above `SPILL_FRACTION` of total VRAM.
      * COLLAPSE  -- throughput below `COLLAPSE_FRACTION` of the best seen so far.

    Returns (best_batch, all_probe_results) so the caller can log what was rejected and
    why, rather than silently truncating.
    """
    vram = vram_gb if vram_gb is not None else total_vram_gb()
    probes: list[BenchResult] = []
    best_batch, best_tps = 0, 0.0
    b = 1
    while b <= max_batch:
        r = probe_subprocess(target, vocab, block, batch=b, chunk=chunk, seconds=8.0)
        if r is None or not r.ok:
            break
        probes.append(r)
        why = reject_reason(r.peak_mem_gb, r.tokens_per_sec, vram, best_tps)
        if why:
            r.note = why
            break
        best_batch, best_tps = b, max(best_tps, r.tokens_per_sec)
        b *= 2
    return best_batch, probes


def _main(argv: list[str]) -> int:
    """Single-probe entry point. Prints one `__BENCH__<json>` line on success."""
    args = json.loads(argv[1])
    try:
        r = run_bench(**args)
    except BaseException as e:  # a poisoned context must not look like a clean failure
        print(f"__BENCH_FAIL__ {type(e).__name__}: {e}", flush=True)
        return 1
    print("__BENCH__" + json.dumps(asdict(r)), flush=True)
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))


def save(results: list[BenchResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
