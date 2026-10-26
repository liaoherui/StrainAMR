"""Timestamped logging and per-step timing for the StrainAMR pipeline.

Every line carries the wall-clock time, so a long run can be read back afterwards to see where it
spent its time, and each phase is wrapped in a timer whose totals are printed as a table at the end
and written to `<outdir>/timing.tsv`.

    from steplog import log, step, summary

    with step('gene calling + RGI'):
        ...
    summary(outdir)
"""

from __future__ import annotations

import contextlib
import os
import time

_STEPS = []
_START = time.time()


def log(msg: str, **_print_kwargs) -> None:
    """Timestamped line. Extra print-style keywords (flush=, file=) are accepted and ignored, so a
    `print(..., flush=True)` can be swapped for `log(...)` without editing every call site."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


@contextlib.contextmanager
def step(name: str, note: str = ""):
    """Time one phase, logging its start and end; skipped phases are recorded too, at 0s."""
    log(f"START  {name}" + (f"  ({note})" if note else ""))
    t0 = time.time()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        d = time.time() - t0
        _STEPS.append((name, d, failed))
        log(f"{'FAILED' if failed else 'DONE  '} {name} in {fmt_time(d)}")


def skipped(name: str) -> None:
    _STEPS.append((name, 0.0, None))


def summary(outdir: str = None, title: str = "Step timings") -> str:
    """Print the per-step totals and, given an output directory, write them to timing.tsv."""
    if not _STEPS:
        return ""
    total = time.time() - _START
    width = max(len(n) for n, _, _ in _STEPS)
    lines = ["", f"{title} (total {fmt_time(total)}):"]
    for name, d, failed in _STEPS:
        if failed is None:
            lines.append(f"  {name.ljust(width)}  {'skipped':>9}")
            continue
        share = 100 * d / total if total else 0
        bar = "#" * max(0, min(20, int(round(share / 5))))
        lines.append(f"  {name.ljust(width)}  {fmt_time(d):>9}  {share:5.1f}%  {bar}"
                     + ("   FAILED" if failed else ""))
    text = "\n".join(lines)
    print(text, flush=True)
    if outdir:
        try:
            with open(os.path.join(outdir, "timing.tsv"), "w") as fh:
                fh.write("step\tseconds\tstatus\n")
                for name, d, failed in _STEPS:
                    status = "skipped" if failed is None else ("failed" if failed else "ok")
                    fh.write(f"{name}\t{d:.1f}\t{status}\n")
                fh.write(f"TOTAL\t{total:.1f}\tok\n")
        except OSError:
            pass
    return text


def reset() -> None:
    _STEPS.clear()
    global _START
    _START = time.time()
