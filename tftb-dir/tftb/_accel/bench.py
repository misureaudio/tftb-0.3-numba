# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.0 benchmark harness.

Pure-numpy baseline for the functions Phase 5.1-5.3 will accelerate. The
harness measures median wall-time (over a fixed number of reps, after one
warm-up) of each TFR at a set of signal lengths N, and emits a markdown table
(``docs/benchmarks.md``).

Design (plan §Phase 5, principle 1): a *single* measurement path that the
accelerated backends will be dropped into later. Nothing here mutates the
public API — we simply construct each representation and call ``.run()``
(exactly what the user-facing API does). Every builder is wrapped so a broken
function is recorded as an error row rather than crashing the whole baseline.

The benchmark is *deterministic*: the standard signal is a fixed two-component
signal (constant tone + linear chirp), so timings are comparable across runs
and the numbers in ``benchmarks.md`` are reproducible.
"""

import os
import time

import numpy as np

from tftb.processing.cohen import (
    WignerVilleDistribution,
    PseudoWignerVilleDistribution,
    RihaczekDistribution,
    BornJordanDistribution,
    ChoiWilliamsDistribution,
    GeneralizedRectangularDistribution,
    ZaoAtlasMarksDistribution,
    ButterworthDistribution,
    ReducedInterferenceDistributionBessel,
    ReducedInterferenceDistributionBinomial,
    ReducedInterferenceDistributionHann,
    ReducedInterferenceDistributionTriangular,
    MargenauHillDistribution,
    PseudoMargenauHillDistribution,
    MargenauHillSpectrogramDistribution,
    MinimumMeanCrossEntropyDistribution,
    Spectrogram,
)
from tftb.processing.linear import StankovicDistribution
from tftb.processing.affine import (
    BertrandDistribution,
    DFlandrinDistribution,
    UnterbergerDistribution,
    Scalogram,
)
from tftb.processing.wavelet import cwt
from tftb.processing.cohen import smoothed_pseudo_wigner_ville
from tftb.processing.utils import tftb_window

__all__ = [
    "make_signal",
    "BENCHMARKS",
    "time_function",
    "run_benchmarks",
    "to_markdown",
    "write_report",
]


# --------------------------------------------------------------------------- #
# Standard signal
# --------------------------------------------------------------------------- #
def make_signal(n, seed=0):
    """Deterministic two-component signal of length ``n``.

    A 0.25 tone plus a linear chirp sweeping 0.1 -> 0.4 cycles/sample, plus a
    small deterministic (seeded) noise floor so the signal is genuinely complex
    and not an exact sum of two real exponentials. ``seed`` is only used for the
    noise floor; the tonal content is fixed.
    """
    t = np.arange(n)
    x = (
        0.5 * np.exp(2j * np.pi * 0.25 * t)
        + 0.5 * np.exp(2j * np.pi * (0.1 + 0.3 * t / n) * t)
    )
    rng = np.random.default_rng(seed)
    x = x + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x.astype(np.complex128)


# --------------------------------------------------------------------------- #
# Benchmark definitions
#
# Each entry: (group, name, builder, overrides). ``builder(signal, n)`` returns
# the TFR array we are timing the production of. ``overrides`` may carry
# ``n_values``/``reps`` for a single entry (used for the O(N^2) Scalogram).
#
# Parameter conventions were probed against the port (see notes): the affine
# family is benchmarked at a fixed ``n_voices=256`` (O(N*n_voices)) so the N
# scaling is visible without the O(N^2) memory blow-up the default voice count
# causes at large N.
# --------------------------------------------------------------------------- #
def _run(cls, *args, **kwargs):
    """Build ``cls(*args, **kwargs)`` and return the TFR from ``.run()``."""
    def _b(x, n):
        c = cls(x, *args, **kwargs)
        return c.run()[0]
    return _b


def _mmce(x, n):
    # MMCE windows: each COLUMN is a frequency window of ODD length (the class
    # enforces odd ROWS = window length). Force odd with the codebase idiom
    # (L += 1 - L%2), then stack as columns -> shape (L, 2).
    L = n // 8
    L += 1 - (L % 2)
    w = tftb_window(L, "Hamming")
    return MinimumMeanCrossEntropyDistribution(x, np.column_stack([w, w])).run()[0]


def _spwv(x, n):
    return smoothed_pseudo_wigner_ville(x)[0]


def _cwt(x, n):
    # Fixed n_voices so the N scaling is the signal length, not the voice count.
    return cwt(x, fmin=0.05, fmax=0.45, n_voices=128, wave=2)[0]


def _scalogram(x, n):
    # Scalogram only runs with n_voices ~ N (O(N^2)); capped by n_values below.
    return Scalogram(x, fmin=0.01, fmax=0.49, n_voices=n).run()[0]


BENCHMARKS = [
    ("cohen", "wv", _run(WignerVilleDistribution), {}),
    ("cohen", "pwv", _run(PseudoWignerVilleDistribution), {}),
    ("cohen", "spwv", _spwv, {}),
    ("cohen", "ri", _run(RihaczekDistribution), {}),
    ("cohen", "bj", _run(BornJordanDistribution), {}),
    ("cohen", "cw", _run(ChoiWilliamsDistribution), {}),
    ("cohen", "grd", _run(GeneralizedRectangularDistribution), {}),
    ("cohen", "zam", _run(ZaoAtlasMarksDistribution), {}),
    ("cohen", "bud", _run(ButterworthDistribution), {}),
    ("cohen", "ridb", _run(ReducedInterferenceDistributionBessel), {}),
    ("cohen", "ridbn", _run(ReducedInterferenceDistributionBinomial), {}),
    ("cohen", "ridh", _run(ReducedInterferenceDistributionHann), {}),
    ("cohen", "ridt", _run(ReducedInterferenceDistributionTriangular), {}),
    ("cohen", "mh", _run(MargenauHillDistribution), {}),
    ("cohen", "pmh", _run(PseudoMargenauHillDistribution), {}),
    ("cohen", "mhs", _run(MargenauHillSpectrogramDistribution), {}),
    ("cohen", "stank", _run(StankovicDistribution), {}),
    ("cohen", "mmce", _mmce, {}),
    ("spectrogram", "stft", _run(Spectrogram), {}),
    ("wavelet", "cwt", _cwt, {}),
    ("affine", "bertrand", _run(BertrandDistribution, fmin=0.02, fmax=0.45, n_voices=256), {}),
    ("affine", "dflandrin", _run(DFlandrinDistribution, fmin=0.02, fmax=0.45, n_voices=256), {}),
    ("affine", "unterberger", _run(UnterbergerDistribution, fmin=0.02, fmax=0.45, n_voices=256), {}),
    ("affine", "scalogram", _scalogram, {"n_values": [512, 2048, 4096], "reps": 3}),
]

DEFAULT_N_VALUES = [512, 2048, 4096, 8192]
DEFAULT_REPS = 5
# Per-cell wall-time budget for the timed reps (after warm-up). Heavy tau-loop
# functions (spwv, mhs at large N) exceed this, so they run fewer reps; the
# actual rep count is recorded per cell. Keeps the full baseline bounded.
DEFAULT_MAX_WALL = 20.0


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def time_function(fn, reps=DEFAULT_REPS, max_wall=DEFAULT_MAX_WALL):
    """Return ``(median_wall_time_s, reps_used)`` for ``fn()``.

    One untimed warm-up (compilation/cache warm), then timed reps: up to
    ``reps``, or stop as soon as the accumulated timed wall time exceeds
    ``max_wall`` (heavy functions then report the median over fewer reps).
    """
    fn()  # warm-up
    times = []
    elapsed = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        elapsed += dt
        if elapsed >= max_wall:
            break
    return float(np.median(times)), len(times)


def run_benchmarks(n_values=None, reps=DEFAULT_REPS, max_wall=DEFAULT_MAX_WALL,
                   only=None, verbose=True):
    """Run the full (or a subset of) the benchmark matrix.

    Returns a list of result dicts:
    ``{group, name, n, reps, time_s, status, error}`` where ``reps`` is the
    number of reps actually timed (may be < the target for slow cells).
    ``status`` is ``"ok"`` or ``"error"``; ``error`` is a short message.
    """
    n_values = list(n_values) if n_values else list(DEFAULT_N_VALUES)
    results = []
    for group, name, builder, ov in BENCHMARKS:
        if only is not None and name not in only and group not in only:
            continue
        n_vals = ov.get("n_values", n_values)
        r = ov.get("reps", reps)
        mw = ov.get("max_wall", max_wall)
        for n in n_vals:
            x = make_signal(n)
            row = {"group": group, "name": name, "n": n, "reps": r,
                   "time_s": None, "status": "ok", "error": ""}
            try:
                row["time_s"], row["reps"] = time_function(
                    lambda: builder(x, n), reps=r, max_wall=mw)
            except Exception as e:  # noqa: BLE001 - record, don't crash
                row["status"] = "error"
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:60])
            results.append(row)
            if verbose:
                _print_row(row)
    return results


def _print_row(row):
    if row["status"] == "ok":
        print("[bench] %-11s N=%-5d reps=%d  %8.4f s"
              % (row["name"], row["n"], row["reps"], row["time_s"]))
    else:
        print("[bench] %-11s N=%-5d  ERROR  %s"
              % (row["name"], row["n"], row["error"]))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def to_markdown(results, title="TFTB Python — Phase 5 benchmark baseline"):
    """Render results as a markdown table grouped by N, one row per function."""
    # Pivot: rows = (group, name), columns = N.
    groups_order = []
    for r in results:
        if r["group"] not in groups_order:
            groups_order.append(r["group"])
    n_cols = sorted({r["n"] for r in results})
    lookup = {(r["group"], r["name"], r["n"]): r for r in results}

    lines = []
    lines.append("# %s" % title)
    lines.append("")
    lines.append("Median wall-time (s) of the pure-numpy path. Each cell is one "
                 "warm-up + timed reps (target 5); a cell that exceeds the 20 s "
                 "per-cell budget shows `<sub>×N</sub>` = the fewer reps actually "
                 "timed. Deterministic two-component signal. `—` = not benchmarked "
                 "at that N (infeasible); **ERROR** = the function raised (recorded, "
                 "not a harness bug).")
    lines.append("")
    header = "| function | group | " + " | ".join("N=%d" % n for n in n_cols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(n_cols)))
    for group in groups_order:
        for name in [nm for (g, nm, _, _) in BENCHMARKS if g == group]:
            cells = []
            for n in n_cols:
                r = lookup.get((group, name, n))
                if r is None:
                    cells.append("—")
                elif r["status"] == "ok":
                    if r["reps"] == DEFAULT_REPS:
                        cells.append("%.4f" % r["time_s"])
                    else:
                        cells.append("%.4f <sub>×%d</sub>" % (r["time_s"], r["reps"]))
                else:
                    cells.append("**ERROR**")
            lines.append("| `%s` | %s | " % (name, group) + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(results, path, title="TFTB Python — Phase 5 benchmark baseline"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(results, title=title))
    return path


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    quiet = "--quiet" in args
    args = [a for a in args if a != "--quiet"]
    reps = DEFAULT_REPS
    max_wall = DEFAULT_MAX_WALL
    if "--reps" in args:
        i = args.index("--reps")
        reps = int(args[i + 1])
        del args[i:i + 2]
    if "--max-wall" in args:
        i = args.index("--max-wall")
        max_wall = float(args[i + 1])
        del args[i:i + 2]
    only = args or None
    res = run_benchmarks(only=only, reps=reps, max_wall=max_wall,
                         verbose=not quiet)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "..", "docs", "benchmarks.md"))
    write_report(res, out)
    print("\nWrote %s (%d rows)" % (out, len(res)))
