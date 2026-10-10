# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Tests for the Phase 5.0 benchmark harness (``tftb/_accel/bench.py``).

The *full* baseline is a CLI job (``python -m tftb._accel.bench``) because the
heavy tau-loop functions take minutes; it is not run on every pytest pass. These
tests exercise the harness machinery on a small, fast subset: that the standard
signal is deterministic and well-formed, that timing returns a finite positive
median with a sane rep count, that a broken builder is recorded (not raised),
that the markdown/report path round-trips, and that the gate-threshold helper
behaves.
"""

import numpy as np

from tftb._accel import bench


# --------------------------------------------------------------------------- #
# Standard signal
# --------------------------------------------------------------------------- #
def test_make_signal_deterministic_and_wellformed():
    a = bench.make_signal(256, seed=0)
    b = bench.make_signal(256, seed=0)
    assert a.shape == (256,)
    assert a.dtype == np.complex128
    assert np.array_equal(a, b)          # deterministic
    assert not np.array_equal(a, bench.make_signal(256, seed=1))  # seed matters
    # genuinely complex (nonzero imag) and finite
    assert np.all(np.isfinite(a))
    assert np.max(np.abs(np.imag(a))) > 0


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def test_time_function_returns_median_and_reps():
    def fn():
        return bench._run(bench.WignerVilleDistribution)(bench.make_signal(128), 128)

    median, reps_used = bench.time_function(fn, reps=3, max_wall=30.0)
    assert reps_used == 3
    assert median > 0.0
    assert np.isfinite(median)


def test_time_function_respects_wall_budget():
    # max_wall=0.0 forces the loop to break after the first timed rep, so we
    # get exactly 1 rep regardless of how fast the function is.
    def fn():
        return bench.make_signal(64)

    median, reps_used = bench.time_function(fn, reps=5, max_wall=0.0)
    assert reps_used == 1
    assert median >= 0.0 and np.isfinite(median)

    # A generous budget runs the full target rep count.
    median, reps_used = bench.time_function(fn, reps=5, max_wall=10.0)
    assert reps_used == 5


# --------------------------------------------------------------------------- #
# run_benchmarks (small fast subset)
# --------------------------------------------------------------------------- #
def test_run_benchmarks_subset_ok():
    res = bench.run_benchmarks(n_values=[128, 256], reps=2,
                               only=["cwt", "stft", "ri"], verbose=False)
    # 3 functions x 2 N values
    assert len(res) == 6
    for r in res:
        assert r["status"] == "ok"
        assert r["time_s"] > 0.0 and np.isfinite(r["time_s"])
        assert r["reps"] >= 1


def test_run_benchmarks_records_error_not_raises():
    # Point the harness at a builder that raises, to confirm error recording.
    orig = list(bench.BENCHMARKS)
    try:
        bench.BENCHMARKS[:] = [
            ("cohen", "boom", lambda x, n: 1 / 0, {}),
        ]
        res = bench.run_benchmarks(n_values=[128], reps=1, verbose=False)
        assert len(res) == 1
        assert res[0]["status"] == "error"
        assert "ZeroDivision" in res[0]["error"]
    finally:
        bench.BENCHMARKS[:] = orig


def test_run_benchmarks_only_filters_by_name_or_group():
    res = bench.run_benchmarks(n_values=[128], reps=1,
                               only=["wavelet"], verbose=False)
    assert {r["name"] for r in res} == {"cwt"}
    assert {r["group"] for r in res} == {"wavelet"}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_to_markdown_roundtrip(tmp_path):
    res = bench.run_benchmarks(n_values=[128, 256], reps=2,
                               only=["cwt", "stft"], verbose=False)
    md = bench.to_markdown(res)
    assert "# " in md and "| function |" in md
    # both functions appear, both N columns present
    assert "`cwt`" in md and "`stft`" in md
    assert "N=128" in md and "N=256" in md
    out = bench.write_report(res, str(tmp_path / "bench.md"))
    assert open(out, encoding="utf-8").read() == md


def test_all_benchmark_builders_exist_and_are_callable():
    # The registry is the contract for 5.1-5.3; every entry must be a callable
    # builder(signal, n).
    for group, name, builder, ov in bench.BENCHMARKS:
        assert isinstance(group, str) and isinstance(name, str)
        assert callable(builder)
        assert isinstance(ov, dict)
    # no duplicate (group, name)
    keys = [(g, n) for g, n, _, _ in bench.BENCHMARKS]
    assert len(keys) == len(set(keys))
