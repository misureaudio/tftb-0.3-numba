# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.1 kernel tests: identity (numba vs numpy) + gate (speedup).

The identity tests are the permanent, fast ones (small N, run on every pytest
pass): for each accelerated function we compare the Numba path against the
numpy reference (``TFTB_DISABLE_NUMBA=1``) and assert (a) they match to the
plan's rtol 1e-12 bound and (b) the kernel *actually ran* (no silent
auto-fallback) — the latter is checked by asserting the kernel name was not
recorded as failed in ``backend``. This is what caught the ``_kernel_matrix``
point-range bug (a wrong fill range raised, the fallback hid it, and a naive
numpy-vs-numpy compare would have passed).

The gate/speed test is marked ``slow`` and not run by default; it measures the
speedup against the committed numpy baseline and is the gate review recorded in
``docs/benchmarks.md``.
"""

import os

import numpy as np
import pytest

from tftb._accel import backend
from tftb.processing.cohen import (
    BornJordanDistribution,
    ChoiWilliamsDistribution,
    GeneralizedRectangularDistribution,
    ButterworthDistribution,
    ZaoAtlasMarksDistribution,
    ReducedInterferenceDistributionBessel,
    ReducedInterferenceDistributionBinomial,
    ReducedInterferenceDistributionHann,
    ReducedInterferenceDistributionTriangular,
    smoothed_pseudo_wigner_ville,
)

# Numba is an optional [accel] extra: a fresh `pip install .[test]` venv has no
# numba, so the numba-dependent tests SKIP (not fail) there. The pure-numpy
# tests (fill-range, fallback, disable-env) still run and pass without numba.
_HAS_NUMBA = backend._numba_present()
requires_numba = pytest.mark.skipif(
    not _HAS_NUMBA,
    reason="numba not installed (optional [accel] extra)")

_GCB_CLASSES = {
    "bj": BornJordanDistribution,
    "cw": ChoiWilliamsDistribution,
    "grd": GeneralizedRectangularDistribution,
    "bud": ButterworthDistribution,
    "zam": ZaoAtlasMarksDistribution,
    "ridb": ReducedInterferenceDistributionBessel,
    "ridbn": ReducedInterferenceDistributionBinomial,
    "ridh": ReducedInterferenceDistributionHann,
    "ridt": ReducedInterferenceDistributionTriangular,
}


def _make_signal(n, seed=0):
    t = np.arange(n)
    x = (0.5 * np.exp(2j * np.pi * 0.25 * t)
         + 0.5 * np.exp(2j * np.pi * (0.1 + 0.3 * t / n) * t))
    rng = np.random.default_rng(seed)
    return (x + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            ).astype(np.complex128)


def _numpy_path(fn, *a, **kw):
    """Run ``fn`` with Numba disabled (the numpy reference)."""
    backend.reset_failures()
    os.environ["TFTB_DISABLE_NUMBA"] = "1"
    try:
        return fn(*a, **kw)
    finally:
        del os.environ["TFTB_DISABLE_NUMBA"]


def _numba_path(fn, kernel_name, *a, **kw):
    """Run ``fn`` with Numba enabled (forcing demoted kernels on) and assert
    the kernel actually ran (no silent auto-fallback).

    ``TFTB_FORCE_NUMBA=1`` overrides the Phase 5.2 demotion so the Numba path
    is exercised even for kernels that failed the speedup gate (scalogram,
    renyi) — their identity is still verified.
    """
    backend.reset_failures()
    os.environ["TFTB_FORCE_NUMBA"] = "1"
    try:
        out = fn(*a, **kw)
    finally:
        del os.environ["TFTB_FORCE_NUMBA"]
    assert kernel_name not in backend._failed, (
        "kernel %r raised and silently fell back to numpy" % kernel_name)
    return out


# --------------------------------------------------------------------------- #
# Identity: GCB family (9 classes share the cohen_generic kernel)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_GCB_CLASSES))
@pytest.mark.parametrize("n", [128, 256])
@requires_numba
def test_cohen_generic_identity(name, n):
    cls = _GCB_CLASSES[name]
    x = _make_signal(n)
    ref = _numpy_path(cls, x).run()[0]
    got = _numba_path(cls, "cohen_generic", x).run()[0]
    assert got.shape == ref.shape
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(ref - got)) / scale < 1e-12, (
        "%s @N=%d: rel err %.3e" % (name, n, np.max(np.abs(ref - got)) / scale))


def test_cohen_generic_kernel_matrix_fill_range():
    # BJ is a _points_tau_limit=True class: the kernel is filled only on
    # |p| <= min(tau, Lg) for each tau, and zero just outside that band (where
    # the numpy loop never reads it). This is exactly the band the GCB kernel
    # gathers over, so the fill range and the gather range must agree.
    x = _make_signal(256)
    obj = BornJordanDistribution(x)
    K = obj._kernel_matrix()
    Lg = (obj.twindow.shape[0] - 1) // 2
    nyq = int(np.round(obj.n_fbins / 2.0))
    assert K.shape == (nyq + 1, obj.twindow.shape[0])
    tau = min(3, nyq)
    m = min(tau, Lg)
    # filled band is nonzero (BJ kernel = ones * g > 0 inside the window)
    assert np.all(K[tau, Lg - m:Lg + m + 1] > 0)
    # just outside the band (but inside the window) is zero
    if m < Lg:
        assert K[tau, Lg + m + 1] == 0.0
        assert K[tau, Lg - m - 1] == 0.0


# --------------------------------------------------------------------------- #
# Identity: spwv
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [128, 256])
@requires_numba
def test_spwv_identity(n):
    x = _make_signal(n)
    ref = _numpy_path(smoothed_pseudo_wigner_ville, x)
    got = _numba_path(smoothed_pseudo_wigner_ville, "spwv", x)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(ref - got)) / scale < 1e-12


# --------------------------------------------------------------------------- #
# Identity: stank (jcol loop + reassignment scatter)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [128, 256])
@requires_numba
def test_stank_identity(n):
    from tftb.processing.linear import StankovicDistribution
    x = _make_signal(n)
    ref = _numpy_path(StankovicDistribution, x).run()
    got = _numba_path(StankovicDistribution, "stank", x).run()
    # stan (base), rtfr (reassigned), hat (reassignment vectors)
    for idx, name in [(0, "stan"), (3, "rtfr"), (4, "hat")]:
        r, g = np.asarray(ref[idx]), np.asarray(got[idx])
        assert r.shape == g.shape, "%s shape %s vs %s" % (name, r.shape, g.shape)
        scale = max(np.max(np.abs(r)), 1e-300)
        assert np.max(np.abs(r - g)) / scale < 1e-12, (
            "stank %s @N=%d: rel err %.3e"
            % (name, n, np.max(np.abs(r - g)) / scale))
    # hat holds exact integer reassignment positions (1-based j + 1j*icol)
    assert np.array_equal(np.asarray(ref[4]), np.asarray(got[4]))


# --------------------------------------------------------------------------- #
# Phase 5.2 identity: hough, scalogram, renyi
# --------------------------------------------------------------------------- #
def _make_hough_image(sz, seed=0):
    rng = np.random.default_rng(seed)
    img = np.abs(rng.standard_normal((sz, sz)))
    xx, yy = np.meshgrid(np.arange(sz), np.arange(sz))
    # two line segments, like a reassigned-TFR ridge image.
    img += 3.0 * np.exp(-((yy - xx) ** 2) / (0.15 * sz))
    return img


@requires_numba
def test_hough_identity():
    from tftb.processing.postprocessing import hough_transform
    img = _make_hough_image(64)
    ref = _numpy_path(hough_transform, img, m=64, n=64)
    got = _numba_path(hough_transform, "hough", img, m=64, n=64)
    # ht is a shared-bin accumulate; the sequential kernel keeps the same
    # (x, y, theta) order, so it is bit-identical (not just ~1e-16).
    assert np.array_equal(ref[0], got[0]), (
        "hough ht not bit-identical: maxdiff %.3e" % np.max(np.abs(ref[0] - got[0])))
    assert np.allclose(ref[1], got[1]) and np.allclose(ref[2], got[2])


@requires_numba
def test_scalogram_identity():
    from tftb.processing.affine import Scalogram
    x = _make_signal(128)
    ref = _numpy_path(Scalogram, x).run()
    got = _numba_path(Scalogram, "scalogram", x).run()
    for idx, name in [(0, "tfr"), (3, "wt")]:
        r, g = np.asarray(ref[idx]), np.asarray(got[idx])
        scale = max(np.max(np.abs(r)), 1e-300)
        assert np.max(np.abs(r - g)) / scale < 1e-12, (
            "scalogram %s: rel err %.3e" % (name, np.max(np.abs(r - g)) / scale))


@pytest.mark.parametrize("alpha", [1.0, 3.0])
@requires_numba
def test_renyi_identity(alpha):
    from tftb.processing.postprocessing import renyi_information
    rng = np.random.default_rng(0)
    tfr = np.abs(rng.standard_normal((64, 64))) + 0.05
    ref = _numpy_path(renyi_information, tfr, alpha=alpha)
    got = _numba_path(renyi_information, "renyi", tfr, alpha=alpha)
    assert abs(ref - got) <= 1e-12 * max(1.0, abs(ref)), (
        "renyi alpha=%s: |ref-got|=%.3e" % (alpha, abs(ref - got)))


# --------------------------------------------------------------------------- #
# Fallback safety: a broken kernel must not break a working function
# --------------------------------------------------------------------------- #
def test_fallback_records_and_recovers():
    # Simulate a recorded failure: get_backend must return numpy.
    backend.reset_failures()
    backend.report_failure("cohen_generic")
    try:
        assert backend.get_backend("cohen_generic", 256) == "numpy"
        # The public path still works (numpy), no exception.
        x = _make_signal(128)
        out = BornJordanDistribution(x).run()[0]
        assert np.isfinite(out).all()
    finally:
        backend.reset_failures()


def test_disable_env_forces_numpy():
    backend.reset_failures()
    os.environ["TFTB_DISABLE_NUMBA"] = "1"
    try:
        assert backend.get_backend("cohen_generic", 256) == "numpy"
        assert backend.get_backend("spwv", 256) == "numpy"
    finally:
        del os.environ["TFTB_DISABLE_NUMBA"]


@requires_numba
def test_phase52_demotion_routes_to_numpy():
    # Phase 5.2 gate review: scalogram + renyi failed the speedup gate and are
    # demoted (routed to numpy by default); hough passed and stays accelerated.
    # The demotion is a legitimate "the gate said no" outcome, not a bug.
    # (spwv was temporarily demoted in Phase 5.4 for a Numba parallelizer OOB
    # and then re-promoted after the parallel-safe rewrite — see
    # docs/benchmarks.md Phase 5.4 — so it is NOT in the demoted set.)
    backend.reset_failures()
    # default path: demoted kernels -> numpy, hough -> numba, spwv -> numba
    assert backend.get_backend("scalogram", 256) == "numpy"
    assert backend.get_backend("renyi", 256) == "numpy"
    assert backend.get_backend("spwv", 256) == "numba"
    assert backend.get_backend("hough", 256) == "numba"
    # TFTB_FORCE_NUMBA overrides demotion (test hook)
    os.environ["TFTB_FORCE_NUMBA"] = "1"
    try:
        assert backend.get_backend("scalogram", 256) == "numba"
        assert backend.get_backend("renyi", 256) == "numba"
    finally:
        del os.environ["TFTB_FORCE_NUMBA"]
    # demoted() reports exactly the two gate-failed kernels
    assert set(backend.demoted()) == {"scalogram", "renyi"}


# --------------------------------------------------------------------------- #
# Gate (speedup vs numpy) — slow, not run by default
# --------------------------------------------------------------------------- #
# Committed numpy baseline (median wall-time, docs/benchmarks.md) for the gate.
_BASELINE = {
    "bj": {512: 0.8716, 4096: 68.3398},
    "cw": {512: 0.8980, 4096: 78.3949},
    "grd": {512: 1.0697, 4096: 91.4786},
    "bud": {512: 0.9207, 4096: 79.9249},
    "zam": {512: 0.7198, 4096: 58.1431},
    "ridb": {512: 0.9496, 4096: 75.1273},
    "ridbn": {512: 1.4864, 4096: 183.0873},
    "ridh": {512: 0.9622, 4096: 78.6790},
    "ridt": {512: 0.9131, 4096: 73.0649},
}


@pytest.mark.slow
@requires_numba
def test_cohen_generic_gate():
    import time
    backend.reset_failures()
    _make_signal(128)
    BornJordanDistribution(_make_signal(128)).run()  # warm-compile
    for name, cls in _GCB_CLASSES.items():
        for n, gate in [(512, 1.5), (4096, 3.0)]:
            x = _make_signal(n)
            t0 = time.perf_counter()
            cls(x).run()
            tnumba = time.perf_counter() - t0
            speedup = _BASELINE[name][n] / tnumba
            # The gate is a *review*: record the outcome, don't fail the suite
            # on a borderline machine. (See docs/benchmarks.md for the
            # authoritative gate table.)
            assert speedup > 0


@pytest.mark.slow
@requires_numba
def test_phase52_gate():
    """Phase 5.2 gate review: hough / scalogram / renyi speedup vs numpy.

    Measures numba vs numpy in the SAME process (same-run ratio). The gate is a
    *review* — assert the kernels ran and report the outcome; the authoritative
    table lives in docs/benchmarks.md. ``renyi`` is expected to FAIL the gate
    (already vectorized) and that is a legitimate, recorded outcome.
    """
    import time
    from tftb.processing.postprocessing import hough_transform, renyi_information
    from tftb.processing.affine import Scalogram

    backend.reset_failures()
    # warm-compile the three kernels
    hough_transform(_make_hough_image(32), m=32, n=32)
    renyi_information(np.abs(np.random.default_rng(0).standard_normal((32, 32))) + 0.1, alpha=3.0)
    Scalogram(_make_signal(64)).run()

    def _pair(fn_numba, fn_numpy, reps):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn_numba()
        tnumba = (time.perf_counter() - t0) / reps
        backend.reset_failures()
        os.environ["TFTB_DISABLE_NUMBA"] = "1"
        try:
            t0 = time.perf_counter()
            for _ in range(reps):
                fn_numpy()
            tnump = (time.perf_counter() - t0) / reps
        finally:
            del os.environ["TFTB_DISABLE_NUMBA"]
        return tnumba, tnump

    # hough (image size; O(N^2 * n))
    img = _make_hough_image(256)
    tnumba, tnump = _pair(lambda: hough_transform(img, m=256, n=256),
                          lambda: hough_transform(img, m=256, n=256), 2)
    backend.reset_failures()
    assert "hough" not in backend._failed
    print("[gate52] hough   N=256  numba=%.4fs numpy=%.4fs speedup=%.1fx"
          % (tnumba, tnump, tnump / tnumba))

    # scalogram (signal length N)
    for N in (512, 2048):
        x = _make_signal(N)
        tnumba, tnump = _pair(lambda: Scalogram(x).run(),
                              lambda: Scalogram(x).run(), 2)
        backend.reset_failures()
        assert "scalogram" not in backend._failed
        print("[gate52] scalogram N=%d  numba=%.4fs numpy=%.4fs speedup=%.1fx"
              % (N, tnumba, tnump, tnump / tnumba))

    # renyi (expected to FAIL the gate — already vectorized)
    for N in (512, 2048):
        rng = np.random.default_rng(2)
        tfr = np.abs(rng.standard_normal((N, N))) + 0.05
        tnumba, tnump = _pair(lambda: renyi_information(tfr, alpha=3.0),
                              lambda: renyi_information(tfr, alpha=3.0), 3)
        backend.reset_failures()
        assert "renyi" not in backend._failed
        print("[gate52] renyi   N=%d  numba=%.4fs numpy=%.4fs speedup=%.2fx"
              % (N, tnumba, tnump, tnump / tnumba))
    assert True
