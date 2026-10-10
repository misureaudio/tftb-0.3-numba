# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Phase 5.3bis CuPy tests (second explicit API).

The ``*_cupy`` functions are an *explicit opt-in* (the name is the selector —
no env vars, no dispatch). These tests:

* skip (not fail) when CuPy is absent or no CUDA device is present;
* verify identity of each ``*_cupy`` against the standard function at the gate
  size (N = 8192) — marked ``slow`` (not run by default);
* verify the guards: N < 8192 raises, a missing device / CuPy raises, and
  there is **no silent fallback** (an explicit CuPy call runs on CuPy or
  raises).

The Numba path and the standard functions are untouched by this module.
"""

import numpy as np
import pytest

from tftb._accel import cupy_kernels as _ck

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover - CuPy is an optional extra
    _HAS_CUPY = False

_HAS_DEVICE = _HAS_CUPY and _cp.cuda.runtime.getDeviceCount() > 0

requires_cupy = pytest.mark.skipif(
    not (_HAS_CUPY and _HAS_DEVICE),
    reason="CuPy / CUDA device not available")


def _make_signal(n, seed=0):
    t = np.arange(n)
    x = (0.5 * np.exp(2j * np.pi * 0.25 * t)
         + 0.5 * np.exp(2j * np.pi * (0.1 + 0.3 * t / n) * t))
    rng = np.random.default_rng(seed)
    return (x + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            ).astype(np.complex128)


# --------------------------------------------------------------------------- #
# Guards (run without a device: the size guard fires before the device check)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [_ck.cwt_cupy, _ck.spwv_cupy,
                                _ck.spectrogram_cupy])
def test_size_guard_below_8192(fn):
    # N = 4096 < 8192 must raise, regardless of whether a device is present.
    x = _make_signal(4096)
    with pytest.raises(ValueError):
        fn(x)


@requires_cupy
@pytest.mark.parametrize("fn", [_ck.cwt_cupy, _ck.spwv_cupy,
                                _ck.spectrogram_cupy])
def test_no_silent_fallback_on_device_error(fn, monkeypatch):
    # Force "no device": a real CuPy call must RAISE, never fall back to
    # numpy/numba.
    monkeypatch.setattr(_ck, "_cupy", lambda: _cp)
    real_count = _cp.cuda.runtime.getDeviceCount

    def _no_device():
        return 0

    monkeypatch.setattr(_cp.cuda.runtime, "getDeviceCount", _no_device)
    x = _make_signal(8192)
    with pytest.raises(RuntimeError):
        fn(x)
    monkeypatch.setattr(_cp.cuda.runtime, "getDeviceCount", real_count)


# --------------------------------------------------------------------------- #
# Identity vs the standard function at the gate size (slow)
# --------------------------------------------------------------------------- #
@requires_cupy
@pytest.mark.slow
def test_spectrogram_cupy_identity():
    from tftb.processing.cohen import Spectrogram
    n = 8192
    x = _make_signal(n)
    ref = Spectrogram(x).run()
    got = _ck.spectrogram_cupy(x)
    assert got[0].shape == ref[0].shape
    scale = np.max(np.abs(ref[0]))
    assert np.max(np.abs(ref[0] - got[0])) / scale < 1e-10, (
        "spectrogram_cupy rel err %.3e" % (np.max(np.abs(ref[0] - got[0])) / scale))
    assert np.allclose(ref[2], got[2])


@requires_cupy
@pytest.mark.slow
def test_spwv_cupy_identity():
    from tftb.processing.cohen import smoothed_pseudo_wigner_ville
    n = 8192
    x = _make_signal(n)
    ref = smoothed_pseudo_wigner_ville(x)
    got = _ck.spwv_cupy(x)
    assert got.shape == ref.shape
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(ref - got)) / scale < 1e-10, (
        "spwv_cupy rel err %.3e" % (np.max(np.abs(ref - got)) / scale))


@requires_cupy
@pytest.mark.slow
def test_cwt_cupy_identity():
    from tftb.processing.wavelet import cwt
    n = 8192
    x = _make_signal(n)
    ref = cwt(x)
    got = _ck.cwt_cupy(x)
    # compare wt (the coefficients); scalo = |wt|^2 follows from it.
    for idx in (0, 1, 3, 4):
        r, g = np.asarray(ref[idx]), np.asarray(got[idx])
        assert r.shape == g.shape, "cwt_cupy %d shape %s vs %s" % (idx, r.shape, g.shape)
        scale = max(np.max(np.abs(r)), 1e-300)
        assert np.max(np.abs(r - g)) / scale < 1e-9, (
            "cwt_cupy idx %d rel err %.3e" % (idx, np.max(np.abs(r - g)) / scale))
