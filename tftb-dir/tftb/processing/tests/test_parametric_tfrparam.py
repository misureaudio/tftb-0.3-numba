# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for ``tfrparam`` (Task 4.4e).

Value-level equivalence against MATLAB is covered by the golden cases
``tfrparam_1`` (AR / fbhermitian) and ``tfrparam_2`` (periodogram /
fbhermitian). These are structural / consistency invariants:

* the output ``tfr`` has shape ``(N, len(t))`` and ``f`` spans ``[0, 0.5]``;
* the periodogram TFR tracks the instantaneous-frequency law of the test
  signal (peak bin near the IF at each time);
* ``tfrparam`` with ``method='periodogram'`` reduces, per column, to the
  single-window ``parafrep`` of the same window (self-consistency with the
  two lower-level primitives);
* an even ``L`` is corrected to ``L+1`` and ``p < 1`` raises.
"""
import numpy as np
import pytest

from tftb.generators import fmconst, fmlin
from tftb.processing.parametric import correlmx, parafrep, tfrparam


def _signal(n=128):
    # the Python generators return (signal, iflaw); take the signal only.
    return fmconst(n, 0.1)[0] + fmlin(n, 0.15, 0.30)[0] + fmlin(n, 0.2, 0.45)[0]


def test_tfrparam_shape_and_freqs():
    x = _signal(128)
    tfr, t, f = tfrparam(x, np.arange(1, 129), 128, 3, 21, 0, "fbhermitian", "ar")
    assert tfr.shape == (128, 128)
    assert t.shape == (128,)
    assert f.shape == (128,)
    assert f[0] == 0.0 and f[-1] == 0.5
    assert np.all(np.isfinite(tfr))


def test_tfrparam_periodogram_tracks_instantaneous_frequency():
    # a single 0.3-cosine: the periodogram TFR peak bin should sit near 0.3
    # at every time (within a couple of bins for N=128 -> bin spacing 0.5/127).
    # (The periodogram is the stable method for an argmax-based property; the
    # AR method's 1/|Z'A|^2 form produces float-noise spurious peaks.)
    x = fmconst(200, 0.3)[0]
    tfr, t, f = tfrparam(x, np.arange(1, 201), 128, 5, 41, 0, "fbhermitian", "periodogram")
    for icol in (0, 100, 199):
        fpk = f[int(np.argmax(tfr[:, icol]))]
        assert abs(fpk - 0.3) < 0.02, (icol, fpk)


def test_tfrparam_periodogram_equals_single_window_parafrep():
    # self-consistency: for the 'periodogram' method each column of tfrparam
    # equals parafrep(correlmx(window)) for the same centered, clipped window.
    x = _signal(128)
    N, p, L = 128, 15, 61
    Lhalf = (L - 1) // 2
    tfr, t, f = tfrparam(x, np.arange(1, 129), N, p, L, 0, "fbhermitian", "periodogram")
    for icol in (0, 5, 63, 127):
        ti = int(t[icol])
        timin = max(1, ti - Lhalf)
        timax = min(128, ti + Lhalf)
        Realp = min(p, timax - timin)
        Rx = correlmx(x[timin - 1:timax], Realp, "fbhermitian")
        spec, _, _ = parafrep(Rx, N, "periodogram", 2)
        assert np.allclose(tfr[:, icol], spec), icol


def test_tfrparam_even_L_corrected():
    # an even L is corrected to L+1 (odd); the result is finite.
    x = _signal(128)
    tfr, t, f = tfrparam(x, np.arange(1, 129), 128, 3, 20, 0, "fbhermitian", "ar")
    assert np.all(np.isfinite(tfr))
    assert tfr.shape == (128, 128)


def test_tfrparam_p_must_be_positive():
    x = _signal(64)
    with pytest.raises(ValueError, match="p must be greater than 0"):
        tfrparam(x, np.arange(1, 65), 128, 0, 21, 0, "fbhermitian", "ar")


def test_tfrparam_x_must_be_column_vector():
    with pytest.raises(ValueError, match="column vector"):
        tfrparam(np.ones((3, 4)), np.arange(1, 13), 128, 2, 21, 0, "fbhermitian", "ar")
