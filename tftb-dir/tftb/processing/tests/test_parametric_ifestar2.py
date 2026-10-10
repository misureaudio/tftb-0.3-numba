# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for ``ifestar2`` (Task 4.4c).

Value-level equivalence against MATLAB is covered by the golden cases
``ifestar2_1`` (linear chirp) and ``ifestar2_2`` (constant tone), both at
rtol 1e-10. These are faithful transcriptions of the certified
``tftb-0.3/tests/ifestart.m`` (5 tests): the 4-point Prony AR(2) estimator
recovers the instantaneous-frequency law of each signal class to the
documented tolerance (the estimator has an approximate 2.5-point delay, so
the constant/chirp recoveries are tight and the AM-FM ones looser).

MATLAB convention: the generator's IF law ``ifl`` is 1-based in the test
(``ifl(t)``), so it is read as ``iflaw[t-1]`` in 0-based Python; ``t`` is the
1-based time instant returned by ``ifestar2``.
"""
import numpy as np

from tftb.generators import fmconst, fmlin, fmsin
from tftb.processing.parametric import ifestar2


def _ifl_at(iflaw, t):
    """MATLAB ``ifl(t)`` (1-based) -> 0-based ``iflaw[t-1]``."""
    return iflaw[np.asarray(t) - 1]


def test_ifestar2_const_default():
    N = 235
    sig, ifl = fmconst(N)
    fnorm, t, ratio = ifestar2(sig.real)
    assert np.abs(_ifl_at(ifl, t) - fnorm).max() <= np.sqrt(np.finfo(float).eps)


def test_ifestar2_const_low():
    N = 235
    sig, ifl = fmconst(N, 0.01)
    fnorm, t, ratio = ifestar2(sig.real)
    assert np.abs(_ifl_at(ifl, t) - fnorm).max() <= np.sqrt(np.finfo(float).eps)


def test_ifestar2_linear_chirp():
    N = 235
    sig, ifl = fmlin(N, 0.05, 0.45)
    fnorm, t, ratio = ifestar2(sig.real)
    assert np.abs(_ifl_at(ifl, t) - fnorm).max() <= 1e-2


def test_ifestar2_am_fm_default():
    N = 235
    sig, ifl = fmsin(N)
    fnorm, t, ratio = ifestar2(sig.real)
    assert np.abs(_ifl_at(ifl, t) - fnorm).max() <= 1e-1


def test_ifestar2_am_fm_long_period():
    N = 235
    sig, ifl = fmsin(N, 0.03, 0.35, 2 * N)
    fnorm, t, ratio = ifestar2(sig.real)
    assert np.abs(_ifl_at(ifl, t) - fnorm).max() <= 1e-2


def test_ifestar2_rejratio_and_range():
    # On a clean tone the estimator succeeds at every instant (ratio == 1),
    # the estimate lies in [0, 0.5], and the returned t is 4..N.
    N = 128
    sig, ifl = fmconst(N, 0.2)
    fnorm, t, ratio = ifestar2(sig.real)
    assert ratio == 1.0
    assert np.all((fnorm >= 0.0) & (fnorm <= 0.5))
    assert np.array_equal(t, np.arange(4, N + 1))


def test_ifestar2_t_bounds():
    sig, _ = fmconst(64, 0.2)
    try:
        ifestar2(sig.real, np.array([1, 2, 3]))
        assert False, "expected ValueError for t < 4"
    except ValueError:
        pass
