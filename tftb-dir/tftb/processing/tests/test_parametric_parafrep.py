# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for ``parafrep`` (Task 4.4d).

Value-level equivalence against MATLAB for all six methods is covered by the
golden case ``parafrep_1`` (rtol 1e-9; the looser bound is set by ``genlag``,
whose ``Rx^2`` amplifies the condition number so cross-platform agreement is
~1e-9 relative, while the other five methods + ``A`` match to ~1e-14). The
parametric methods have no dedicated certified test file, so these are
self-consistency / structural invariants:

* ``capnorm = (p+1) * capon`` (identical resolvent, scaled).
* the periodogram peaks at the dominant mode (a 0.2-cosine in the test
  signal -> peak near 0.2).
* the AR prediction vector ``A`` is real for a real-valued process.
* every ``spec`` is real, finite, non-negative; ``freqs`` spans ``[0, 0.5]``.
"""
import numpy as np
import pytest

from tftb.generators import fmconst
from tftb.processing.parametric import correlmx, parafrep


def _rx(p=4):
    """A well-conditioned Hermitian correlation matrix (AR(1)-like): the
    correlation matrix of a 0.2-cosine plus mild noise."""
    rng = np.random.default_rng(0)
    x = fmconst(200, 0.2)[0].real + 0.1 * rng.standard_normal(200)
    return correlmx(x, p, "hermitian")


def test_parafrep_capnorm_is_p1_times_capon():
    Rx = _rx(4)
    spec_capon, _, _ = parafrep(Rx, 128, "capon", 2)
    spec_capnorm, _, _ = parafrep(Rx, 128, "capnorm", 2)
    assert np.allclose(spec_capnorm, 5.0 * spec_capon)   # p+1 = 5


def test_parafrep_periodogram_peaks_near_dominant_mode():
    Rx = _rx(4)
    spec, freqs, _ = parafrep(Rx, 128, "periodogram", 2)
    fpk = freqs[int(np.argmax(spec))]
    assert abs(fpk - 0.2) < 0.02, fpk


def test_parafrep_ar_coefficient_real():
    Rx = _rx(4)
    _, _, A = parafrep(Rx, 128, "ar", 2)
    assert np.abs(A.imag).max() < 1e-12
    assert np.all(np.isfinite(A.real))


def test_parafrep_spec_real_finite_nonnegative():
    Rx = _rx(4)
    for method in ("ar", "periodogram", "capon", "capnorm", "lagunas", "genlag"):
        spec, freqs, _ = parafrep(Rx, 128, method, 2)
        assert np.all(np.isfinite(spec)), method
        assert np.all(spec >= 0.0), method
        assert freqs[0] == 0.0 and freqs[-1] == 0.5
        assert spec.shape == (128,)


def test_parafrep_ar_only_method_returns_A():
    Rx = _rx(4)
    for method in ("periodogram", "capon", "capnorm", "lagunas", "genlag"):
        _, _, A = parafrep(Rx, 128, method, 2)
        assert A is None, method
    _, _, A = parafrep(Rx, 128, "ar", 2)
    assert A is not None and A.shape == (5,)


def test_parafrep_unknown_method_raises():
    Rx = _rx(2)
    with pytest.raises(ValueError, match="unknown frequency representation"):
        parafrep(Rx, 128, "notamethod", 2)


def test_parafrep_rx_must_be_square():
    with pytest.raises(ValueError, match="square"):
        parafrep(np.ones((3, 4)), 128, "capon", 2)
