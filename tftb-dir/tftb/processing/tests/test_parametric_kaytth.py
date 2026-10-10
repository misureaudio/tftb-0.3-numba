# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for ``kaytth`` (Task 4.4b).

Value-level equivalence against MATLAB is covered by the golden case
``kaytth_1`` (rtol 1e-10). The shipped ``kayttht.m`` is an empty stub (O.
Lemoine, August 1996), so these are the property tests:

* the closed form  ``H[i] = (L(L+1) - i(i-1)) / (2 L (L+1) (2L+1) / 3)``
* the kernel is positive and monotonically decreasing
* the kernel sums to exactly ``1/2`` (the factor of 2 in the denominator)
"""
import numpy as np

from tftb.processing.parametric import kaytth


def _closed_form(length):
    """Independent evaluation of the Kay-Tretter closed form."""
    n = length
    pp1 = n * (n + 1)
    den = 2.0 * n * (n + 1) * (2.0 * n + 1.0) / 3.0
    return (pp1 - np.arange(1, n + 1) * np.arange(0, n)) / den


def test_kaytth_closed_form():
    for L in (1, 2, 5, 16, 33, 100):
        assert np.allclose(kaytth(L), _closed_form(L)), L


def test_kaytth_sums_to_half():
    # The denominator carries a factor of 2, so the kernel sums to 1/2.
    for L in (1, 2, 5, 16, 33, 100):
        assert abs(kaytth(L).sum() - 0.5) < 1e-15, L


def test_kaytth_positive_and_decreasing():
    for L in (2, 5, 16, 33):
        H = kaytth(L)
        assert np.all(H > 0), L
        assert np.all(np.diff(H) < 0), L   # strictly decreasing


def test_kaytth_length_and_endpoint():
    H = kaytth(8)
    assert H.shape == (8,)
    # H[1] = L(L+1) - 0 over den; for L=8: 72 / (2*8*9*17/3) = 72/816.
    assert abs(H[0] - 72.0 / 816.0) < 1e-15
