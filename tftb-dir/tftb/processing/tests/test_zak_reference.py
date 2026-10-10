# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for the Zak transform (Task 4.3).

Value-level equivalence against MATLAB is covered by the golden cases
``zak_1`` (N=256, default divider 16x16) and ``zak_2`` (odd N=315, default
divider 15x21), both at rtol 1e-10. These are faithful transcriptions of the
certified ``tftb-0.3/tests/zakt.m`` (8 tests) and ``izakt.m`` (6 tests), which
validate the transform's *properties*:

* inner-product preservation (unitarity):  ``<sig1, sig2> == <DZT1, DZT2>``
* energy preservation (Parseval):          ``norm(sig)^2 == sum(|DZT|^2)``
* exact round-trip:                        ``izak(zak(sig)) == sig``

The Zak transform is unitary (a column-major reshape, i.e. a permutation,
followed by a length-N FFT scaled by 1/sqrt(N), both unitary), so it preserves
the inner product and the energy, and its inverse reconstructs the signal
exactly when ``N1`` is a multiple of both ``M`` and ``N``.
"""
import numpy as np

from tftb.generators import noisecg, noisecu, fmlin, amgauss
from tftb.processing.zak import zak, izak


def _inner_product_test(sig1, sig2, N=None, M=None):
    """MATLAB ``zakt`` inner-product check:  sig1.'*conj(sig2) ==
    sum(sum(DZT1.*conj(DZT2))).  (Conjugate-transpose convention, exactly as
    the certified test.)"""
    DZT1 = zak(sig1, N, M)
    DZT2 = zak(sig2, N, M)
    # MATLAB  sig1.'*conj(sig2)  :  ' .' ' is the NON-conjugate transpose,
    # so this is  sum(sig1 .* conj(sig2))  (sig1 NOT conjugated).
    sp1 = np.sum(sig1 * np.conj(sig2))
    sp2 = np.sum(DZT1 * np.conj(DZT2))              # sum(sum(DZT1.*conj(DZT2)))
    assert abs(sp1 - sp2) <= np.sqrt(np.finfo(float).eps)


def _energy_test(sig, N=None, M=None):
    """MATLAB ``zakt`` energy (Parseval) check:  norm(sig)^2 == sum(sum(|DZT|^2))."""
    DZT = zak(sig, N, M)
    E1 = float(np.linalg.norm(sig) ** 2)
    E2 = float(np.sum(np.abs(DZT) ** 2))
    assert abs(E1 - E2) <= np.sqrt(np.finfo(float).eps)


def _round_trip(sig, N=None, M=None):
    """MATLAB ``izakt`` round-trip check:  izak(zak(sig)) == sig."""
    DZT = zak(sig, N, M)
    sigr = izak(DZT)
    assert np.abs(sig - sigr).max() <= np.sqrt(np.finfo(float).eps)


# ---- zakt.m: N=256 ----
def test_zak_inner_product_default():
    N = 256
    sig1 = fmlin(N, 0.1, 0.4)[0] * amgauss(N, N // 2, N // 2)
    sig2 = noisecg(N)
    _inner_product_test(sig1, sig2)


def test_zak_inner_product_32_8():
    N = 256
    _inner_product_test(noisecu(N), noisecg(N), N=32, M=8)


def test_zak_inner_product_2_128():
    N = 256
    _inner_product_test(noisecu(N), noisecg(N), N=2, M=128)


def test_zak_energy_default():
    N = 256
    _energy_test(noisecg(N))


# ---- zakt.m: N=315 (odd) ----
def test_zak_inner_product_odd_default():
    N = 5 * 7 * 3
    sig1 = fmlin(N, 0.1, 0.4)[0] * amgauss(N, round(N / 2), round(N / 2))
    sig2 = noisecg(N)
    _inner_product_test(sig1, sig2)


def test_zak_inner_product_odd_35_3():
    N = 5 * 7 * 3
    _inner_product_test(noisecu(N), noisecg(N), N=35, M=3)


def test_zak_inner_product_odd_5_21():
    N = 5 * 7 * 3
    _inner_product_test(noisecu(N), noisecg(N), N=5, M=21)


def test_zak_energy_odd_default():
    N = 5 * 7 * 3
    _energy_test(noisecg(N))


# ---- izakt.m: N=256 ----
def test_zak_round_trip_default():
    _round_trip(noisecg(256))


def test_zak_round_trip_8_32():
    _round_trip(noisecg(256), N=8, M=32)


def test_zak_round_trip_128_2():
    _round_trip(noisecg(256), N=128, M=2)


# ---- izakt.m: N=315 (odd) ----
def test_zak_round_trip_odd_default():
    _round_trip(noisecg(315))


def test_zak_round_trip_odd_9_35():
    _round_trip(noisecg(315), N=9, M=35)


def test_zak_round_trip_odd_3_105():
    _round_trip(noisecg(315), N=3, M=105)
