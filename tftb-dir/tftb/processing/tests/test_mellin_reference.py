# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Reference / property tests for the fast Mellin transform (Task 4.2).

Value-level equivalence against MATLAB is covered by the golden case
``fmt_1`` (L1, rtol 1e-10). These are faithful transcriptions of the
certified ``tftb-0.3/tests/fmtt.m`` (8 tests), which validate the
transform's *properties*:

1. round-trip ``ifmt(fmt(x)) == x`` (analytic, fmax = 0.5)
2. energy conservation  ``norm(mellin)^2 == in-band signal energy``
3. cross-correlation    ``<mellin1, mellin2> == <spec1, spec2>`` (in-band)
4. multiplicative       ``mellin(conv(x1,x2)) == conv(mellin1, mellin2)``
   (the Fourier duality inherited by the Mellin transform)
5-8. the same four, at odd length N=121 (N+1 even voices), looser tolerance.

A key MATLAB convention these depend on: ``Z.'`` is the NON-conjugate
transpose (``Z'`` would be the conjugate-transpose), so ``fmt``'s
``Z.'*tfmatx`` is ``Z @ tfmatx`` with no conjugation.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, fmlin, fmsin
from tftb.processing.mellin import fmt, ifmt


def _band_energy(x, fmin, fmax, N, hilbert=False):
    """In-band energy  sum(|SP[indmin:indmax]|^2 / nu),  nu = idx/N.

    Mirrors the normalization band used by ``fmt``.  ``hilbert=False`` uses
    the raw fft (the certified test 2); the transform itself normalizes with
    the analytic signal, but for a pure positive-frequency sinusoid
    ``hilbert(real(x)) == x`` so the two agree.
    """
    SP = np.fft.fft(x)
    indmin = 1 + int(round(fmin * (x.shape[0] - 2)))
    indmax = 1 + int(round(fmax * (x.shape[0] - 2)))
    SPana = SP[indmin - 1:indmax]
    nu = np.arange(indmin, indmax + 1) / N
    SPp = SPana / nu
    return float(np.vdot(SPp, SPana).real)


# ---- 1. round-trip (fmt test 1) ----
def test_mellin_round_trip():
    N, fmin, fmax = 128, 0.1, 0.5
    sig = amgauss(N) * fmconst(N, 0.3)[0]
    mellin, beta = fmt(sig, fmin, fmax, N)
    X = ifmt(mellin, beta, N)
    assert np.abs(X - sig).max() <= 1e-7


# ---- 2. energy conservation (fmt test 2) ----
def test_mellin_energy_conservation():
    N, fmin, fmax = 128, 0.1, 0.4
    x = fmconst(N)[0]
    mellin, _ = fmt(x, fmin, fmax, N)
    Es = _band_energy(x, fmin, fmax, N)
    Efmt = float(np.linalg.norm(mellin) ** 2)
    assert abs(Es - Efmt) <= np.sqrt(np.finfo(float).eps)


# ---- 3. cross-correlation (fmt test 3) ----
def test_mellin_cross_correlation():
    N, fmin, fmax = 128, 0.01, 0.49
    x1 = amgauss(N) * fmlin(N, 0.15, 0.35)[0]
    x2 = amgauss(N) * fmconst(N)[0]
    FMT1, _ = fmt(x1, fmin, fmax, 2 * N)
    FMT2, _ = fmt(x2, fmin, fmax, 2 * N)
    indmin = 1 + int(round(fmin * (2 * N - 2)))
    indmax = 1 + int(round(fmax * (2 * N - 2)))
    SP1, SP2 = np.fft.fft(x1), np.fft.fft(x2)
    # MATLAB:  nu = (indmin:indmax)'/N/2  = idx/(2N)  (left-associative).
    nu = np.arange(indmin, indmax + 1) / (2 * N)
    SP1p = SP1[indmin - 1:indmax] / nu
    # MATLAB:  cor1 = SP1p'*SP2(band)  and  cor2 = conj(FMT1*FMT2')  -- the
    # plain ' is the CONJUGATE transpose, so both are vdot (conjugated).
    cor1 = np.vdot(SP1p, SP2[indmin - 1:indmax])
    cor2 = np.vdot(FMT1, FMT2)
    assert abs(cor1 - cor2) <= N * 1e-2


# ---- 4. multiplicative property (fmt test 4) ----
def test_mellin_multiplicative():
    N, fmin, fmax = 128, 0.01, 0.49
    x1 = amgauss(N) * fmlin(N, 0.15, 0.35)[0]
    x2 = amgauss(N) * fmsin(N, 0.15, 0.35)[0]
    FMT1, _ = fmt(x1, fmin, fmax, 2 * N)
    FMT2, _ = fmt(x2, fmin, fmax, 2 * N)
    FMT = np.convolve(FMT1, FMT2)
    FMT = FMT / np.max(FMT.real)
    X1, X2 = np.fft.fft(x1), np.fft.fft(x2)
    x = np.fft.fftshift(np.fft.ifft(X1 * X2))
    FMTp, _ = fmt(x, fmin, fmax, 2 * N)
    FMTp = FMTp / np.max(FMTp.real)
    Diff = FMTp - FMT[N:3 * N]          # MATLAB FMT(N+1:3*N), 1-based
    assert np.abs(Diff).max() <= 1e-4


# ---- 5-8. odd-length (N=121) versions, looser tolerance ----
def test_mellin_round_trip_odd():
    N, fmin, fmax = 121, 0.1, 0.5
    sig = amgauss(N) * fmconst(N, 0.3)[0]
    mellin, beta = fmt(sig, fmin, fmax, N + 1)
    X = ifmt(mellin, beta, N)
    assert np.abs(X - sig).max() <= 5e-2


def test_mellin_energy_conservation_odd():
    N, fmin, fmax = 121, 0.1, 0.4
    x = fmconst(N)[0]
    mellin, _ = fmt(x, fmin, fmax, N + 1)
    Es = _band_energy(x, fmin, fmax, N + 1)
    Efmt = float(np.linalg.norm(mellin) ** 2)
    assert abs(Es - Efmt) <= 5e-2 * abs(Es)


def test_mellin_cross_correlation_odd():
    N, fmin, fmax = 121, 0.01, 0.49
    x1 = amgauss(N) * fmlin(N, 0.15, 0.35)[0]
    x2 = amgauss(N) * fmconst(N)[0]
    FMT1, _ = fmt(x1, fmin, fmax, 2 * N)
    FMT2, _ = fmt(x2, fmin, fmax, 2 * N)
    indmin = 1 + int(round(fmin * (2 * N - 2)))
    indmax = 1 + int(round(fmax * (2 * N - 2)))
    SP1, SP2 = np.fft.fft(x1), np.fft.fft(x2)
    # MATLAB:  nu = (indmin:indmax)'/N/2  = idx/(2N)  (left-associative).
    nu = np.arange(indmin, indmax + 1) / (2 * N)
    SP1p = SP1[indmin - 1:indmax] / nu
    # MATLAB:  cor1 = SP1p'*SP2(band)  and  cor2 = conj(FMT1*FMT2')  (vdot).
    cor1 = np.vdot(SP1p, SP2[indmin - 1:indmax])
    cor2 = np.vdot(FMT1, FMT2)
    # MATLAB tolerance:  1e-1 * norm(cor1).
    assert abs(cor1 - cor2) <= 1e-1 * abs(cor1)


def test_mellin_multiplicative_odd():
    N, fmin, fmax = 121, 0.01, 0.49
    x1 = amgauss(N) * fmlin(N, 0.15, 0.35)[0]
    x2 = amgauss(N) * fmsin(N, 0.15, 0.35)[0]
    FMT1, _ = fmt(x1, fmin, fmax, 2 * N)
    FMT2, _ = fmt(x2, fmin, fmax, 2 * N)
    FMT = np.convolve(FMT1, FMT2)
    FMT = FMT / np.max(FMT.real)
    X1, X2 = np.fft.fft(x1), np.fft.fft(x2)
    x = np.fft.fftshift(np.fft.ifft(X1 * X2))
    FMTp, _ = fmt(x, fmin, fmax, 2 * N)
    FMTp = FMTp / np.max(FMTp.real)
    Diff = FMTp - FMT[N:3 * N]
    assert np.abs(Diff).max() <= 1e-4
