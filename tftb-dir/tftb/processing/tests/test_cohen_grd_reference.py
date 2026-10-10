# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference test for GeneralizedRectangularDistribution.

Direct transcription of ``tftb-0.3/tests/tfrgrdt.m`` (F. Auger, 1994-96).
Checks the generalized-rectangular distribution identities: translation
covariance, realness (auto, N=128 and N=127), energy conservation, time and
frequency marginals under a custom window, and time localization.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, noisecg
from tftb.processing.cohen import GeneralizedRectangularDistribution

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _grd(x, **kw):
    return GeneralizedRectangularDistribution(x, **kw).run()[0]


def test_grd_translation_covariance():
    # tfrgrd test 1
    N = 128
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _grd(sig1)
    tfr2 = _grd(sig2)
    tc = tfr1.shape[1]
    s = t2 - t1
    # MATLAB: tfr1 - tfr2(:, modulo((1:tc)-t1+t2, tc));  0-based: roll(tfr2, -s)
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_grd_real_auto(N):
    # tfrgrd test 2 (N=128) and test 7 (N=127)
    sig = noisecg(N)
    tfr = _grd(sig)
    assert np.abs(np.imag(tfr)).max() < EPS


def test_grd_energy():
    # tfrgrd test 3
    N = 128
    sig = noisecg(N)
    tfr = _grd(sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))  # MATLAB: sum(mean(tfr)) (column mean)
    assert abs(Es - Etfr) < EPS


def test_grd_time_marginal_custom_window():
    # tfrgrd test 4 : tfrgrd(sig, 1:N, N, [1], ones(N-1,1))
    N = 128
    sig = noisecg(N)
    g = np.array([1.0])          # MATLAB g = [1]
    h = np.ones(N - 1)           # MATLAB h = ones(N-1,1)
    tfr = _grd(sig, twindow=g, fwindow=h)
    ip1 = np.abs(sig) ** 2
    ip2 = np.mean(tfr, axis=0)  # MATLAB: mean(tfr)'
    assert np.abs(ip1 - ip2).max() < EPS


def test_grd_freq_marginal_custom_window():
    # tfrgrd test 5 : tfrgrd(sig, 1:N, N, [1], ones(N-1,1))
    N = 128
    sig = noisecg(N)
    g = np.array([1.0])          # MATLAB g = [1]
    h = np.ones(N - 1)           # MATLAB h = ones(N-1,1)
    tfr = _grd(sig, twindow=g, fwindow=h)
    FFT = np.fft.fft(sig)
    psd1 = np.abs(FFT[1: N // 2]) ** 2 / (2 * N)
    psd2 = np.mean(tfr[0::2, :], axis=1)  # MATLAB: mean(tfr(1:2:N,:)')
    assert np.abs(psd1 - psd2[1: N // 2]).max() < EPS


def test_grd_time_localization():
    # tfrgrd test 6
    N = 128
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr = _grd(sig)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    # all nonzero entries in the single column (t0-1) 0-based, all rows
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1
