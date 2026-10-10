# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference test for BornJordanDistribution.

Direct transcription of ``tftb-0.3/tests/tfrbjt.m`` (F. Auger, 1994-96).
Checks the Born-Jordan distribution identities:
translation covariance, realness (auto), energy conservation, time marginal,
and the frequency marginal under a specific rectangular-time / ones-frequency
window configuration.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, noisecg
from tftb.processing.cohen import BornJordanDistribution

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _bj(x, **kw):
    return BornJordanDistribution(x, **kw).run()[0]


def test_bj_translation_covariance():
    # tfrbj test 1
    N = 128
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _bj(sig1)
    tfr2 = _bj(sig2)
    tc = tfr1.shape[1]
    s = t2 - t1
    # MATLAB: tfr1 - tfr2(:, modulo((1:tc)-t1+t2, tc));  0-based: roll(tfr2, -s)
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < EPS


def test_bj_real_auto():
    # tfrbj test 2
    N = 128
    sig = noisecg(N)
    tfr = _bj(sig)
    assert np.abs(np.imag(tfr)).max() < EPS


def test_bj_energy():
    # tfrbj test 3
    N = 128
    sig = noisecg(N)
    tfr = _bj(sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))  # MATLAB: sum(mean(tfr)) (column mean)
    assert abs(Es - Etfr) < EPS


def test_bj_time_marginal():
    # tfrbj test 4
    N = 128
    sig = noisecg(N)
    tfr = _bj(sig)
    ip1 = np.abs(sig) ** 2
    ip2 = np.mean(tfr, axis=0)  # MATLAB: mean(tfr)' == row of column means
    assert np.abs(ip1 - ip2).max() < EPS


def test_bj_freq_marginal_custom_window():
    # tfrbj test 5 : tfrbj(sig, 1:N, N, [1], ones(N-1,1))
    N = 128
    sig = noisecg(N)
    g = np.array([1.0])          # MATLAB g = [1] (rectangular time window, len 1)
    h = np.ones(N - 1)           # MATLAB h = ones(N-1,1) (len 127, odd)
    tfr = _bj(sig, twindow=g, fwindow=h)
    FFT = np.fft.fft(sig)
    psd1 = np.abs(FFT[1: N // 2]) ** 2 / (2 * N)
    psd2 = np.mean(tfr[0::2, :], axis=1)  # MATLAB: mean(tfr(1:2:N,:)')
    assert np.abs(psd1 - psd2[1: N // 2]).max() < EPS
