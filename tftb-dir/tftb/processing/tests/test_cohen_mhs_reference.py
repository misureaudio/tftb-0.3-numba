# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference test for MargenauHillSpectrogramDistribution.

Direct transcription of ``tftb-0.3/tests/tfrmhst.m`` (F. Auger, 1994-96).
Checks the Margenau-Hill-Spectrogram distribution identities for N=128 (even)
and N=123 (odd): translation covariance, realness, energy conservation
(custom rectangular time window / ones frequency window), and time
localization (custom window).
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, noisecg
from tftb.processing.cohen import MargenauHillSpectrogramDistribution

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _mhs(x, **kw):
    return MargenauHillSpectrogramDistribution(x, **kw).run()[0]


# Custom windows from the reference test: g=[1], h=ones(len).
def _custom_windows(N):
    g = np.array([1.0])
    # MATLAB: h=ones(N-1,1) for N=128 (even); h=ones(N,1) for N=123 (odd).
    hlen = N - 1 if N % 2 == 0 else N
    h = np.ones(hlen)
    return g, h


@pytest.mark.parametrize("N", [128, 123])
def test_mhs_translation_covariance(N):
    # tfrmhs test 1 (N=128) / test 5 (N=123)
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _mhs(sig1)
    tfr2 = _mhs(sig2)
    tc = tfr1.shape[1]
    s = t2 - t1
    # MATLAB: tfr1 - tfr2(:, modulo((1:tc)-t1+t2, tc));  0-based: roll(tfr2, -s)
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < EPS


@pytest.mark.parametrize("N", [128, 123])
def test_mhs_real_auto(N):
    # tfrmhs test 2 (N=128) / test 6 (N=123)
    tfr = _mhs(noisecg(N))
    assert np.abs(np.imag(tfr)).max() < EPS


@pytest.mark.parametrize("N", [128, 123])
def test_mhs_energy_custom_window(N):
    # tfrmhs test 3 (N=128) / test 7 (N=123)
    sig = noisecg(N)
    g, h = _custom_windows(N)
    tfr = _mhs(sig, twindow=g, fwindow=h)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))  # MATLAB: sum(mean(tfr)) (column mean)
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("N", [128, 123])
def test_mhs_time_localization_custom_window(N):
    # tfrmhs test 4 (N=128) / test 8 (N=123)
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    g, h = _custom_windows(N)
    tfr = _mhs(sig, twindow=g, fwindow=h)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    # all rows, single column (t0-1) 0-based
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1
