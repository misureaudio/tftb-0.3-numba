# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference test for ZaoAtlasMarksDistribution.

Direct transcription of ``tftb-0.3/tests/tfrzamt.m`` (F. Auger, 1994-96).
Checks the Zao-Atlas-Marks distribution identities for N=128 and N=127:
translation covariance, realness (auto), energy conservation, time-support
conservation (wide-sense), and time localization.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, fmlin, noisecg
from tftb.processing.cohen import ZaoAtlasMarksDistribution

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _zam(x, **kw):
    return ZaoAtlasMarksDistribution(x, **kw).run()[0]


@pytest.mark.parametrize("N", [128, 127])
def test_zam_translation_covariance(N):
    # tfrzam test 1 (N=128) / test 6 (N=127)
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _zam(sig1)
    tfr2 = _zam(sig2)
    tc = tfr1.shape[1]
    s = t2 - t1
    # MATLAB: tfr1 - tfr2(:, modulo((1:tc)-t1+t2, tc));  0-based: roll(tfr2, -s)
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_zam_real_auto(N):
    # tfrzam test 2 (N=128) / test 7 (N=127)
    tfr = _zam(noisecg(N))
    assert np.abs(np.imag(tfr)).max() < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_zam_energy(N):
    # tfrzam test 3 (N=128) / test 8 (N=127)
    sig = noisecg(N)
    tfr = _zam(sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))  # MATLAB: sum(mean(tfr)) (column mean)
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_zam_time_support(N):
    # tfrzam test 4 (N=128) / test 9 (N=127)
    # sig = [zeros(N/4); fmlin(N/2); zeros(N/4)]  (round() for odd N).
    # For N=127 this rounds to a 128-sample signal (32+64+32), matching MATLAB.
    q = N // 4 if N % 2 == 0 else int(round(N / 4))
    half = N // 2 if N % 2 == 0 else int(round(N / 2))
    sig = np.r_[np.zeros(q), fmlin(half)[0], np.zeros(q)]
    tfr = _zam(sig)
    # MATLAB: tfr(:,1:N/4-1) and tfr(:,(3*N/4+1):N) even /
    #         tfr(:,1:round(N/4)-1) and tfr(:,(round(3N/4)+2):N) odd.
    # Both reduce to 0-based columns [0, q-2] and [3q, end].
    left = np.abs(tfr[:, :q - 1])
    right = np.abs(tfr[:, 3 * q:])
    assert not (left > 1e-7).any(), f"time support left (N={N})"
    assert not (right > 1e-7).any(), f"time support right (N={N})"


@pytest.mark.parametrize("N", [128, 127])
def test_zam_time_localization(N):
    # tfrzam test 5 (N=128) / test 10 (N=127)
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr = _zam(sig)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    # all rows, single column (t0-1) 0-based
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1
