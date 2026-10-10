# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the reassigned Gabor spectrogram.

Direct transcription of ``tftb-0.3/tests/tfrrgabt.m`` (F. Auger, 1994-1995).
Two lengths (N=128 even, N=127 odd). Properties checked on the reassigned
output ``rtfr``: realness, energy conservation (``sum(mean(rtfr)) ==
norm(sig)^2``), non-negativity, time localization (impulse stays in one
column), and frequency localization (pure tone concentrates in one bin).
"""
import numpy as np
import pytest

from tftb.generators import fmconst, fmlin, noisecg
from tftb.processing.reassigned import reassigned_gabor

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


@pytest.mark.parametrize("N", [128, 127])
def test_rgab_real(N):
    sig = noisecg(N)
    tfr, rtfr, hat = reassigned_gabor(sig)
    assert np.abs(np.imag(rtfr)).max() < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_rgab_energy(N):
    sig = fmlin(N)[0]
    tfr, rtfr, hat = reassigned_gabor(sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(rtfr, axis=0))
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("N", [128, 127])
def test_rgab_non_negative(N):
    sig = fmlin(N)[0]
    tfr, rtfr, hat = reassigned_gabor(sig)
    assert not (rtfr < 0).any()


@pytest.mark.parametrize("N, t0", [(128, 30), (127, 31)])
def test_rgab_time_localization(N, t0):
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr, rtfr, hat = reassigned_gabor(sig)
    nz = np.argwhere(np.abs(rtfr) > EPS)
    # every non-zero entry sits in column t0 (1-based) and rows span 1..N
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1


@pytest.mark.parametrize("N", [128, 127])
def test_rgab_freq_localization(N):
    f0 = 10
    sig = fmconst(N + 6, f0 / N)[0]
    if N % 2 == 0:
        t_matlab = N // 2 + 2      # MATLAB: N/2+2 (1-based)
        nh = N + 1
    else:
        t_matlab = int(round(N / 2)) + 2  # MATLAB: round(N/2)+2 (1-based)
        nh = N
    tfr, rtfr, hat = reassigned_gabor(sig, time_samples=np.array([t_matlab - 1]),
                                      n_fbins=N, window_length=nh)
    # MATLAB: `any(find(rtfr>max(rtfr)/N)~=f0+1)` (vector) errors unless every
    # bin exceeding max/N is bin f0+1 (1-based) — i.e. the peak bin is f0.
    peak = np.nonzero(rtfr > rtfr.max() / N)[0]
    assert peak.size > 0
    assert (peak == f0).all()
    assert abs(np.mean(rtfr) - 1.0) < EPS
