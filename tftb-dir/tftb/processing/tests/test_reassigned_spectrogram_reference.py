# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the reassigned spectrogram.

Direct transcription of ``tftb-0.3/tests/tfrrspt.m`` (F. Auger 1995,
O. Lemoine 1996). Two lengths (N=128 even, N=123 odd). Properties checked on
the reassigned output ``rtfr``: realness, energy conservation
(``mean(rtfr) == norm(sig)^2``), non-negativity, time localization (impulse
stays in one column), and frequency localization (pure tone concentrates in
one bin).
"""
import numpy as np
import pytest

from tftb.generators import fmconst, fmlin, noisecg
from tftb.processing.reassigned import spectrogram
from tftb.processing.utils import tftb_window

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


@pytest.mark.parametrize("N", [128, 123])
def test_rsp_real(N):
    sig = noisecg(N)
    tfr, rtfr, hat = spectrogram(sig)
    assert np.abs(np.imag(rtfr)).max() < EPS


@pytest.mark.parametrize("N", [128, 123])
def test_rsp_energy(N):
    sig = fmlin(N)[0]
    tfr, rtfr, hat = spectrogram(sig)
    Es = np.vdot(sig, sig).real
    # MATLAB sum(mean(rtfr)): mean is column-wise (dim=1), then summed.
    Etfr = np.sum(np.mean(rtfr, axis=0))
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("N", [128, 123])
def test_rsp_non_negative(N):
    sig = fmlin(N)[0]
    tfr, rtfr, hat = spectrogram(sig)
    assert not (rtfr < 0).any()


@pytest.mark.parametrize("N, t0", [(128, 30), (123, 30)])
def test_rsp_time_localization(N, t0):
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr, rtfr, hat = spectrogram(sig)
    nz = np.argwhere(np.abs(rtfr) > EPS)
    # every non-zero entry sits in column t0 (1-based) and the rows span 1..N
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert np.array_equal(np.sort(nz[:, 0]), np.arange(N))


@pytest.mark.parametrize("N", [128, 123])
def test_rsp_freq_localization(N):
    f0 = 30
    sig = fmconst(N + 6, f0 / N)[0]
    if N % 2 == 0:
        t0_1based = N // 2 + 2          # MATLAB: N/2+2
        wlen = N + 1
    else:
        t0_1based = int(round(N / 2)) + 2  # MATLAB: round(N/2)+2
        wlen = N
    tfr, rtfr, hat = spectrogram(
        sig, time_samples=np.array([t0_1based - 1]), n_fbins=N,
        window=tftb_window(wlen, 'rect'))
    # MATLAB: any(find(rtfr>max(rtfr)/N)~=f0+1) (vector) errors unless every
    # row exceeding max/N is row f0+1 (1-based) — i.e. the peak row is f0.
    peak = np.nonzero(rtfr.ravel() > rtfr.max() / N)[0]
    assert peak.size > 0
    assert (peak == f0).all()
    assert abs(np.mean(rtfr) - 1.0) < EPS
