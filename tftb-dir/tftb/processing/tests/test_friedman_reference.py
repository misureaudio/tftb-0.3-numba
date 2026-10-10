# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the Friedman instantaneous-frequency density.

Direct transcription of ``tftb-0.3/tests/friedmat.m`` (O. Lemoine, 1996).
Two lengths (N=128 even, N=111 odd). Properties checked on the TIFD
(``friedman_density``): unit integral, realness, non-negativity, time
localization (impulse stays in one column, rows span 1..N), and frequency
localization (pure tone concentrates in one bin).

Note: MATLAB ``friedman`` ignores its ``method`` argument for the computation
(only the display path uses it), so ``friedman_density(tfr, hat)`` is the
faithful port.
"""
import numpy as np
import pytest

from tftb.generators import fmconst, noisecg
from tftb.processing.reassigned import (
    pseudo_wigner_ville as reassigned_pwv,
    pseudo_page as reassigned_ppag,
    spectrogram as reassigned_spectrogram,
)
from tftb.processing.postprocessing import friedman_density
from tftb.processing.utils import tftb_window

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)

# (N, t0, f0, fmconst_freq, rect window length, freq-loc time sample 1-based)
_CASES = [
    (128, 60, 30, 30 / 128.0, 129, 128 // 2 + 2),
    (111, 51, 31, 31 / 111.0, 111, int(round(111 / 2.0)) + 2),
]


@pytest.mark.parametrize("N, t0, f0, ffreq, wlen, tloc", _CASES)
def test_friedman_unit_integral(N, t0, f0, ffreq, wlen, tloc):
    sig = noisecg(N)
    tfr, rtfr, hat = reassigned_pwv(sig)
    tifd = friedman_density(tfr, hat)
    assert abs(np.sum(tifd) - 1.0) < EPS


@pytest.mark.parametrize("N, t0, f0, ffreq, wlen, tloc", _CASES)
def test_friedman_real(N, t0, f0, ffreq, wlen, tloc):
    sig = noisecg(N)
    tfr, rtfr, hat = reassigned_pwv(sig)
    tifd = friedman_density(tfr, hat)
    assert np.abs(np.imag(tifd)).max() < EPS


@pytest.mark.parametrize("N, t0, f0, ffreq, wlen, tloc", _CASES)
def test_friedman_non_negative(N, t0, f0, ffreq, wlen, tloc):
    sig = noisecg(N)
    tfr, rtfr, hat = reassigned_pwv(sig)
    tifd = friedman_density(tfr, hat)
    assert not (tifd < 0).any()


@pytest.mark.parametrize("N, t0, f0, ffreq, wlen, tloc", _CASES)
def test_friedman_time_localization(N, t0, f0, ffreq, wlen, tloc):
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr, rtfr, hat = reassigned_ppag(sig)
    tifd = friedman_density(tfr, hat)
    nz = np.argwhere(tifd != 0.0)
    # every non-zero entry sits in column t0 (1-based) and the rows span 1..N
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert np.array_equal(np.sort(nz[:, 0]), np.arange(N))


@pytest.mark.parametrize("N, t0, f0, ffreq, wlen, tloc", _CASES)
def test_friedman_freq_localization(N, t0, f0, ffreq, wlen, tloc):
    sig = fmconst(N + 6, ffreq)[0]
    tfr, rtfr, hat = reassigned_spectrogram(
        sig, time_samples=np.array([tloc - 1]), n_fbins=N,
        window=tftb_window(wlen, 'rect'))
    tifd = friedman_density(tfr, hat)
    # MATLAB: any(find(tifd>max(tifd)/N)~=f0+1) — the bins exceeding max/N are
    # all bin f0+1 (1-based) = f0 (0-based).
    peak = np.nonzero(tifd.ravel() > tifd.max() / N)[0]
    assert peak.size > 0
    assert (peak == f0).all()
