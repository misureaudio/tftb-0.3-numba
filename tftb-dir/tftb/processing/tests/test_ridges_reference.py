# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for ridge extraction.

Direct transcription of ``tftb-0.3/tests/ridgest.m`` (O. Lemoine, 1996).
Two lengths (N=128 even, N=117 odd), both using the 2-axis ``'rsp'`` method
(pointsf = indices/Nf). Properties checked on the stationary points of the
reassignment: time localization (impulse -> all N rows in one column,
pointsf = (1:N)/N) and frequency localization (pure tone -> a contiguous time
span at the constant IF, pointsf = (f0+1)/N).
"""
import numpy as np
import pytest

from tftb.generators import fmconst
from tftb.processing.reassigned import spectrogram as reassigned_spectrogram
from tftb.processing.postprocessing import ridges

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)

# (N, t0, f0, fmconst_freq, (lo, hi) 1-based ptt slice, ptt slice value range)
_CASES = [
    (128, 60, 30, 30 / 128.0, (3, 106), (13, 116)),
    (117, 53, 31, 31 / 117.0, (3, 95), (11, 103)),
]


@pytest.mark.parametrize("N, t0, f0, ffreq, sl, ptt_range", _CASES)
def test_ridges_time_localization(N, t0, f0, ffreq, sl, ptt_range):
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr, rtfr, hat = reassigned_spectrogram(sig)
    ptt, ptf = ridges(tfr, hat, np.arange(1, N + 1), method='rsp')
    # every point sits in column t0 (1-based) and the rows span 1..N.
    assert len(ptt) == N
    assert (ptt == t0).all()
    assert np.allclose(ptf, np.arange(1, N + 1) / N, atol=EPS)


@pytest.mark.parametrize("N, t0, f0, ffreq, sl, ptt_range", _CASES)
def test_ridges_freq_localization(N, t0, f0, ffreq, sl, ptt_range):
    sig = fmconst(N, ffreq)[0]
    tfr, rtfr, hat = reassigned_spectrogram(sig)
    ptt, ptf = ridges(tfr, hat, np.arange(1, N + 1), method='rsp')
    lo, hi = sl  # 1-based inclusive slice (MATLAB ptt(lo:hi))
    a, b = lo - 1, hi  # -> 0-based [a:b]
    ptt_range_lo, ptt_range_hi = ptt_range
    # the interior time span is the contiguous range 13:116 (or 11:103).
    assert np.allclose(ptt[a:b], np.arange(ptt_range_lo, ptt_range_hi + 1), atol=EPS)
    # and the interior frequency is the constant IF (f0+1)/N (1-based bin).
    assert np.all(ptf[a:b] - (f0 + 1) / N <= EPS)
