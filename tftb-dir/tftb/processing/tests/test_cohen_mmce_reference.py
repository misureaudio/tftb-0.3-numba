# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the minimum mean cross-entropy distribution.

Direct transcription of ``tftb-0.3/tests/tfrmmcet.m`` (F. Auger, 1995).
Two window sets: N=128 with three Hamming windows (11/15/19) and N=121 with
gauss/kaiser/parzen windows (13/17/15). Properties checked for each:
translation covariance, realness, and non-negativity.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, noisecg
from tftb.processing.cohen import MinimumMeanCrossEntropyDistribution
from tftb.processing.utils import tftb_window

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _hamming_set():
    """MATLAB: h=zeros(19,3); h(10+(-5:5),1)=tftb_window(11); ... (N=128)."""
    h = np.zeros((19, 3))
    h[4:15, 0] = tftb_window(11)      # 10+(-5:5) 1-based -> 4..14 0-based
    h[2:17, 1] = tftb_window(15)      # 10+(-7:7)
    h[0:20, 2] = tftb_window(19)      # 10+(-9:9)
    return h


def _odd_set():
    """MATLAB: h=zeros(19,3); gauss(13)/kaiser(17)/parzen(15) (N=121)."""
    h = np.zeros((19, 3))
    h[3:16, 0] = tftb_window(13, 'gauss')    # 10+(-6:6)
    h[1:18, 1] = tftb_window(17, 'kaiser')   # 10+(-8:8)
    h[2:17, 2] = tftb_window(15, 'parzen')   # 10+(-7:7)
    return h


@pytest.mark.parametrize("N, hset", [(128, _hamming_set), (121, _odd_set)])
def test_mmce_translation_covariance(N, hset):
    h = hset()
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = MinimumMeanCrossEntropyDistribution(sig1, h).run()[0]
    tfr2 = MinimumMeanCrossEntropyDistribution(sig2, h).run()[0]
    cov = tfr1 - np.roll(tfr2, -(t2 - t1), axis=1)
    assert np.abs(cov).max() < EPS


@pytest.mark.parametrize("N, hset", [(128, _hamming_set), (121, _odd_set)])
def test_mmce_real(N, hset):
    np.random.seed(0)  # deterministic (MATLAB harness seeds rng(0) per test)
    h = hset()
    tfr = MinimumMeanCrossEntropyDistribution(noisecg(N), h).run()[0]
    assert np.abs(np.imag(tfr)).max() < EPS


@pytest.mark.parametrize("N, hset", [(128, _hamming_set), (121, _odd_set)])
def test_mmce_non_negative(N, hset):
    np.random.seed(0)
    h = hset()
    tfr = MinimumMeanCrossEntropyDistribution(noisecg(N), h).run()[0]
    assert not (tfr < 0).any()


def test_mmce_rejects_single_window_column():
    with pytest.raises(ValueError):
        MinimumMeanCrossEntropyDistribution(np.ones(32), np.ones((11, 1)))


def test_mmce_rejects_even_row_count():
    with pytest.raises(ValueError):
        MinimumMeanCrossEntropyDistribution(np.ones(32), np.ones((12, 2)))
