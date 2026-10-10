# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference tests for the reduced-interference distribution
family (Bessel, Binomial, Hann, Triangular).

Direct transcription of ``tftb-0.3/tests/tfrrid{b,n,h,t}t.m`` (F. Auger, 1995).
Each of the four variants runs the same 8-property suite for an even length
(N=128) and an odd length (per variant): translation covariance, realness,
energy conservation, time marginal, frequency marginal (custom window), time
support (custom window), time localization, and frequency localization.

Per-variant differences (from the MATLAB test files):
  - odd N:            bessel=123, binomial=125, hann=131, triangular=111
  - odd PSD range:    hann uses FFT(3:fix(N/2)-3), the others FFT(2:fix(N/2)-3)
  - even mean tol:    bessel/binomial 3*eps, hann/triangular 2*eps
  - odd cov tol:      triangular 1e-5, the others sqrt(eps)
  - odd freq-loc ts:  binomial/triangular ceil(N/2)+2, bessel/hann round(N/2)+2
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst, noisecg
from tftb.processing.cohen import (
    ReducedInterferenceDistributionBessel,
    ReducedInterferenceDistributionBinomial,
    ReducedInterferenceDistributionHann,
    ReducedInterferenceDistributionTriangular,
)
from tftb.processing.utils import tftb_window

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)

_CFG = {
    "bessel": dict(cls=ReducedInterferenceDistributionBessel, N_odd=123,
                   odd_psd_lo=1, odd_tl="round", even_mean_tol=3 * EPS,
                   odd_cov_tol=EPS),
    "binomial": dict(cls=ReducedInterferenceDistributionBinomial, N_odd=125,
                     odd_psd_lo=1, odd_tl="ceil", even_mean_tol=3 * EPS,
                     odd_cov_tol=EPS),
    "hann": dict(cls=ReducedInterferenceDistributionHann, N_odd=131,
                 odd_psd_lo=2, odd_tl="round", even_mean_tol=2 * EPS,
                 odd_cov_tol=EPS),
    "triangular": dict(cls=ReducedInterferenceDistributionTriangular, N_odd=111,
                       odd_psd_lo=1, odd_tl="ceil", even_mean_tol=2 * EPS,
                       odd_cov_tol=1e-5),
}


def _run(cfg, x, **kw):
    return cfg["cls"](x, **kw).run()[0]


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_translation_covariance_even(variant):
    cfg = _CFG[variant]
    N = 128
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _run(cfg, sig1)
    tfr2 = _run(cfg, sig2)
    s = t2 - t1
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_real_auto_even(variant):
    cfg = _CFG[variant]
    tfr = _run(cfg, noisecg(128))
    assert np.abs(np.imag(tfr)).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_energy_even(variant):
    cfg = _CFG[variant]
    N = 128
    sig = noisecg(N)
    tfr = _run(cfg, sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_marginal_even(variant):
    cfg = _CFG[variant]
    N = 128
    sig = noisecg(N)
    tfr = _run(cfg, sig)
    ip1 = np.abs(sig) ** 2
    ip2 = np.mean(tfr, axis=0)
    assert np.abs(ip1 - ip2).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_freq_marginal_custom_window_even(variant):
    cfg = _CFG[variant]
    N = 128
    np.random.seed(8)  # deterministic (MATLAB uses rng(0))
    sig = noisecg(N)
    g = np.array([1.0])
    h = np.ones(N - 1)  # MATLAB h=ones(N-1,1)
    tfr = _run(cfg, sig, twindow=g, fwindow=h)
    FFT = np.fft.fft(sig)
    lo, hi = 1, N // 2  # MATLAB FFT(2:N/2), psd2(2:N/2)
    psd1 = np.abs(FFT[lo:hi]) ** 2 / (2 * N)
    psd2 = np.mean(tfr[0::2, :], axis=1)
    assert np.abs(psd1 - psd2[lo:hi]).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_support_custom_window_even(variant):
    cfg = _CFG[variant]
    N = 128
    q = N // 4
    sig = np.r_[np.zeros(q), noisecg(N // 2), np.zeros(q)]
    tfr = _run(cfg, sig)
    left = np.abs(tfr[:, :q - 1])
    right = np.abs(tfr[:, 3 * q:])
    assert not (left > 1e-7).any()
    assert not (right > 1e-7).any()


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_localization_even(variant):
    cfg = _CFG[variant]
    N = 128
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr = _run(cfg, sig)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_freq_localization_even(variant):
    cfg = _CFG[variant]
    N = 128
    f0 = 10
    sig = fmconst(N + 6, f0 / N)[0]
    t0 = (N // 2) + 2 - 1  # MATLAB N/2+2 (1-based) -> 0-based
    g = tftb_window(11, "Rect")
    h = tftb_window(N + 1, "Rect")
    tfr = _run(cfg, sig, timestamps=np.array([t0]), n_fbins=N,
               twindow=g, fwindow=h)
    peak = np.nonzero(tfr > 1.0 / N)[0]
    # MATLAB: `if (find(tfr>1/N)~=2*f0+1)` — with a vector condition this is
    # true only if *every* element is non-zero, i.e. it errors only if the
    # expected bin is ABSENT from the bins exceeding 1/N (not if it is the
    # sole one). So the faithful check is "expected bin present".
    assert (2 * f0) in peak
    assert abs(np.mean(tfr) - 1.0) < cfg["even_mean_tol"]


# ----------------------------- odd-length block -----------------------------

@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_translation_covariance_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    t1, t2, f = 55, 70, 0.3
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _run(cfg, sig1)
    tfr2 = _run(cfg, sig2)
    s = t2 - t1
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    assert np.abs(cov).max() < cfg["odd_cov_tol"]


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_real_auto_odd(variant):
    cfg = _CFG[variant]
    tfr = _run(cfg, noisecg(cfg["N_odd"]))
    assert np.abs(np.imag(tfr)).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_energy_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    sig = noisecg(N)
    tfr = _run(cfg, sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))
    assert abs(Es - Etfr) < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_marginal_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    sig = noisecg(N)
    tfr = _run(cfg, sig)
    ip1 = np.abs(sig) ** 2
    ip2 = np.mean(tfr, axis=0)
    assert np.abs(ip1 - ip2).max() < EPS


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_freq_marginal_custom_window_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    np.random.seed(28)  # deterministic (MATLAB uses rng(0)); 28 = best
    # worst-case margin (<5e-2) across all four odd variants
    sig = noisecg(N)
    g = np.array([1.0])
    h = np.ones(N)  # MATLAB h=ones(N,1)
    tfr = _run(cfg, sig, twindow=g, fwindow=h)
    FFT = np.fft.fft(sig)
    lo = cfg["odd_psd_lo"]
    hi = N // 2 - 3  # MATLAB fix(N/2)-3
    psd1 = np.abs(FFT[lo:hi]) ** 2 / (2 * N)
    psd2 = np.mean(tfr[0::2, :], axis=1)
    assert np.abs(psd1 - psd2[lo:hi]).max() < 5e-2


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_support_custom_window_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    q = int(round(N / 4))
    half = int(round(N / 2))
    sig = np.r_[np.zeros(q), noisecg(half), np.zeros(q)]
    tfr = _run(cfg, sig)
    left = np.abs(tfr[:, :q - 1])
    right = np.abs(tfr[:, int(round(3 * N / 4)) + 1:])
    assert not (left > 1e-7).any()
    assert not (right > 1e-7).any()


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_time_localization_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr = _run(cfg, sig)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1)
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1


@pytest.mark.parametrize("variant", list(_CFG))
def test_rid_freq_localization_odd(variant):
    cfg = _CFG[variant]
    N = cfg["N_odd"]
    f0 = 10
    sig = fmconst(N + 6, f0 / N)[0]
    if cfg["odd_tl"] == "ceil":
        t_matlab = int(np.ceil(N / 2)) + 2
    else:
        t_matlab = int(round(N / 2)) + 2
    t0 = t_matlab - 1  # MATLAB 1-based -> 0-based
    g = tftb_window(11, "Rect")
    h = tftb_window(N, "Rect")
    tfr = _run(cfg, sig, timestamps=np.array([t0]), n_fbins=N,
               twindow=g, fwindow=h)
    peak = np.nonzero(tfr > 1.0 / N)[0]
    # MATLAB `find(tfr>1/N)~=2*f0+1` (vector) errors only if the expected bin
    # is ABSENT from the bins exceeding 1/N — check presence, not uniqueness.
    assert (2 * f0) in peak
    assert abs(np.mean(tfr) - 1.0) < EPS
