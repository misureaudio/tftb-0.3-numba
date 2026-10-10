# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Transcribed reference test for RihaczekDistribution.

Direct transcription of ``tftb-0.3/tests/tfrrit.m`` (O. Lemoine, 1996).
Checks the Cohen-class identities that hold for the Rihaczek distribution:
translation covariance, energy conservation, time/frequency marginals,
unitarity, wide-sense time-support conservation, and time localization —
for both even (N=128) and odd (N=127) signal lengths.
"""
import numpy as np
import pytest

from tftb.generators import amgauss, amexpo2s, fmconst, fmlin, noisecg
from tftb.processing.cohen import RihaczekDistribution

EPS = np.sqrt(np.finfo(float).eps)  # ~1.49e-8, matches MATLAB sqrt(eps)


def _ri(x):
    return RihaczekDistribution(x).run()[0]


@pytest.mark.parametrize("N", [128, 127])
def test_rihaczek_reference(N):
    # ---- 1. Covariance by translation in time ----
    t1, t2, f = (60, 70, 0.3) if N == 128 else (61, 70, 0.3)
    sig1 = amgauss(N, t1) * fmconst(N, f, t1)[0]
    sig2 = amgauss(N, t2) * fmconst(N, f, t2)[0]
    tfr1 = _ri(sig1)
    tfr2 = _ri(sig2)
    tc = tfr1.shape[1]
    s = t2 - t1
    # Rihaczek is translation-covariant: R_sig2(t,f)=R_sig1(t-s,f)
    # => tfr1[:,k] = tfr2[:,k+s]  =>  tfr1 - roll(tfr2, -s) == 0
    cov = tfr1 - np.roll(tfr2, -s, axis=1)
    tol = 1e-7 if N == 127 else EPS
    assert np.abs(cov).max() < tol, f"translation covariance (N={N})"

    # ---- 2. Energy conservation ----
    # MATLAB: Etfr = sum(mean(tfr)) where mean is down columns (axis=0).
    # Rihaczek: tfr.sum() = N*||x||^2  =>  tfr.sum()/N = ||x||^2 = Es.
    sig = noisecg(N)
    tfr = _ri(sig)
    Es = np.vdot(sig, sig).real
    Etfr = np.sum(np.mean(tfr, axis=0))
    assert abs(Es - Etfr) < 1e-6 * max(1.0, Es), f"energy (N={N})"

    # ---- 3. Time marginal ----
    sig = noisecg(N)
    tfr = _ri(sig)
    ip1 = np.abs(sig) ** 2
    ip2 = np.mean(tfr, axis=0)
    assert np.abs(ip1 - ip2).max() < 1e-6 * max(1.0, ip1.max()), f"time marginal (N={N})"

    # ---- 4. Frequency marginal ----
    sig = noisecg(N)
    tfr = _ri(sig)
    FFT = np.fft.fft(sig)
    hi = N // 2
    psd1 = np.abs(FFT[1:hi]) ** 2 / N
    psd2 = np.real(np.mean(tfr[1:hi, :], axis=1))
    assert np.abs(psd1 - psd2).max() < 1e-6 * max(1.0, psd1.max()), f"freq marginal (N={N})"

    # ---- 5. Unitarity ----
    x1 = amgauss(N) * fmlin(N)[0]
    x2 = amexpo2s(N)
    tfr1 = _ri(x1)
    tfr2 = _ri(x2)
    cor1 = abs(np.vdot(x1, x2)) ** 2
    cor2 = np.real(np.sum(tfr1 * np.conj(tfr2)) / N)
    assert abs(cor1 - cor2) < 1e-6 * max(1.0, cor1), f"unitarity (N={N})"

    # ---- 6. Conservation of the time support (wide-sense) ----
    q = N // 4 if N % 2 == 0 else int(round(N / 4))
    half = N // 2 if N % 2 == 0 else int(round(N / 2))
    sig = np.r_[np.zeros(q), fmlin(half)[0], np.zeros(N - q - half)]
    tfr = _ri(sig)
    # MATLAB: tfr(:,1:q-1)  and  tfr(:,(3*q+1):N) even / (round(3N/4)+2):N odd
    # both start at 0-based column 3*q (first sample outside the support)
    right_start = 3 * q
    left = np.abs(tfr[:, :q - 1])
    right = np.abs(tfr[:, right_start:])
    assert not (left > 1e-7).any(), f"time support left (N={N})"
    assert not (right > 1e-7).any(), f"time support right (N={N})"

    # ---- 7. Time localization ----
    t0 = 30
    sig = (np.arange(1, N + 1) == t0).astype(float)
    tfr = _ri(sig)
    nz = np.argwhere(np.abs(tfr) > 1e-9)
    # all nonzero entries in the single column (t0-1) 0-based
    assert nz[:, 1].min() == nz[:, 1].max() == (t0 - 1), f"time localization (N={N})"
    assert nz[:, 0].min() == 0 and nz[:, 0].max() == N - 1, f"time localization rows (N={N})"
