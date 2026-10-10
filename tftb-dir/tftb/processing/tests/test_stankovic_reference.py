# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""L3 property / naive-reference tests for the Stanković distribution.

There is no ``tftb-0.3/tests`` reference file for ``tfrrstan``, so this module
provides independent (L3) checks:
  * realness (the distribution is explicitly real-valued),
  * a fully independent naive reference (circular STFT placement + direct DFT,
    no ``np.fft``) — an identity check that does not share the fast path's FFT,
  * the closed-form identity for the default 3-point window ``G=[.25,.5,.25]``.

Value-level equivalence against MATLAB is covered by the golden cases
``stan_1`` / ``stan_2`` (L1, rtol 1e-10).
"""
import numpy as np
import pytest

from tftb.generators import amgauss, fmconst
from tftb.processing.linear import StankovicDistribution
from tftb.processing.postprocessing import ridges
from tftb.processing.utils import tftb_window


def _naive_stan(x, N, G, h):
    """Independent Stanković reference: circular STFT placement + direct DFT.

    Deliberately avoids ``np.fft`` (uses an explicit O(N²) DFT) so it shares no
    code with the fast path.
    """
    xrow = len(x)
    lh = (len(h) - 1) // 2
    lg = (len(G) - 1) // 2
    nmax = int(np.floor(N / 2.0 + 0.5)) - 1
    V = np.zeros((N, N))
    for c in range(N):
        ti = c
        tau = np.arange(-min([nmax, lh, ti]),
                        min([nmax, lh, xrow - 1 - ti]) + 1)
        nidx = (N + tau) % N
        seg = x[ti + tau] * np.conj(h[lh - tau])
        X = np.array([np.sum(seg * np.exp(-2j * np.pi * f * nidx / N))
                      for f in range(N)])
        kmax = min(N // 2 - 1, lg)
        for j in range(N):
            v = G[lg] * np.abs(X[j]) ** 2
            for k in range(1, kmax + 1):
                b = (j - k) % N
                a = (j + k) % N
                v += (G[lg - k] * X[b] * np.conj(X[a])
                      + G[lg + k] * X[a] * np.conj(X[b]))
            V[j, c] = np.real(v)
    return V


@pytest.mark.parametrize("N", [32, 33])
def test_stan_is_real(N):
    sig = amgauss(N, N // 2) * fmconst(N, 0.25, N // 2)[0]
    tfr = StankovicDistribution(sig).run()[0]
    assert np.abs(np.imag(tfr)).max() < 1e-12


@pytest.mark.parametrize("N", [32, 33])
def test_stan_matches_naive_direct_dft(N):
    """Fast path vs an independent direct-DFT reference (both G lengths)."""
    sig = amgauss(N, N // 2) * fmconst(N, 0.25, N // 2)[0]
    h = tftb_window(17, 'Hamming')
    for G in (np.array([0.25, 0.5, 0.25]), tftb_window(7, 'Hanning')):
        fast = StankovicDistribution(sig, fwindow=h, fsmooth=G).run()[0]
        slow = _naive_stan(sig, N, G, h)
        assert np.abs(fast - slow).max() < 1e-9


@pytest.mark.parametrize("N", [32, 33])
def test_stan_default_g_closed_form(N):
    """Default G=[.25,.5,.25]: V[j]=.5|X[j]|²+.5 Re(X[j-1]conj(X[j+1]))."""
    sig = amgauss(N, N // 2) * fmconst(N, 0.25, N // 2)[0]
    h = StankovicDistribution(sig).fwindow  # default odd-length Hamming
    tfr = StankovicDistribution(sig).run()[0]
    lh = (h.shape[0] - 1) // 2
    nmax = int(np.floor(N / 2.0 + 0.5)) - 1
    for c in range(N):
        ti = c
        tau = np.arange(-min([nmax, lh, ti]), min([nmax, lh, N - 1 - ti]) + 1)
        col = np.zeros(N, dtype=complex)
        col[(N + tau) % N] = sig[ti + tau] * np.conj(h[lh - tau])
        X = np.fft.fft(col)
        # G=[.25,.5,.25] (LG=1): V[j]=.5|X[j]|²+.5 Re(X[j-1]·conj(X[j+1]))
        v = 0.5 * np.abs(X) ** 2 + 0.5 * np.real(np.roll(X, 1) * np.roll(np.conj(X), -1))
        assert np.abs(tfr[:, c] - v).max() < 1e-9


def test_stan_rejects_even_smoothing_window():
    with pytest.raises(ValueError):
        StankovicDistribution(np.ones(32), fsmooth=np.ones(4))


# ---- Task 3.4: reassigned Stankovic (rtfr / hat) ----
# Value-level equivalence against MATLAB is covered by the golden cases
# stan_1 / stan_2 (L1, rtol 1e-10, now including rtfr and hat). These are
# independent closed-form checks on the reassigned output.

def test_stan_reassigned_energy_conserved():
    # reassignment only moves energy between bins -> the total is preserved.
    N = 128
    sig = amgauss(N, 40, 8) * fmconst(N, 0.3)[0]
    stan, ts, freqs, rtfr, hat = StankovicDistribution(sig).run()
    assert abs(np.sum(stan).real - np.sum(rtfr).real) < 1e-8


def test_stan_reassigned_ridges_recover_if_law():
    # a single constant-frequency atom -> the reassignment ridge is the flat
    # line at f0, located at the atom's time.
    N = 128
    t0, f0, spread = 40, 0.3, 8
    sig = amgauss(N, t0, spread) * fmconst(N, f0)[0]
    stan, ts, freqs, rtfr, hat = StankovicDistribution(sig).run()
    t = np.arange(1, N + 1)
    pt, pf = ridges(stan, hat, t, method='rsp')
    m = (pt >= t0 - 2 * spread) & (pt <= t0 + 2 * spread)
    assert m.sum() > 3, "too few ridge points near the atom"
    assert abs(np.median(pf[m]) - f0) < 0.02, (
        f"ridge freq median {np.median(pf[m]):.4f} != IF law {f0}")
    # the ridge sits at the atom's time (1-based).
    assert abs(np.median(pt[m]) - t0) < 2 * spread


def test_stan_hat_is_one_based():
    # above-threshold bins: hat = 1-based (jcol, icol); sub-threshold bins
    # record their own 1-based position (tfrrstan convention, not inf).
    N = 64
    sig = amgauss(N, 20, 6) * fmconst(N, 0.25)[0]
    stan, ts, freqs, rtfr, hat = StankovicDistribution(sig).run()
    finite = np.isfinite(hat)
    assert finite.any()
    # every finite hat entry has integer 1-based coordinates in range.
    hr = np.real(hat[finite]).astype(int)
    hi = np.imag(hat[finite]).astype(int)
    assert (hr >= 1).all() and (hr <= N).all()
    assert (hi >= 1).all() and (hi <= N).all()
    # sub-threshold bins (where stan is tiny) record their own position.
    tiny = np.abs(stan) < 1e-6
    if tiny.any():
        r, i = np.nonzero(tiny)
        assert np.allclose(np.real(hat[r, i]), r + 1)
        assert np.allclose(np.imag(hat[r, i]), i + 1)
