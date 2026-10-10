# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Fast Mellin transform and its inverse.

Faithful port of the ``tftb-0.3`` Mellin-transform pair:

* ``fmt``  <- ``fmt.m``   (fast Fourier-Mellin transform)
* ``ifmt`` <- ``ifmt.m``  (inverse fast Mellin transform)

The Mellin transform maps a signal to the log-frequency (``beta``) domain via
a geometrically-sampled Fourier transform; it is the natural tool for
multiplicative (log-scale) structure.  The inverse recovers the signal and is
exact only when the transform ran from ``fmin`` up to ``0.5`` Hz and the
original signal is analytic (per the ``ifmt.m`` warning).

Conventions matched to MATLAB:

* ``hilbert`` -> ``scipy.signal.hilbert`` (verified identical to MATLAB's
  ``hilbert`` to ~7e-16 on an analytic cosine; no 2x scaling difference).
* ``geo_f`` is a **row** vector in MATLAB (colon -> row), so the DFT matrices
  ``t'*geo_f`` are outer products -> ``np.outer(t, geo_f)``.
* ``integ`` integrates a **complex** integrand here (unlike ``gaussn``'s
  real-only fallback), so a complex-capable trapezoid is used.
* 1-based MATLAB slices ``SP(indmin:indmax)`` with ``indmin = 1+round(...)``
  become ``SP[indmin-1:indmax]`` in 0-based Python.
"""
import numpy as np
from scipy.signal import hilbert


def _integ(y, x):
    """MATLAB ``integ.m`` trapezoid sum, complex-capable.

    ``integ(y, x) = sum((y[i]+y[i+1]) * (x[i+1]-x[i])/2)``.  Unlike the
    real-only fallback in ``wavelet.py``, this preserves a complex ``y``
    (used by ``ifmt`` on the complex inverse-Fourier matrix).
    """
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    dy = y[:-1] + y[1:]
    dx = (x[1:] - x[:-1]) / 2.0
    return dy @ dx


def _nextpow2(n):
    """Smallest power of two >= ``n`` (MATLAB ``2^nextpow2(n)``)."""
    if n <= 1:
        return 1
    return 1 << int(np.ceil(np.log2(n)))


def fmt(x, fmin, fmax, N=None):
    """Fast Fourier-Mellin transform of ``x``.

    Port of ``fmt.m``.  Geometrically samples ``N`` voices between ``fmin`` and
    ``fmax`` (both in ``(0, 0.5]``), ``fmax > fmin``), computes the analytic
    signal's Fourier coefficients on that grid, and folds them into the
    ``beta`` (log-frequency) Mellin domain.  Returns ``(mellin, beta)`` where
    ``mellin`` is the ``N``-point Mellin transform (normalized so its energy
    equals the in-band signal energy) and ``beta`` is the ``N``-point Mellin
    variable.

    Parameters
    ----------
    x : (N,) array_like
        Signal (real or complex; the real part is analyzed).
    fmin, fmax : float
        Lower/upper frequency bounds, in ``(0, 0.5]``, ``fmax > fmin``.
    N : int, optional
        Number of voices (must be even).  Defaults to the next power of two
        above the minimum needed for the given bandwidth.

    Returns
    -------
    (mellin, beta) : (complex (N,), float (N,))
    """
    x = np.asarray(x)
    Mt = x.shape[0]
    xrow = Mt  # MATLAB size(X) row count (column vector in the oracle)

    if N is None:
        N = _default_N(x, fmin, fmax)
    if N % 2 != 0:
        raise ValueError("N must be even")
    if not (fmax >= fmin):
        raise ValueError("FMAX must be greater or equal to FMIN")
    if not (0.0 < fmin <= 0.5):
        raise ValueError("FMIN must be > 0 and <= 0.5")
    if not (0.0 < fmax <= 0.5):
        raise ValueError("FMAX must be > 0 and <= 0.5")

    B = fmax - fmin
    R = B / ((fmin + fmax) / 2)

    # Geometric sampling of the analyzed spectrum.
    No2 = N // 2
    k = np.arange(1, No2 + 1)
    q = (fmax / fmin) ** (1.0 / (No2 - 1))
    t = (np.arange(1, Mt + 1) - (Mt + Mt % 2) / 2.0 - 1.0)
    geo_f = fmin * np.exp((k - 1) * np.log(q))
    tfmatx = np.exp(-2j * np.pi * np.outer(t, geo_f))

    Z = hilbert(x.real)
    # MATLAB  Z.'*tfmatx :  Z.'  is the NON-conjugate transpose (Z' would be
    # the conjugate-transpose).  So this is  Z @ tfmatx  with no conjugation.
    ZS_full = Z @ tfmatx             # (No2,)  == MATLAB  Z.'*tfmatx  (1 x No2)
    ZS = np.zeros(N, dtype=complex)
    ZS[:No2] = ZS_full               # MATLAB  ZS(No2+1:N) = zeros  (analytic: no -freqs)

    # Mellin transform computation.
    p = np.arange(N)
    L = np.log(fmin) / np.log(q)
    mellin = N * np.log(q) * np.fft.fftshift(np.fft.ifft(ZS)) \
        * np.exp(1j * 2 * np.pi * L * (p / N - 0.5))
    beta = (p / N - 0.5) / np.log(q)

    # Normalization (in-band energy).
    SP = np.fft.fft(hilbert(x.real))
    indmin = 1 + int(round(fmin * (xrow - 2)))
    indmax = 1 + int(round(fmax * (xrow - 2)))
    SPana = SP[indmin - 1:indmax]
    nu = np.arange(indmin, indmax + 1) / N
    SPp = SPana / nu
    Normsig = np.sqrt(np.vdot(SPp, SPana))
    mellin = mellin * Normsig / np.linalg.norm(mellin)

    return mellin, beta


def _default_N(x, fmin, fmax):
    """Minimum-voices power-of-two default (MATLAB ``Ndflt`` path)."""
    Mt = x.shape[0]
    B = fmax - fmin
    R = B / ((fmin + fmax) / 2)
    Nq = int(np.ceil((B * Mt * (1 + 2 / R)
                     * np.log((1 + R / 2) / (1 - R / 2))) / 2))
    Nmin = Nq - Nq % 2
    return _nextpow2(Nmin)


def ifmt(mellin, beta, M=None):
    """Inverse fast Mellin transform.

    Port of ``ifmt.m``.  Recovers the ``M``-point analytic signal from its
    ``N``-point Mellin transform.  Correct only when the transform was
    computed from ``fmin`` up to ``0.5`` Hz and the original signal is
    analytic (``ifmt.m`` warning).

    Parameters
    ----------
    mellin : (N,) complex
        Mellin transform from :func:`fmt`.
    beta : (N,) float
        Mellin variable from :func:`fmt`.
    M : int, optional
        Number of output time points (defaults to ``len(mellin)``).

    Returns
    -------
    x : (M,) complex
        Reconstructed (analytic) signal.
    """
    mellin = np.asarray(mellin)
    beta = np.asarray(beta)
    N = mellin.shape[0]
    if M is None:
        M = N

    No2 = (N + N % 2) // 2
    q = np.exp(1.0 / (N * (beta[1] - beta[0])))
    fmin = 0.5 / (q ** (No2 - 1))

    # Inverse Mellin transform computation.
    p = np.arange(N)
    L = np.log(fmin) / np.log(q)
    S = np.fft.fft(np.fft.fftshift(
        mellin * np.exp(-1j * 2 * np.pi * L * (p / N - 0.5))
        / (N * np.log(q))))
    S = S[:No2]

    # Inverse Fourier transform (trapezoid integral over the geometric grid).
    k = np.arange(1, No2 + 1)
    t = (np.arange(1, M + 1) - (M + M % 2) / 2.0 - 1.0)
    geo_f = fmin * np.exp((k - 1) * np.log(q))
    itfmatx = np.exp(2j * np.pi * np.outer(t, geo_f))
    Y = itfmatx * S                     # (M, No2)
    dx = (geo_f[1:] - geo_f[:-1]) / 2.0
    x = np.real((Y[:, :-1] + Y[:, 1:]) @ dx)   # row-wise integ(., geo_f)

    x = hilbert(x)

    # Normalization.
    SP = np.fft.fft(x)
    fmax = 0.5
    indmin = 1 + int(round(fmin * (M - 2)))
    indmax = 1 + int(round(fmax * (M - 2)))
    SPana = SP[indmin - 1:indmax]
    nu = np.arange(indmin, indmax + 1) / M
    SPp = SPana / nu
    Esm = np.vdot(SPp, SPana)
    x = x * np.linalg.norm(mellin) / np.sqrt(Esm)

    return x
