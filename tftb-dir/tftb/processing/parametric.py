# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Parametric / AR-based TFR utilities (Phase 4 Task 4.4).

Faithful port of the ``tftb-0.3`` parametric family.  Each sub-function is
ported and committed separately (Task 4.4a..e):

* 4.4a ``correlmx``   <- ``correlmx.m``   (correlation matrix, 4 algorithms)
* 4.4b ``kaytth``     <- ``kaytth.m``     (Kay's th / AR parameter estimator)
* 4.4c ``ifestar2``   <- ``ifestar2.m``   (AR(2) instantaneous frequency)
* 4.4d ``parafrep``   <- ``parafrep.m``   (parametric representation)
* 4.4e ``tfrparam``   <- ``tfrparam.m``   (sliding-window parametric TFR)
"""
import numpy as np


def correlmx(x, p, rxtype="fbhermitian"):
    """Correlation matrix of a signal (port of ``correlmx.m``).

    Builds the ``(p+1) x (p+1)`` correlation matrix of ``x`` up to lag ``p``.

    Parameters
    ----------
    x : (L,) array_like
        Signal (column vector).  ``L >= p+1``.
    p : int
        Last autocorrelation lag (``p >= 1``).
    rxtype : str, optional
        Algorithm: ``'hermitian'``, ``'fbhermitian'`` (default), ``'burg'``
        or ``'fbburg'``.  The ``'fb'`` variants additionally force exact
        Hermitian symmetry with ``Rx = 0.5*(Rx + conj(Rx.T))``.

    Returns
    -------
    Rx : (p+1, p+1) complex
        The correlation matrix.

    Notes
    -----
    * **Hermitian** form: ``Rx(i,j) = (1/(L-p)) * sum_s conj(x[s+p-i])
      x[s+p-j]`` over the ``L-p`` windows — accumulated as a sum of outer
      products ``conj(v) v'`` in MATLAB (``'`` is the non-conjugate
      transpose), vectorized here as ``conj(H) @ H.T / (L-p)`` with
      ``H[i,s] = x[s+p-i]``.
    * **Burg** form: unbiased lag estimates ``Rpos(n) = sum(x[n:] conj(x[:L-n]))/(L-n)``,
      assembled into a Hermitian Toeplitz matrix (first column ``[R0, Rpos]``,
      first row ``[R0, conj(Rpos)]``).
    """
    x = np.asarray(x, dtype=complex).ravel()
    L = x.shape[0]
    p = int(p)
    if p > L:
        raise ValueError("L must be greater than p")
    if p < 1:
        raise ValueError("p must be greater than 0")
    rxtype = rxtype.upper()

    if rxtype in ("HERMITIAN", "FBHERMITIAN"):
        # H[i, s] = x[s + p - i]  (i = 0..p, s = 0..L-p-1)
        idx = (p - np.arange(p + 1))[:, None] + np.arange(L - p)[None, :]
        H = x[idx]
        Rx = np.conj(H) @ H.T / (L - p)
    elif rxtype in ("BURG", "FBBURG"):
        R0 = float(np.sum(np.abs(x) ** 2) / L)
        Rpos = np.empty(p, dtype=complex)
        for n in range(1, p + 1):
            Rpos[n - 1] = np.sum(x[n:L] * np.conj(x[0:L - n])) / (L - n)
        Rfull = np.concatenate([[R0], Rpos])          # Rfull[k] = Rpos(k), k=0..p
        lags = np.arange(p + 1)[:, None] - np.arange(p + 1)[None, :]
        Rx = np.where(lags >= 0, Rfull[np.abs(lags)],
                      np.conj(Rfull[np.abs(lags)]))
    else:
        raise ValueError(f"unknown algorithm name {rxtype}")

    if rxtype[:2] == "FB":
        # MATLAB  Rx = 0.5*(Rx + Rx')  :  Rx' is the CONJUGATE transpose.
        Rx = 0.5 * (Rx + np.conj(Rx.T))
    return Rx


def kaytth(length):
    """Kay-Tretter filter (port of ``kaytth.m``).

    Computes the length-``length`` Kay-Tretter kernel, a weighting used by
    ``instfreq`` to estimate the instantaneous frequency from the short-time
    fourth-order moments (a dot product with ``H``).  The kernel is positive,
    monotonically decreasing, and sums to exactly ``1/2``.

    Parameters
    ----------
    length : int
        Kernel length (``>= 1``).

    Returns
    -------
    H : (length,) float
        The Kay-Tretter kernel, with ``sum(H) == 0.5``.
    """
    n = int(length)
    pp1 = n * (n + 1)
    den = 2.0 * n * (n + 1) * (2.0 * n + 1.0) / 3.0
    i = np.arange(1, n + 1)
    H = (pp1 - i * (i - 1)) / den
    return H


def ifestar2(x, t=None):
    """Instantaneous frequency via an AR(2) "4-point Prony" model
    (port of ``ifestar2.m``).

    Estimates the instantaneous frequency of the real signal ``x`` at the
    time instant(s) ``t`` from the four points ``x(t), x(t-1), x(t-2),
    x(t-3)``.  The estimate ``fnorm`` lies in ``[0, 0.5]`` and has an
    approximate delay of 2.5 points.  Where the short-time determinant is
    not positive the estimator yields no value, so the returned ``t`` is a
    subset of the input and ``rejratio`` is the proportion of instants that
    succeeded.

    Parameters
    ----------
    x : (N,) array_like
        Real signal (the real part is analyzed).
    t : array_like, optional
        Time instants (1-based, each ``> 4``).  Defaults to ``4:N``.

    Returns
    -------
    fnorm : (K,) float
        Instantaneous-frequency estimates in ``[0, 0.5]``.
    t : (K,) int
        The (1-based) time instants for which an estimate was produced.
    rejratio : float
        ``K / len(t)`` — the proportion of instants that succeeded.
    """
    x = np.asarray(x, dtype=complex).ravel()
    xrow = x.shape[0]
    x = x.real

    if t is None:
        t = np.arange(4, xrow + 1)          # MATLAB  t = 4:xrow  (1-based)
    t = np.asarray(t, dtype=int).ravel()
    if t.size == 0:
        raise ValueError("The smallest value of T must be greater than 4")
    if int(t.min()) < 4:
        raise ValueError("The smallest value of T must be greater than 4")

    # 1-based MATLAB indices -> 0-based:  x(t)  -> x[t-1], etc.
    xt = x[t - 1]                            # x(t)
    xt1 = x[t - 2]                           # x(t-1)
    xt2 = x[t - 3]                           # x(t-2)
    xt3 = x[t - 4]                           # x(t-3)

    Kappa = xt1 * xt2 - xt * xt3
    psi1 = xt1 * xt1 - xt * xt2
    psi2 = xt2 * xt2 - xt1 * xt3
    den = psi1 * psi2
    indices = np.where(den > 0)[0]
    arg = 0.5 * Kappa[indices] / np.sqrt(den[indices])
    indarg = np.where(np.abs(arg) > 1)[0]
    arg[indarg] = np.sign(arg[indarg])       # clip to [-1, 1]
    fnorm = np.arccos(arg) / (2.0 * np.pi)
    rejratio = len(indices) / len(t)
    t = t[indices]
    return fnorm, t, rejratio


def parafrep(Rx, N=128, method="capon", q=2):
    """Parametric frequency representation of a signal (port of
    ``parafrep.m``).

    Computes the frequency spectrum of a signal from its correlation matrix
    ``Rx`` (of size ``(p+1) x (p+1)``) using one of six parametric methods.

    Parameters
    ----------
    Rx : (p+1, p+1) array
        Correlation matrix (square), e.g. from :func:`correlmx`.
    N : int, optional
        Number of frequency bins between 0 and 0.5 (default 128).
    method : str, optional
        One of ``'ar'``, ``'periodogram'``, ``'capon'`` (default),
        ``'capnorm'``, ``'lagunas'``, ``'genlag'``.
    q : int, optional
        Order of the generalized Lagunas method (``'genlag'``; default 2).

    Returns
    -------
    spec : (N,) float
        The estimated spectrum (linear, not dB).
    freqs : (N,) float
        The frequency bins, ``linspace(0, 0.5, N)``.
    A : (p+1,) complex or None
        The AR prediction-coefficient vector (``'ar'`` method only); ``None``
        for the other methods.
    """
    Rx = np.asarray(Rx, dtype=complex)
    if Rx.ndim != 2 or Rx.shape[0] != Rx.shape[1]:
        raise ValueError("Rx must be a square matrix")
    Rxrow = Rx.shape[0]
    p = Rxrow - 1
    freqs = np.linspace(0.0, 0.5, N)
    spec = np.zeros(N)
    method = method.upper()
    A = None

    # Z(ifreq) = exp(2j*pi*freqs(ifreq)*(0:Rxrow-1)')  (column vector)
    exponents = np.arange(Rxrow)
    if method == "AR":
        Un = np.ones(Rxrow)
        Rxm1Un = np.linalg.solve(Rx, Un)
        P1 = float(np.real(np.conj(Un) @ Rxm1Un))
        A = Rxm1Un / P1
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            spec[ifreq] = P1 / np.abs(np.conj(Z) @ A) ** 2
    elif method == "PERIODOGRAM":
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            spec[ifreq] = np.real(np.conj(Z) @ Rx @ Z) / (p + 1) ** 2
    elif method == "CAPON":
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            spec[ifreq] = 1.0 / np.real(np.conj(Z) @ np.linalg.solve(Rx, Z))
    elif method == "CAPNORM":
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            spec[ifreq] = (p + 1) / np.real(np.conj(Z) @ np.linalg.solve(Rx, Z))
    elif method == "LAGUNAS":
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            Rxm1Z = np.linalg.solve(Rx, Z)
            spec[ifreq] = np.real(np.conj(Z) @ Rxm1Z) / np.real(
                np.conj(Z) @ np.linalg.solve(Rx, Rxm1Z))
    elif method == "GENLAG":
        Rxq = np.linalg.matrix_power(Rx, int(q))
        for ifreq in range(N):
            Z = np.exp(2j * np.pi * freqs[ifreq] * exponents)
            Rxqm1Z = np.linalg.solve(Rxq, Z)
            spec[ifreq] = np.real(np.conj(Z) @ Rx @ Rxqm1Z) / np.real(
                np.conj(Z) @ np.linalg.solve(Rx, Rxqm1Z))
    else:
        raise ValueError("unknown frequency representation")
    return spec, freqs, A


def tfrparam(x, t=None, N=None, p=None, L=None, trace=0,
             Rxtype="fbhermitian", method="ar", q=2):
    """Parametric time-frequency representation (port of ``tfrparam.m``).

    A sliding-window parametric TFR: at each time instant ``t`` it extracts an
    ``L``-length window (centered on the sample, clipped at the signal edges),
    builds the correlation matrix with :func:`correlmx`, and estimates the
    spectrum with :func:`parafrep`.

    Parameters
    ----------
    x : array_like
        Signal (column vector).
    t : array_like, optional
        Time instants (1-based).  Defaults to ``1:length(x)``.
    N : int, optional
        Number of frequency bins (default ``max(256, length(x))``).
    p : int, optional
        Last autocorrelation lag (default 2).
    L : int, optional
        Window length around the analyzed sample; must be odd (default
        ``max(51, round(length(x)/10))``).  If even, it is corrected to ``L+1``.
    trace : int, optional
        Progress flag (kept for MATLAB signature compatibility; unused).
    Rxtype : str, optional
        Correlation-matrix algorithm (default ``'fbhermitian'``): one of
        ``'hermitian'``, ``'fbhermitian'``, ``'burg'``, ``'fbburg'``.
    method : str, optional
        Parametric method (default ``'ar'``); one of the six ``parafrep``
        methods.
    q : int, optional
        Order of the generalized Lagunas method (default 2; passed to
        ``parafrep`` only when ``method == 'genlag'``).

    Returns
    -------
    tfr : (N, len(t)) float
        The time-frequency representation (each column is one time instant).
    t : (len(t),) int
        The time instants (echoed, 1-based).
    f : (N,) float
        The frequency bins, ``linspace(0, 0.5, N)``.
    """
    x = np.asarray(x, dtype=complex)
    if x.ndim == 2:
        if x.shape[1] > 1:
            raise ValueError("x must be a column vector")
        x = x.ravel()
    else:
        x = x.ravel()
    xrow = x.shape[0]

    # default handling (mirrors the MATLAB nargin ladder)
    if t is None:
        t = np.arange(1, xrow + 1)
    t = np.asarray(t, dtype=int).ravel()
    if N is None:
        N = max(256, xrow)
    if p is None:
        p = 2
    if L is None:
        L = max(51, int(round(xrow / 10)))
    N = int(N)
    p = int(p)
    L = int(L)

    if p < 1:
        raise ValueError("p must be greater than 0")
    if L % 2 == 0:
        L += 1
    Lhalf = (L - 1) // 2

    tfr = np.zeros((N, t.shape[0]))
    f = np.linspace(0.0, 0.5, N)
    for icol in range(t.shape[0]):
        ti = int(t[icol])
        timin = ti - Lhalf
        if timin < 1:
            timin = 1
        timax = ti + Lhalf
        if timax > xrow:
            timax = xrow
        Realp = min(p, timax - timin)
        Rx = correlmx(x[timin - 1:timax], Realp, Rxtype)
        if method.upper() == "GENLAG":
            spec, f, _ = parafrep(Rx, N, method, q)
        else:
            spec, f, _ = parafrep(Rx, N, method)
        tfr[:, icol] = spec
    return tfr, t, f
