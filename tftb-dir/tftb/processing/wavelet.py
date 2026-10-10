# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Continuous wavelet transform with Gaussian-derivative wavelets.

Faithful port of the ``tftb-0.3`` ``contwtgn`` family:

* ``gaussn``   <- ``gaussn.m``       (order-n Gaussian derivative wavelet)
* ``cwt``      <- ``contwtgn.m``     (continuous wavelet transform)
* ``cwt_mirrored`` <- ``contwtgnmir.m`` (CWT of a mirrored signal)

**Oracle note (two certified-toolbox defects, supplied as scratch shims):**
the certified ``tftb-0.3`` references two functions that are *not shipped* in
the toolbox, so its ``gaussn``/``contwtgn``/``contwtgnmir`` error in the
oracle out of the box:

* ``fact2`` (odd double factorial, used by ``gaussn`` for the analytic
  energy) -- undefined. Standard definition supplied in the scratch path.
* ``lenfth`` (``contwtgnmir`` line 21, ``T = 1:lenfth(x)``) -- an obvious
  typo for ``length``, undefined.

Both are unambiguous, so the scratch shims restore the intended behaviour and
let the MATLAB oracle generate goldens; the Python port is validated against
them. The shims live only in ``golden/`` (a scratch path) and do not modify
the certified toolbox.
"""
import numpy as np


def _fact2(n):
    """Odd double factorial ``n!! = n*(n-2)*(n-4)*...`` (1 for n<=1).

    Matches the (missing) MATLAB ``fact2`` used by ``gaussn``.
    """
    y = 1.0
    for k in range(n, 0, -2):
        y *= k
    return y


def _integ(y, x):
    """MATLAB ``integ.m``: trapezoid-like sum ``sum((y[i]+y[i+1])*(dx/2))``.

    Used by ``gaussn`` as the fallback energy when the analytic expression
    overflows.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    dy = y[:-1] + y[1:]
    dx = (x[1:] - x[:-1]) / 2.0
    return float(dy @ dx)


# 4th-order polynomial (ref freq f0 = 0.1) modeling the wavelet support at
# tolerance 1e-3, as in gaussn.m.
_POLY_COEF = np.array([-0.00000001420054, 0.00001199042535, -0.00403466080616,
                       0.92639865727446, 20.64720217778273])


def gaussn(f0, n):
    """Order-``n`` Gaussian-derivative wavelet centered at frequency ``f0``.

    Port of ``tftb-0.3/gaussn.m``. The analytic wavelet is synthesized first in
    the frequency domain, the real part is taken and L2-normalized by the
    analytic energy (with a numerical ``integ`` fallback when the analytic
    expression overflows).

    :param f0: Center frequency (cycles/sample).
    :param n: Order of the Gaussian derivative (wavelet order; >= 1).
    :return: Real, L2-normalized wavelet (``2N-1`` samples).
    """
    alpha = 2.0 * np.pi ** 2 * f0 ** 2 / n
    N = int(np.ceil((0.1 / f0) * np.polyval(_POLY_COEF, n)))
    f = np.linspace(0, 1, 2 * N + 1)[:2 * N]
    G = (2.0 * np.pi / alpha) ** 0.25 * (2j * np.pi * f) ** n \
        * np.exp(-(np.pi ** 2 * f ** 2) / alpha)
    p = (2.0 * np.pi ** 2) / alpha
    E = (2.0 * np.pi / alpha) ** 0.5 * (2.0 * np.pi) ** (2 * n) \
        * (_fact2(2 * n - 1) / (2.0 * (2.0 * p) ** n)) * np.sqrt(np.pi / p)
    if np.isinf(E) or np.isnan(E) or E == 0:
        E = _integ(np.abs(G) ** 2, f)
    g = np.real(np.fft.ifftshift(np.fft.ifft(G)) / np.sqrt(E / 2.0))
    return g[1:2 * N]  # MATLAB g(2:2*N), 1-based


def cwt(signal, fmin=0.01, fmax=0.5, n_voices=None, wave=2):
    """Continuous wavelet transform (``contwtgn``).

    :param signal: 1-D signal.
    :param fmin: Lower frequency bound (cycles/sample).
    :param fmax: Upper frequency bound (cycles/sample).
    :param n_voices: Number of (geometrically sampled) frequencies.
    :param wave: Wavelet order (Gaussian derivative order; default 2).
    :return: ``(scalo, f, T, a, wt)`` -- scalogram (|wt|²), frequency samples
        (ascending, fmin first), time samples, scale vector (ascending), and
        the complex wavelet coefficients (rows = ascending frequency, cols =
        time).
    """
    x = np.asarray(signal, dtype=complex).ravel()
    if fmin >= fmax:
        raise ValueError('fmax must be greater than fmin')
    if n_voices is None:
        n_voices = x.shape[0]
    nt = x.shape[0]
    T = np.arange(1, nt + 1)
    f = np.logspace(np.log10(fmax), np.log10(fmin), n_voices)
    a = np.logspace(0.0, np.log10(fmax / fmin), n_voices)
    wt = np.zeros((n_voices, nt), dtype=complex)
    for ptr in range(n_voices):
        ha = gaussn(f[ptr], wave)
        nha = (ha.shape[0] - 1) // 2
        # MATLAB conv(x,ha) is cross-correlation with the unflipped kernel.
        detail = np.convolve(x, ha, mode='full')
        wt[ptr, :] = detail[nha:nha + nt]
    # TFTB convention: first row = lowest frequency.
    wt = wt[::-1, :]
    f = f[::-1]
    a = a[::-1]
    scalo = wt * np.conj(wt)
    return scalo, f, T, a, wt


def cwt_mirrored(signal, fmin=0.01, fmax=0.5, n_voices=None, wave=2):
    """Continuous wavelet transform of a mirrored signal (``contwtgnmir``).

    Mirrors ``nbmir = min(nt, nha)`` samples on each side before the CWT to
    reduce boundary artefacts (the number of mirrored samples depends on the
    scale/wavelet length, so it is done per voice).

    :param signal: 1-D signal.
    :param fmin: Lower frequency bound (cycles/sample).
    :param fmax: Upper frequency bound (cycles/sample).
    :param n_voices: Number of (geometrically sampled) frequencies.
    :param wave: Wavelet order (default 2).
    :return: ``(scalo, f, T, a, wt)`` as in :func:`cwt`.
    """
    x = np.asarray(signal, dtype=complex).ravel()
    if fmin >= fmax:
        raise ValueError('fmax must be greater than fmin')
    if n_voices is None:
        n_voices = x.shape[0]
    nt = x.shape[0]
    T = np.arange(1, nt + 1)
    f = np.logspace(np.log10(fmax), np.log10(fmin), n_voices)
    a = np.logspace(0.0, np.log10(fmax / fmin), n_voices)
    wt = np.zeros((n_voices, nt), dtype=complex)
    for ptr in range(n_voices):
        ha = gaussn(f[ptr], wave)
        nha = (ha.shape[0] - 1) // 2
        nbmir = min(nt, nha)
        # MATLAB: [x(nbmir:-1:2) x x(nt-1:-1:nt-nbmir+1)], 1-based.
        left = x[nbmir - 1:0:-1]      # x(nbmir),...,x(2)
        right = x[nt - 2:nt - nbmir - 1:-1]  # x(nt-1),...,x(nt-nbmir+1)
        x_mir = np.concatenate([left, x, right])
        detail = np.convolve(x_mir, ha, mode='full')
        # MATLAB detail(nha+nbmir : nha+nbmir+nt-1), 1-based.
        wt[ptr, :nt] = detail[nha + nbmir - 1:nha + nbmir + nt - 1]
    wt = wt[::-1, :]
    f = f[::-1]
    a = a[::-1]
    scalo = wt * np.conj(wt)
    return scalo, f, T, a, wt
