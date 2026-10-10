#! /usr/bin/env python
# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Miscellaneous processing utilities."""

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.special import j0, jn, i0, i1


def get_spectrum(signal):
    return np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)


def tftb_window(n, name='Hamming', param=None, param2=None):
    """Window generation — faithful port of MATLAB ``tftb_window.m`` (F. Auger,
    CNRS 1996).

    IMPORTANT: the MATLAB default window (``name='Hamming'``) is
    ``0.54 - 0.46*cos(2*pi*(1:N)/(N+1))``, which is NOT the same as
    ``scipy.signal.windows.hamming`` (scipy uses a different index spacing and
    different endpoints). The MATLAB toolbox uses this function for every
    windowed TFR, so the Python port must use it too to match the golden
    references.

    :param n: Length of the window (positive integer).
    :param name: Window shape — 'Hamming' (default), 'Hanning', 'Nuttall',
        'Papoulis', 'Harris', 'Rect', 'Triang', 'Bartlett', 'BartHann',
        'Blackman', 'Gauss', 'Parzen', 'Kaiser', 'Dolph', 'Hanna', 'Nutbess',
        'Spline', 'Flattop'.
    :param param: Optional parameter (Gauss edge value, Kaiser scale, ...).
    :param param2: Second optional parameter (Nutbess nu, Spline order p).
    :return: Window as a 1-D numpy array of length ``n``.
    :rtype: numpy.ndarray
    """
    if n <= 0:
        raise ValueError("N should be strictly positive.")
    n = int(n)
    name = str(name).upper()
    if name in ('RECTANG', 'RECT'):
        return np.ones(n)
    if name == 'HAMMING':
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(1, n + 1) / (n + 1))
    if name in ('HANNING', 'HANN'):
        return 0.50 - 0.50 * np.cos(2.0 * np.pi * np.arange(1, n + 1) / (n + 1))
    if name == 'KAISER':
        beta = 3.0 * np.pi  # MATLAB tftb_window hard-codes beta=3*pi
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 / n
        # MATLAB: besselj(0, 1j*beta*x) = modified Bessel i0(beta*x) (real).
        return i0(beta * np.sqrt(1.0 - ind ** 2)) / i0(beta)
    if name == 'NUTTALL':
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 * np.pi / n
        return (+0.3635819 + 0.4891775 * np.cos(ind)
                + 0.1363995 * np.cos(2.0 * ind) + 0.0106411 * np.cos(3.0 * ind))
    if name == 'BLACKMAN':
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 * np.pi / n
        return +0.42 + 0.50 * np.cos(ind) + 0.08 * np.cos(2.0 * ind)
    if name == 'HARRIS':
        ind = np.arange(1, n + 1) * 2.0 * np.pi / (n + 1)
        return (+0.35875 - 0.48829 * np.cos(ind)
                + 0.14128 * np.cos(2.0 * ind) - 0.01168 * np.cos(3.0 * ind))
    if name in ('BARTLETT', 'TRIANG'):
        a = np.arange(1, n + 1)
        return 2.0 * np.minimum(a, n - a + 1) / (n + 1)
    if name == 'BARTHANN':
        a = np.arange(1, n + 1)
        return (0.38 * (1.0 - np.cos(2.0 * np.pi * a / (n + 1)))
                + 0.48 * np.minimum(a, n - a + 1) / (n + 1))
    if name == 'PAPOULIS':
        return np.sin(np.arange(1, n + 1) * np.pi / (n + 1))
    if name == 'GAUSS':
        k = param if param is not None else 0.005
        return np.exp(np.log(k) * np.linspace(-1, 1, n) ** 2)
    if name == 'PARZEN':
        ind = np.abs(np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1)) * 2.0 / n
        temp = 2.0 * (1.0 - ind) ** 3
        return np.minimum(temp - (1 - 2.0 * ind) ** 3, temp)
    if name == 'HANNA':
        ell = param if param is not None else 1
        ind = np.arange(n)
        return np.sin((2 * ind + 1) * np.pi / (2 * n)) ** (2 * ell)
    if name in ('DOLPH', 'DOLF'):
        oddn = 1 if (n % 2 == 0) else 0
        if oddn:
            n = 2 * n + 1
        a = 10 ** (param / 20.0) if param is not None else 1e-3
        k = n - 1
        z0 = np.cosh(np.arccosh(1.0 / a) / k)
        x0 = np.arccos(1.0 / z0) / np.pi
        x = np.arange(0, k + 1) / n
        h = np.zeros(n)
        idx1 = np.nonzero((x < x0) | (x > 1 - x0))[0]
        idx2 = np.nonzero((x >= x0) & (x <= 1 - x0))[0]
        # MATLAB computes cosh(K*acosh(z)) on the outer lobes; there |z|>=1 and
        # on the right lobe z<=-1, where MATLAB's complex acosh is real but
        # numpy's real-only arccosh is NaN.  cosh(K*acosh(z)) == T_K(z) (the
        # Chebyshev polynomial of the first kind) for all real z, so evaluate
        # it that way — identical to MATLAB, finite on both lobes.
        _ck = np.zeros(k + 1)
        _ck[k] = 1.0
        h[idx1] = chebval(z0 * np.cos(np.pi * x[idx1]), _ck)
        h[idx2] = np.cos(k * np.arccos(z0 * np.cos(np.pi * x[idx2])))
        h = np.fft.fftshift(np.real(np.fft.ifft(a * np.real(h))))
        h = h / h[k // 2]
        if oddn:
            h = h[1::2]
        return h
    if name == 'NUTBESS':
        if param is None:
            beta, nu = 3 * np.pi, 0.5
        else:
            beta = param
            nu = param2 if param2 is not None else 0.5
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 / n
        return (np.sqrt(1 - ind ** 2) ** nu
                * np.real(jn(nu, 1j * beta * np.sqrt(1.0 - ind ** 2)))
                / np.real(jn(nu, 1j * beta)))
    if name == 'SPLINE':
        if param is None:
            raise ValueError("Three or four parameters required for spline windows")
        nfreq = param
        p = np.pi * n * nfreq / 10.0 if param2 is None else param2
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1)
        return np.sinc((0.5 * nfreq / p) * ind) ** p
    if name == 'FLATTOP':
        ind = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1) * 2.0 * np.pi / (n - 1)
        return (+0.2810639 + 0.5208972 * np.cos(ind) + 0.1980399 * np.cos(2.0 * ind))
    raise ValueError(f"unknown window name: {name}")


def integrate_2d(mat, x=None, y=None):
    """integrate_2d

    :param mat:
    :param x:
    :param y:
    :type mat:
    :type x:
    :type y:
:return:
:rtype:
    :Example:
    >>> from __future__ import print_function
    >>> from tftb.generators import altes
    >>> from tftb.processing import Scalogram
    >>> x = altes(256, 0.1, 0.45, 10000)
    >>> tfr, t, f, _ = Scalogram(x).run()
    >>> print("%.3f" % integrate_2d(tfr, t, f))
    2.000
    """
    m, n = mat.shape
    if x is None:
        x = np.arange(n)
    if y is None:
        y = np.arange(m)

    mat = (mat.sum(1) - mat[:, 0] / 2.0 - mat[:, n - 1] / 2.0) * (x[1] - x[0])
    dmat = mat[:(m - 1)] + mat[1:]
    dy = (y[1:] - y[:(m - 1)]) / 2.0
    return np.sum(dmat * dy)


def derive_window(window):
    """Calculate derivative of a window function.

    :param window: Window function to be differentiated. This is expected to be
    a standard window function with an odd length.
    :type window: array-like
    :return: Derivative of the input window
    :rtype: array-like
    :Example:
    >>> from scipy.signal import hann
    >>> import matplotlib.pyplot as plt                   #doctest: +SKIP
    >>> window = hann(210)
    >>> derivation = derive_window(window)
    >>> plt.subplot(211), plt.plot(window)                #doctest: +SKIP
    >>> plt.subplot(212), plt.plot(derivation) #doctest: +SKIP

    .. plot:: docstring_plots/processing/utils/derive_window.py
    """
    lh = (window.shape[0] - 1) / 2.0
    step_height = (window[0] + window[-1]) / 2.0
    # MATLAB dwindow: ramp = (h(hrow)-h(1))/(hrow-1)  (last - first).
    ramp = (window[-1] - window[0]) / (window.shape[0] - 1)
    base = np.arange(-lh, lh + 1)
    base = window - step_height - ramp * base
    base = np.hstack((np.array([0]), base, np.array([0])))
    dw = (base[2:(window.shape[0] + 2)] - base[:window.shape[0]]) / 2.0 + ramp
    dw[0] += step_height
    dw[-1] -= step_height
    return dw


if __name__ == '__main__':
    x = np.arange(1, 16).reshape(5, 3)
    print(integrate_2d(x))
