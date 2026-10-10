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

"""
Linear Time Frequency Processing.
"""

import numpy as np
from tftb.processing.base import BaseTFRepresentation
from tftb.processing.utils import derive_window
from tftb.utils import nearest_odd, divider, modulo, izak
from tftb._accel import backend as _accel_backend


def _matlab_round(x):
    """MATLAB ``round`` (half away from zero) for real x.

    numpy's ``round`` is half-to-even, which differs from MATLAB for ``.5``
    values; ``floor(x+0.5)`` matches MATLAB for positive x (the only case the
    Stanković tau-range uses).
    """
    x = np.asarray(x, dtype=float)
    return np.floor(x + 0.5)


class ShortTimeFourierTransform(BaseTFRepresentation):
    """Short time Fourier transform."""

    name = "stft"

    def __init__(self, signal, timestamps=None, n_fbins=None, fwindow=None):
        """Create a ShortTimeFourierTransform object.

        :param signal: Signal to be analyzed.
        :param timestamps: Time instants of the signal (default:
            ``np.arange(len(signal))``)
        :param n_fbins: Number of frequency bins (default: ``len(signal)``)
        :param fwindow: Frequency smoothing window (default: Hamming window of
            length ``len(signal) / 4``)
        :type signal: array-like
        :type timestamps: array-like
        :type n_fbins: int
        :type fwindow: array-like
        :return: ShortTimeFourierTransform object
        :Example:

        >>> from tftb.generators import fmconst
        >>> sig = np.r_[fmconst(128, 0.2)[0], fmconst(128, 0.4)[0]]
        >>> stft = ShortTimeFourierTransform(sig)
        >>> tfr, t, f = stft.run()
        >>> stft.plot() #doctest: +SKIP

        .. plot:: docstring_plots/processing/stft.py
        """
        super(ShortTimeFourierTransform, self).__init__(signal=signal,
                                                        n_fbins=n_fbins,
                                                        timestamps=timestamps,
                                                        fwindow=fwindow)

    def run(self):
        r"""Compute the STFT according to:

        .. math:: X[m, w] = \sum_{n=-\infty}^{\infty}x[n]w[n - m]e^{-j\omega n}

        Where :math:`w` is a Hamming window."""
        lh = (self.fwindow.shape[0] - 1) // 2
        rangemin = min([round(self.n_fbins / 2.0), lh])
        starts = -np.min(np.c_[rangemin * np.ones(self.ts.shape),
                               np.arange(self.ts.shape[0]) - 1],
                         axis=1).astype(int)
        ends = np.min(np.c_[rangemin * np.ones(self.ts.shape),
                            self.signal.shape[0] - np.arange(self.ts.shape[0])],
                      axis=1).astype(int)
        conj_fwindow = np.conj(self.fwindow)
        for icol in range(self.tfr.shape[1]):
            start = starts[icol]
            end = ends[icol]
            tau = np.arange(start, end + 1).astype(int)
            index = np.remainder(self.n_fbins + tau, self.n_fbins)
            self.tfr[index, icol] = self.signal[(icol + tau - 1).astype(int)] * \
                conj_fwindow[(lh + tau).astype(int)]
        self.tfr = np.fft.fft(self.tfr, axis=0)
        return self.tfr, self.ts, self.freqs

    def plot(self, ax=None, kind='cmap', sqmod=True, threshold=0.05, **kwargs):
        """Display the spectrogram of an STFT.

        :param ax: axes object to draw the plot on. If None(default), one will
            be created.
        :param kind: Choice of visualization type, either "cmap"(default) or "contour".
        :param sqmod: Whether to take squared modulus of TFR before plotting.
            (Default: True)
        :param threshold: Percentage of the maximum value of the TFR, below
            which all values are set to zero before plotting.
        :param **kwargs: parameters passed to the plotting function.
        :type ax: matplotlib.axes.Axes object
        :type kind: str
        :type sqmod: bool
        :type threshold: float
        :return: None
        :rtype: None
        """
        self.tfr = self.tfr[:int(self.n_fbins / 2.0), :]
        self.freqs = self.freqs[:int(self.n_fbins / 2.0)]
        if sqmod:
            self.tfr = np.abs(self.tfr) ** 2
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(ShortTimeFourierTransform, self).plot(ax=ax, kind=kind,
                                                    threshold=threshold,
                                                    **kwargs)


class StankovicDistribution(BaseTFRepresentation):
    """Stanković distribution (base, non-reassigned).

    MATLAB ``tfrrstan.m`` (F. Auger, 1997). Computes the Stanković
    distribution: a frequency-smoothed STFT where the smoothing is a circular
    bilinear convolution with the frequency window ``G``.

    For each time column the (complex) STFT column ``X`` is formed, then the
    distribution is::

        V[j] = G[LG] |X[j]|² + Σ_{k=1}^{min(N/2-1, LG)}
                ( G[LG-k] X[j-k] conj(X[j+k]) + G[LG+k] X[j+k] conj(X[j-k]) )

    (indices circular modulo ``N``), and the result is taken real. The default
    frequency window is ``G = [0.25, 0.5, 0.25]`` and the default STFT window
    is a Hamming of odd length ``floor(N/4)`` (matching MATLAB's
    ``hlength = floor(N/4)+1-rem(hlength,2)``).

    ``run()`` also returns the reassigned version (``rtfr``) and the complex
    reassignment-vector matrix ``hat`` (Phase 3, Task 3.4): the two derivative
    TFRs (``stanTh`` = STFT weighted by ``h·tau``, ``stanDh`` = STFT weighted
    by the derived window ``dwindow(h)``) give the time/frequency reassignment
    shifts, and energy above the ``1e-3·mean(|x|²)`` threshold is scattered to
    the reassigned bin (frequency wrapped modulo ``N``, time clamped). Sub-
    threshold bins stay in place and ``hat`` records their original 1-based
    position (``jcol + 1j·icol``), unlike the ``tfrrsp`` family which records
    ``inf``.
    """

    name = "stankovic"

    def __init__(self, signal, timestamps=None, n_fbins=None, fwindow=None,
                 fsmooth=None):
        """Create a StankovicDistribution object.

        :param signal: Signal to be analysed.
        :param timestamps: Time instants (default ``np.arange(len(signal))``).
        :param n_fbins: Number of frequency bins (default ``len(signal)``).
        :param fwindow: STFT (time) window (default: odd-length Hamming).
        :param fsmooth: Frequency smoothing window ``G`` (odd length; default
            ``[0.25, 0.5, 0.25]``).
        """
        super(StankovicDistribution, self).__init__(signal=signal,
                                                    n_fbins=n_fbins,
                                                    timestamps=timestamps,
                                                    fwindow=fwindow)
        if fsmooth is None:
            fsmooth = np.array([0.25, 0.5, 0.25])
        self.fsmooth = np.asarray(fsmooth, dtype=float).ravel()
        if self.fsmooth.shape[0] % 2 == 0:
            raise ValueError('The frequency smoothing window must have an odd length.')
        self.lg = (self.fsmooth.shape[0] - 1) // 2

    def run(self):
        """Compute the Stanković distribution and its reassigned version.

        Returns ``(stan, ts, freqs, rtfr, hat)``: the base distribution, the
        time/frequency axes, the reassigned distribution, and the complex
        reassignment-vector matrix (1-based ``jcol + 1j·icol``, original
        position at sub-threshold bins).
        """
        N = self.n_fbins
        xrow = self.signal.shape[0]
        lh = (self.fwindow.shape[0] - 1) // 2
        lg = self.lg
        G = self.fsmooth
        h = self.fwindow
        dh = derive_window(h)
        th = h * np.arange(-lh, lh + 1)  # Th = h .* [-Lh:Lh]'
        nmax = int(_matlab_round(N / 2.0)) - 1
        tmin = int(self.ts.min())
        tmax = int(self.ts.max())
        threshold = 1.0e-3 * np.mean(np.abs(self.signal[tmin:tmax + 1]) ** 2)
        # Dt: the uniform time step of the timestamps (1 for the default
        # np.arange(N)); MATLAB computes it from the t differences.
        dt = float(np.diff(self.ts)[0]) if self.ts.shape[0] > 1 else 1.0
        nts = self.ts.shape[0]
        kmax = min(N // 2 - 1, lg)
        # Build the three FFT'd STFT matrices T1/T2/T3 once; both the Numba and
        # numpy paths consume them (only the O(N*kmax) jcol loop + reassignment
        # scatter differ).
        T1 = np.zeros((N, nts), dtype=complex)
        T2 = np.zeros((N, nts), dtype=complex)
        T3 = np.zeros((N, nts), dtype=complex)
        for c in range(nts):
            ti = int(self.ts[c])
            tau_low = -min([nmax, lh, ti])
            tau_high = min([nmax, lh, xrow - 1 - ti])
            tau = np.arange(tau_low, tau_high + 1)
            xseg = self.signal[(ti + tau).astype(int)]
            hseg = np.conj(h[(lh - tau).astype(int)])
            thseg = np.conj(th[(lh - tau).astype(int)])
            dhseg = np.conj(dh[(lh - tau).astype(int)])
            idx = np.remainder(N + tau, N)
            c1 = np.zeros(N, dtype=complex); c1[idx] = xseg * hseg
            c2 = np.zeros(N, dtype=complex); c2[idx] = xseg * thseg
            c3 = np.zeros(N, dtype=complex); c3[idx] = xseg * dhseg
            T1[:, c] = np.fft.fft(c1)
            T2[:, c] = np.fft.fft(c2)
            T3[:, c] = np.fft.fft(c3)
        # Phase 5.1: Numba jcol-loop + reassignment scatter (prange over
        # columns), auto-fallback to the numpy loop below.
        if _accel_backend.get_backend("stank", xrow) == "numba":
            try:
                from tftb._accel import kernels as _k
                stan_real, stanTh, stanDh = _k.stank_jcols(
                    T1, T2, T3, G, lg, N, nts, kmax)
                rtfr, hat = _k.stank_scatter(
                    stan_real, stanTh, stanDh, N, nts, threshold, dt)
                # Match the numpy path's dtype (complex, holding real values).
                self.tfr = stan_real.astype(complex)
                return self.tfr, self.ts, self.freqs, rtfr, hat
            except Exception:
                _accel_backend.report_failure("stank")
        rtfr = np.zeros((N, nts), dtype=complex)
        hat = np.zeros((N, nts), dtype=complex)
        for c in range(nts):
            tfr1 = T1[:, c]
            tfr2 = T2[:, c]
            tfr3 = T3[:, c]
            stan_col = (G[lg] * np.abs(tfr1) ** 2).astype(complex)
            stanTh_col = (G[lg] * tfr1 * np.conj(tfr2)).astype(complex)
            stanDh_col = (G[lg] * tfr1 * np.conj(tfr3)).astype(complex)
            for jcol in range(N):
                for k in range(1, kmax + 1):
                    # MATLAB 1-based before=rem(jcol-k-1,N)+1, after=
                    # rem(jcol+k-1,N)+1 -> 0-based (jcol-k)%N / (jcol+k)%N.
                    before = (jcol - k) % N
                    after = (jcol + k) % N
                    stan_col[jcol] += (G[lg - k] * tfr1[before] * np.conj(tfr1[after])
                                       + G[lg + k] * tfr1[after] * np.conj(tfr1[before]))
                    # The derivative TFRs use the same symmetric k-loop as the
                    # base stan, with the derivative STFT in the conjugate slot:
                    #   G[lg-k] t1[before] conj(t{2,3}[after])
                    # + G[lg+k] t1[after]  conj(t{2,3}[before])
                    stanTh_col[jcol] += (G[lg - k] * tfr1[before] * np.conj(tfr2[after])
                                         + G[lg + k] * tfr1[after] * np.conj(tfr2[before]))
                    stanDh_col[jcol] += (G[lg - k] * tfr1[before] * np.conj(tfr3[after])
                                         + G[lg + k] * tfr1[after] * np.conj(tfr3[before]))
                stan_col[jcol] = stan_col[jcol].real
                if abs(stan_col[jcol]) > threshold:
                    # MATLAB: icolhat = round(icol - real(stanTh/stan)/Dt)
                    # (icol is 1-based = c+1), jcolhat = round(jcol -
                    # N*imag(stanDh/stan)/(2*pi)) (jcol 1-based = jcol+1).
                    icolhat = int(_matlab_round((c + 1)
                                 - np.real(stanTh_col[jcol] / stan_col[jcol]) / dt))
                    jcolhat = int(_matlab_round((jcol + 1)
                                 - N * np.imag(stanDh_col[jcol] / stan_col[jcol]) / (2.0 * np.pi)))
                    jcolhat = ((jcolhat - 1) % N) + 1  # rem(rem(jcolhat-1,N)+N,N)+1 (1-based)
                    icolhat = min(max(icolhat, 1), self.ts.shape[0])
                    # rtfr is 0-based; hat is stored 1-based (MATLAB convention).
                    rtfr[jcolhat - 1, icolhat - 1] += stan_col[jcol]
                    hat[jcol, c] = jcolhat + 1j * icolhat
                else:
                    rtfr[jcol, c] += stan_col[jcol]
                    hat[jcol, c] = (jcol + 1) + 1j * (c + 1)
            self.tfr[:, c] = stan_col
        return self.tfr, self.ts, self.freqs, rtfr, hat


def gabor(signal, n_coeff=None, q_oversample=None, window=None):
    """Compute the Gabor representation of a signal.

    :param signal: Singal to be analyzed.
    :param n_coeff: number of Gabor coefficients in time.
    :param q_oversample: Degree of oversampling
    :param window: Synthesis window
    :type signal: array-like
    :type n_coeff: integer
    :type q_oversample: int
    :type window: array-like
    :return: Tuple of Gabor coefficients, biorthogonal window associated with the synthesis window.
    :rtype: tuple
    """
    if n_coeff is None:
        n_coeff, _ = divider(signal.shape[0])
    if q_oversample is None:
        q_oversample, _ = divider(n_coeff)
    if window is None:
        window = np.exp(np.log(0.005) * np.linspace(-1, 1, nearest_odd(n_coeff)) ** 2)
        window = window / np.linalg.norm(window)
    m = int(q_oversample * signal.shape[0] / float(n_coeff))
    mb = int(signal.shape[0] / float(n_coeff))
    nb = int(signal.shape[0] / float(m))

    # Zak transform?
    nh = window.shape[0]
    if nh % 2 == 0:
        raise ValueError("The window function should have an odd length.")
    alpha = np.round((2 * signal.shape[0] / float(n_coeff) - 1 - nh) / (2 * q_oversample))
    hn1 = np.zeros((signal.shape[0],))
    start = np.round(((signal.shape[0] - (nh - 1))) / 2) - alpha
    end = np.round((signal.shape[0] + nh - 1) / 2) - alpha
    hn1[np.arange(start - 1, end).astype(int)] = window

    msig = hn1.reshape(int(nb), int(m), order='F')
    dzth = np.fft.fft(msig.T, axis=0) / np.sqrt(m)
    mzh = np.zeros((m, nb))
    x = np.arange(1, m + 1, dtype=float)
    for l in range(q_oversample):  # NOQA: E741
        mod = modulo(x - l * m / q_oversample, m).astype(int)
        mzh += np.abs(dzth[mod - 1, :]) ** 2

    mzh[mzh < np.spacing(1)] = 1

    # Za transform of biorthogonal dual frame window gam
    dztgam = dzth / mzh
    gam = np.real(izak(dztgam)) / signal.shape[0]

    # Computation of Gabor coefficient of dual frame window.
    dgrn1 = np.zeros((signal.shape[0], n_coeff), dtype=complex)
    k = np.arange(1, signal.shape[0] + 1)
    for n in range(n_coeff):
        index = modulo(k - (n + 1) * m / q_oversample, signal.shape[0]).astype(int) - 1
        dgrn1[:, n] = np.fft.fft(signal * np.fft.fftshift(gam[index]), axis=0)
    dgr = dgrn1[np.arange(signal.shape[0], step=nb).astype(int), :]
    tfr = np.abs(dgr) ** 2
    return tfr, dgr, gam


if __name__ == '__main__':
    from tftb.generators import fmconst
    import matplotlib.pyplot as plt
    sig = np.r_[fmconst(128, 0.2)[0], fmconst(128, 0.4)[0]]
    ts = np.linspace(0, 1, 256)
    tfr = ShortTimeFourierTransform(sig, timestamps=ts)
    tfr.run()
    tfr.plot(show_tf=True, cmap=plt.cm.viridis)
