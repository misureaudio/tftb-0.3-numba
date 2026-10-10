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
Bilinear Time-Frequency Processing in the Cohen’s Class.
"""

import numpy as np
from tftb.processing.linear import ShortTimeFourierTransform, _matlab_round
from tftb.processing.base import BaseTFRepresentation
from tftb.processing.utils import tftb_window
from tftb._accel import backend as _accel_backend


class Spectrogram(ShortTimeFourierTransform):

    name = "spectrogram"

    def run(self):
        lh = (self.fwindow.shape[0] - 1) // 2
        rangemin = min([round(self.n_fbins / 2.0) - 1, lh])
        starts = -np.min(np.c_[rangemin * np.ones(self.ts.shape), self.ts - 1],
                         axis=1).astype(int)
        ends = np.min(np.c_[rangemin * np.ones(self.ts.shape),
                      self.signal.shape[0] - self.ts], axis=1).astype(int)
        conj_fwindow = np.conj(self.fwindow)
        for icol in range(self.tfr.shape[1]):
            ti = self.ts[icol]
            start = starts[icol]
            end = ends[icol]
            tau = np.arange(start, end + 1).astype(int)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins)
            self.tfr[indices.astype(int), icol] = self.signal[ti + tau - 1] * \
                conj_fwindow[lh + tau] / np.linalg.norm(self.fwindow[lh + tau])
        self.tfr = np.abs(np.fft.fft(self.tfr, axis=0)) ** 2
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', **kwargs):
        thresh = kwargs.pop("threshold", 0.0)
        super(Spectrogram, self).plot(kind=kind, sqmod=False, threshold=thresh,
                                      **kwargs)


class PageRepresentation(BaseTFRepresentation):

    name = "page representation"

    def run(self):
        for icol in range(self.ts.shape[0]):
            ti = self.ts[icol]
            tau = np.arange(-min([self.n_fbins - ti,
                                  self.signal.shape[0] - ti]), ti)
            indices = np.remainder(self.n_fbins + tau + 1, self.n_fbins)
            self.tfr[indices, icol] = np.dot(self.signal[ti],
                                             np.conj(self.signal[ti - tau - 1]))
        self.tfr = np.real(np.fft.fft(self.tfr, axis=0))
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, sqmod=True, **kwargs):
        self.tfr = self.tfr[:(self.tfr.shape[0] // 2), :]
        self.tfr = np.abs(self.tfr) ** 2
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(PageRepresentation, self).plot(kind=kind, **kwargs)


class PseudoPageRepresentation(PageRepresentation):

    name = "pseudo page"

    def _make_window(self):
        hlength = int(np.floor(self.n_fbins / 4.0))
        if hlength % 2 == 0:
            hlength += 1
        # MATLAB tfrppage uses tftb_window(hlength) (symmetric 'Hamming'),
        # NOT scipy.signal.hamming (periodic form) — the two differ (N+1 vs
        # N-1 denominator). Using scipy corrupted the default window for every
        # N (caught by the ppage golden cases).
        fwindow = tftb_window(hlength, 'Hamming')
        lh = (fwindow.shape[0] - 1) // 2
        return fwindow / fwindow[lh]

    def run(self):
        lh = (self.fwindow.shape[0] - 1) // 2
        # MATLAB tfrppage (1-based ti = icol+1): tau = 0:min(N-1, Lh, ti-1)
        # -> 0-based tau = 0:min(N-1, lh, icol); row = rem(N+tau, N) (NO +1 —
        # the earlier port shifted every row up by one, corrupting every N).
        for icol in range(self.ts.shape[0]):
            tau = np.arange(min([self.n_fbins - 1, lh, icol]) + 1).astype(int)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins).astype(int)
            self.tfr[indices, icol] = self.fwindow[lh + tau] * \
                self.signal[icol] * np.conj(self.signal[icol - tau])
        self.tfr = np.real(np.fft.fft(self.tfr, axis=0))
        return self.tfr, self.ts, self.freqs


class RihaczekDistribution(BaseTFRepresentation):
    """Rihaczek time-frequency distribution (complex Cohen class).

    The Rihaczek distribution is the complex version of the Page representation:
    its real part equals the Margenau-Hill distribution.

    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.1, 0.4)[0]
    >>> tfr, t, f = RihaczekDistribution(sig).run()
    >>> RihaczekDistribution(sig).plot() #doctest: +SKIP
    """

    name = "rihaczek"

    def __init__(self, signal, timestamps=None, n_fbins=None, fwindow=None, **kwargs):
        # Capture the raw signal BEFORE the base ravels it, so the cross-Ri
        # (two-column) case can be detected. BaseTFRepresentation ravels 2-D.
        self._raw_signal = np.asarray(signal)
        super(RihaczekDistribution, self).__init__(
            signal, timestamps=timestamps, n_fbins=n_fbins, fwindow=fwindow,
            **kwargs)

    def run(self):
        # Shared pre-FFT array (auto + cross), matching tfrri.m exactly:
        #   for icol: ti = t(icol); tau = -min([N-ti, xrow-ti]) : (ti-1)
        #   indices = rem(N+tau, N)+1
        #   tfr(indices, icol) = x(ti,1) * conj(x(ti-tau, xcol))
        raw = self._raw_signal
        if raw.ndim == 2 and raw.shape[1] == 2:
            x1 = raw[:, 0]
            x2 = raw[:, 1]
            n = raw.shape[0]
        else:
            x1 = x2 = self.signal
            n = self.signal.shape[0]
        for icol in range(self.ts.shape[0]):
            ti = int(self.ts[icol])
            tau = np.arange(-min([self.n_fbins - ti, n - ti]), ti).astype(int)
            indices = np.remainder(self.n_fbins + tau + 1, self.n_fbins)
            self.tfr[indices, icol] = x1[ti] * np.conj(x2[ti - tau - 1])
        self.tfr = np.fft.fft(self.tfr, axis=0)
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, **kwargs):
        self.tfr = np.real(self.tfr)[:self.tfr.shape[0] // 2, :]
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(RihaczekDistribution, self).plot(kind=kind, **kwargs)


class GeneralizedCohenBase(BaseTFRepresentation):
    """Shared skeleton for generalized Cohen-class distributions with a time
    smoothing window ``g`` and a frequency smoothing window ``h``:
    Born-Jordan, Zao-Atlas-Marks, and the reduced-interference family.

    All of tfrbj.m / tfrzam.m / tfrrid*.m share this exact loop structure;
    they differ only in four knobs (see class attributes) and the kernel.

    MATLAB loop (1-based), translated to 0-based:
        ti=t(icol); t0=ti-1; taumax=min(t0+Lg, n-t0-1+Lg, round(N/2)-1, Lh)
        tfr[0,icol] = [_g_factor] x1[t0]*conj(x2[t0])
        for tau=1..taumax:
            p_lo=-min(tau,Lg,n-t0-1-tau); p_hi=min(tau,Lg,t0-tau)
            g2 = g[Lg+points] [.* kernel(tau,points)] [/sum(g2)]
            R  = sum(g2 .* x1[t0+tau-points] .* conj(x2[t0-tau-points]))
            tfr[tau,icol]   = h[Lh+tau]*R
            R  = sum(g2 .* x1[t0-tau-points] .* conj(x2[t0+tau-points]))
            tfr[N-tau,icol] = h[Lh-tau]*R
        [special] tau=round(N/2): tfr[tau,icol]=0.5*(...+...)
        tfr=fft(tfr); if auto: tfr=real(tfr)
    """

    name = "generalized-cohen"
    _g_center_norm = False   # g = g / g[Lg] at setup (ZAM, RID: True; BJ: False)
    _g2_sum_norm = False     # g2 = g2 / sum(g2) in the main tau loop (BJ: True)
    _has_special = False     # tau = round(N/2) special Nyquist bin (BJ, RID: True)
    _tfr1_g = False          # tfr[0,icol] *= g[Lg] (ZAM, RID: True; BJ: False)
    _points_tau_limit = True # include tau in the points range min (BJ: True; CW: False)
    # Phase 5.1: g2-normalization mode for the Numba kernel.
    #   0 = none (ZAM), 1 = unconditional /sum(g2) (BJ/CW/GRD/Bud),
    #   2 = guarded /sum(g2) when sum>eps (RID family).
    _g2_norm_code = 0

    def __init__(self, signal, timestamps=None, n_fbins=None,
                 twindow=None, fwindow=None, **kwargs):
        # Capture raw signal before the base ravels it (cross case).
        self._raw_signal = np.asarray(signal)
        # Cross input (n, 2): the base would ravel it to 2n and build the
        # default windows from 2n. Pass the first column instead so the base
        # sets signal=x1, ts=arange(n), n_fbins=n and builds windows from n.
        base_signal = (self._raw_signal[:, 0]
                       if (self._raw_signal.ndim == 2
                           and self._raw_signal.shape[1] == 2)
                       else self._raw_signal)
        super(GeneralizedCohenBase, self).__init__(
            base_signal, timestamps=timestamps, n_fbins=n_fbins, fwindow=fwindow,
            **kwargs)
        if self.n_fbins is None:
            self.n_fbins = self.signal.shape[0]
        N = self.n_fbins
        # frequency window h (self.fwindow)
        if self.fwindow is None:
            hlength = int(np.floor(N / 4.0))
            hlength += 1 - (hlength % 2)
            self.fwindow = tftb_window(hlength, 'Hamming')
        self.fwindow = self.fwindow / self.fwindow[(self.fwindow.shape[0] - 1) // 2]
        # time window g (self.twindow)
        if twindow is None:
            glength = int(np.floor(N / 10.0))
            glength += 1 - (glength % 2)
            self.twindow = tftb_window(glength, 'Hamming')
        else:
            self.twindow = np.asarray(twindow, dtype=float).ravel()
        if self.twindow.shape[0] % 2 == 0:
            raise ValueError("twindow must have an odd length.")
        if self.fwindow.shape[0] % 2 == 0:
            raise ValueError("fwindow must have an odd length.")
        if self._g_center_norm:
            Lg = (self.twindow.shape[0] - 1) // 2
            self.twindow = self.twindow / self.twindow[Lg]

    def _kernel(self, tau, points):
        """Return the kernel weight for given tau and points (identity by default)."""
        return np.ones_like(points, dtype=float)

    def _normalize_g2(self, g2):
        """Normalize g2 (overridable). Default: unconditional division by sum.

        The reduced-interference family overrides this with a guarded division
        (MATLAB ``if (sum(g2)>eps), g2=g2/sum(g2); end;``).
        """
        return g2 / np.sum(g2)

    def _kernel_matrix(self):
        """Precompute ``K[tau, p+Lg] = g[Lg+p] * kernel(tau, p)`` (Phase 5.1).

        Called on the numpy side so the Numba kernel is a pure gather+dot with
        *bit-identical* weights to the numpy path (the per-class ``_kernel`` —
        CW exponential, GRD ``sinc``, Butterworth, RID Bessel/binomial/Hann/
        triangular — is evaluated once here with the same numpy calls the
        reference uses). Shape is ``(nyq+1, 2*Lg+1)`` with ``nyq = round(N/2)``.
        """
        N = self.n_fbins
        g = self.twindow
        Lg = (g.shape[0] - 1) // 2
        nyq = int(np.round(N / 2.0))
        K = np.zeros((nyq + 1, g.shape[0]), dtype=float)
        for tau in range(1, nyq + 1):
            # Match the numpy loop's point range exactly so the kernel is only
            # evaluated where it is defined:
            #   _points_tau_limit True  (BJ/ZAM/RID): |p| <= min(tau, Lg)
            #   _points_tau_limit False (CW/GRD/Bud): |p| <= Lg
            # (The binomial kernel, a True class, is only defined for |p| <=
            # tau, so over-filling would index out of its Pascal row.)
            m = min(tau, Lg) if self._points_tau_limit else Lg
            points = np.arange(-m, m + 1)
            K[tau, Lg + points] = g[Lg + points] * self._kernel(tau, points)
        return K

    def run(self):
        raw = self._raw_signal
        if raw.ndim == 2 and raw.shape[1] == 2:
            x1, x2, n = raw[:, 0], raw[:, 1], raw.shape[0]
        else:
            x1 = x2 = self.signal
            n = self.signal.shape[0]
        # MATLAB passes 1-based timestamps (t=1:N); convert to 0-based for the
        # algorithm. Default 0-based timestamps are used as-is.
        ts = self.ts
        if ts.ndim == 1 and ts.shape[0] == n and len(ts) > 0 and ts[0] == 1:
            ts = ts - 1
        N = self.n_fbins
        g, h = self.twindow, self.fwindow
        Lg = (g.shape[0] - 1) // 2
        Lh = (h.shape[0] - 1) // 2
        # Phase 5.1: Numba kernel (prange over time columns), auto-fallback to
        # the numpy loop below on any failure. Identity is enforced by the
        # test suite; the public API and return value are unchanged.
        if _accel_backend.get_backend("cohen_generic", n) == "numba":
            try:
                from tftb._accel import kernels as _k
                K = self._kernel_matrix()
                nyq = int(np.round(N / 2.0))
                self.tfr = _k.cohen_generic_numba(
                    x1, x2, g, h, K, N, ts.astype(np.int64),
                    self._tfr1_g, self._points_tau_limit, self._g2_norm_code,
                    self._has_special, nyq - 1, nyq)
                self.tfr = np.fft.fft(self.tfr, axis=0)
                if x1 is x2:
                    self.tfr = np.real(self.tfr)
                self.freqs = 0.5 * np.arange(self.n_fbins, dtype=float) / self.n_fbins
                return self.tfr, self.ts, self.freqs
            except Exception:
                _accel_backend.report_failure("cohen_generic")
        for icol in range(ts.shape[0]):
            t0 = int(ts[icol])
            taumax = min(t0 + Lg, n - t0 - 1 + Lg, int(np.round(N / 2.0)) - 1, Lh)
            if self._tfr1_g:
                self.tfr[0, icol] = g[Lg] * x1[t0] * np.conj(x2[t0])
            else:
                self.tfr[0, icol] = x1[t0] * np.conj(x2[t0])
            for tau in range(1, taumax + 1):
                if self._points_tau_limit:
                    p_lo = -min(tau, Lg, n - t0 - 1 - tau)
                    p_hi = min(tau, Lg, t0 - tau)
                else:
                    p_lo = -min(Lg, n - t0 - 1 - tau)
                    p_hi = min(Lg, t0 - tau)
                points = np.arange(p_lo, p_hi + 1)
                g2 = g[Lg + points] * self._kernel(tau, points)
                if self._g2_sum_norm:
                    g2 = self._normalize_g2(g2)
                R = np.sum(g2 * x1[t0 + tau - points] * np.conj(x2[t0 - tau - points]))
                self.tfr[tau, icol] = h[Lh + tau] * R
                R = np.sum(g2 * x1[t0 - tau - points] * np.conj(x2[t0 + tau - points]))
                self.tfr[N - tau, icol] = h[Lh - tau] * R
            if self._has_special:
                tau = int(np.round(N / 2.0))
                if (t0 + 1 <= n - tau) and (t0 + 1 >= tau + 1) and (tau <= Lh):
                    if self._points_tau_limit:
                        p_lo = -min(tau, Lg, n - t0 - 1 - tau)
                        p_hi = min(tau, Lg, t0 - tau)
                    else:
                        p_lo = -min(Lg, n - t0 - 1 - tau)
                        p_hi = min(Lg, t0 - tau)
                    points = np.arange(p_lo, p_hi + 1)
                    g2 = g[Lg + points] * self._kernel(tau, points)
                    g2 = g2 / np.sum(g2)
                    R1 = np.sum(g2 * x1[t0 + tau - points] * np.conj(x2[t0 - tau - points]))
                    R2 = np.sum(g2 * x1[t0 - tau - points] * np.conj(x2[t0 + tau - points]))
                    self.tfr[tau, icol] = 0.5 * (h[Lh + tau] * R1 + h[Lh - tau] * R2)
        self.tfr = np.fft.fft(self.tfr, axis=0)
        if x1 is x2:
            self.tfr = np.real(self.tfr)
        # raw 0 -> 0.5 frequency axis (unshifted FFT convention, like tfrwv)
        self.freqs = 0.5 * np.arange(self.n_fbins, dtype=float) / self.n_fbins
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, **kwargs):
        self.tfr = np.abs(self.tfr) ** 2
        self.tfr = self.tfr[:self.tfr.shape[0] // 2, :]
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(GeneralizedCohenBase, self).plot(kind=kind, **kwargs)


class BornJordanDistribution(GeneralizedCohenBase):
    """Born-Jordan time-frequency distribution.

    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.05, 0.3)[0] + fmlin(128, 0.15, 0.4)[0]
    >>> tfr, t, f = BornJordanDistribution(sig).run()
    """

    name = "born-jordan"
    _g_center_norm = False
    _g2_sum_norm = True
    _has_special = True
    _tfr1_g = False
    _g2_norm_code = 1


class ChoiWilliamsDistribution(GeneralizedCohenBase):
    """Choi-Williams time-frequency distribution.

    Identical to :class:`BornJordanDistribution` except the time-smoothing
    window is multiplied by the Choi-Williams kernel
    ``exp(-points^2 * sigma / (16 * tau^2))`` (MATLAB ``tfrcw.m``).

    :param sigma: Kernel width (default ``1.0``). Must be positive.
    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.05, 0.3)[0] + fmlin(128, 0.15, 0.4)[0]
    >>> tfr, t, f = ChoiWilliamsDistribution(sig).run()
    """

    name = "choi-williams"
    _g_center_norm = False
    _g2_sum_norm = True
    _has_special = True
    _tfr1_g = False
    _points_tau_limit = False
    _g2_norm_code = 1

    def __init__(self, signal, timestamps=None, n_fbins=None,
                 twindow=None, fwindow=None, sigma=1.0, **kwargs):
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        self.sigma = float(sigma)
        super(ChoiWilliamsDistribution, self).__init__(
            signal, timestamps=timestamps, n_fbins=n_fbins,
            twindow=twindow, fwindow=fwindow, **kwargs)

    def _kernel(self, tau, points):
        spreadfac = 16.0 / self.sigma
        return np.exp(-(points ** 2) / (spreadfac * tau ** 2))


class GeneralizedRectangularDistribution(GeneralizedCohenBase):
    """Generalized rectangular time-frequency distribution (MATLAB ``tfrgrd.m``).

    Same structure as :class:`ChoiWilliamsDistribution` but with the
    generalized-rectangular kernel
    ``sinc(2*rs*points*(1/(2*tau))^MoverN)`` (``sinc`` = normalized, i.e.
    ``sin(pi x)/(pi x)``).

    :param rs: Kernel width (default ``1.0``), must be positive.
    :param MoverN: Dissymmetry ratio (default ``1.0``), must be positive.
    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.05, 0.3)[0] + fmlin(128, 0.15, 0.4)[0]
    >>> tfr, t, f = GeneralizedRectangularDistribution(sig).run()
    """

    name = "generalized-rectangular"
    _g_center_norm = False
    _g2_sum_norm = True
    _has_special = True
    _tfr1_g = False
    _points_tau_limit = False
    _g2_norm_code = 1

    def __init__(self, signal, timestamps=None, n_fbins=None,
                 twindow=None, fwindow=None, rs=1.0, MoverN=1.0, **kwargs):
        if rs is None or rs <= 0:
            raise ValueError("rs must be strictly positive")
        if MoverN is None or MoverN <= 0:
            raise ValueError("MoverN must be strictly positive")
        self.rs = float(rs)
        self.MoverN = float(MoverN)
        super(GeneralizedRectangularDistribution, self).__init__(
            signal, timestamps=timestamps, n_fbins=n_fbins,
            twindow=twindow, fwindow=fwindow, **kwargs)

    def _kernel(self, tau, points):
        return np.sinc(2.0 * self.rs * points * (1.0 / (2.0 * tau)) ** self.MoverN)


class ButterworthDistribution(GeneralizedCohenBase):
    """Butterworth time-frequency distribution (MATLAB ``tfrbud.m``).

    Same structure as :class:`ChoiWilliamsDistribution` but with the
    Butterworth kernel ``exp(-abs(points)*sqrt(sigma)/(2*tau))``.

    :param sigma: Kernel width (default ``1.0``), must be positive.
    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.05, 0.3)[0] + fmlin(128, 0.15, 0.4)[0]
    >>> tfr, t, f = ButterworthDistribution(sig).run()
    """

    name = "butterworth"
    _g_center_norm = False
    _g2_sum_norm = True
    _has_special = True
    _tfr1_g = False
    _points_tau_limit = False
    _g2_norm_code = 1

    def __init__(self, signal, timestamps=None, n_fbins=None,
                 twindow=None, fwindow=None, sigma=1.0, **kwargs):
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        self.sigma = float(sigma)
        super(ButterworthDistribution, self).__init__(
            signal, timestamps=timestamps, n_fbins=n_fbins,
            twindow=twindow, fwindow=fwindow, **kwargs)

    def _kernel(self, tau, points):
        return np.exp(-np.abs(points) * np.sqrt(self.sigma) / (2.0 * tau))


class ReducedInterferenceBase(GeneralizedCohenBase):
    """Base for the reduced-interference distribution family (MATLAB
    ``tfrridb/ridbn/ridh/ridt.m``).

    All four variants share the same structure and knob set as
    :class:`ZaoAtlasMarksDistribution` (g center-normalized, ``tfr[0]`` scaled
    by ``g(Lg)``, points range includes ``tau``, Nyquist special bin), but use
    a *guarded* per-tau normalization:
    ``if (sum(g2) > eps), g2 = g2/sum(g2); end;`` (MATLAB). Subclasses supply
    the kernel.
    """

    name = "reduced-interference"
    _g_center_norm = True
    _g2_sum_norm = True
    _has_special = True
    _tfr1_g = True
    _points_tau_limit = True
    _g2_norm_code = 2

    def _normalize_g2(self, g2):
        """Guarded normalization (MATLAB ``if (sum(g2)>eps), g2=g2/sum(g2);``)."""
        s = np.sum(g2)
        if s > np.finfo(float).eps:
            return g2 / s
        return g2


class ReducedInterferenceDistributionBessel(ReducedInterferenceBase):
    """Reduced-interference distribution, Bessel kernel (MATLAB ``tfrridb.m``).

    Kernel: ``sqrt(1 - (points/tau)^2)``.
    """

    name = "reduced-interference-bessel"

    def _kernel(self, tau, points):
        return np.sqrt(1.0 - (points / tau) ** 2)


class ReducedInterferenceDistributionBinomial(ReducedInterferenceBase):
    """Reduced-interference distribution, binomial kernel (MATLAB
    ``tfrridbn.m``).

    Kernel: the normalized Pascal row ``Ker(tau+points+1)`` where
    ``Ker = C(2*tau, 0:2*tau)/sum`` (computed in log-space for stability).
    """

    name = "reduced-interference-binomial"

    @staticmethod
    def _pascal_row(tau):
        from scipy.special import loggamma
        k = np.arange(2 * tau + 1)
        logC = (loggamma(2 * tau + 1) - loggamma(k + 1)
                - loggamma(2 * tau - k + 1))
        c = np.exp(logC)
        return c / np.sum(c)

    def _kernel(self, tau, points):
        return self._pascal_row(tau)[tau + points]


class ReducedInterferenceDistributionHann(ReducedInterferenceBase):
    """Reduced-interference distribution, Hann kernel (MATLAB ``tfrridh.m``).

    Kernel: ``1 + cos(pi*points/tau)``.
    """

    name = "reduced-interference-hann"

    def _kernel(self, tau, points):
        return 1.0 + np.cos(np.pi * points / tau)


class ReducedInterferenceDistributionTriangular(ReducedInterferenceBase):
    """Reduced-interference distribution, triangular kernel (MATLAB
    ``tfrridt.m``).

    Kernel: ``1 - abs(points/tau)``.
    """

    name = "reduced-interference-triangular"

    def _kernel(self, tau, points):
        return 1.0 - np.abs(points / tau)


class MinimumMeanCrossEntropyDistribution(BaseTFRepresentation):
    """Minimum mean cross-entropy combination of spectrograms (MATLAB
    ``tfrmmce.m``, F. Auger 1995).

    Takes a matrix ``windows`` whose *columns* are frequency-smoothing windows
    (each of odd length, zero-padded to a common length). Each column is
    normalized to unit energy, and the representation is the geometric mean of
    the per-window spectrogram energies::

        tfr[j, t] = ( prod_h |STFT_h[j, t]|^2 )^(1/hcol)

    where ``STFT_h`` uses window column ``h`` (window value at the fixed center
    offset by ``tau``, i.e. ``windows[Lh + tau, h]``). The result is then
    energy-normalized so that ``sum(sum(tfr)) == sum(|x[t]|^2)``.

    ``windows`` must be a 2-D matrix with an odd number of rows and at least
    two columns.
    """

    name = "mmce"

    def __init__(self, signal, windows, timestamps=None, n_fbins=None):
        """Create a MinimumMeanCrossEntropyDistribution object.

        :param signal: Signal to be analysed.
        :param windows: 2-D matrix; each column is a frequency window (odd
            length, zero-padded to a common length). At least two columns.
        :param timestamps: Time instants (default ``np.arange(len(signal))``).
        :param n_fbins: Number of frequency bins (default ``len(signal)``).
        """
        super(MinimumMeanCrossEntropyDistribution, self).__init__(
            signal=signal, n_fbins=n_fbins, timestamps=timestamps, fwindow=None)
        self.windows = np.asarray(windows, dtype=float)
        if self.windows.ndim != 2:
            raise ValueError('windows must be a 2-D matrix (columns = windows)')
        if self.windows.shape[0] % 2 == 0:
            raise ValueError('windows must have an odd number of rows')
        if self.windows.shape[1] < 2:
            raise ValueError('windows must have at least two columns')
        # normalize each column to unit energy (MATLAB: h*diag(1./sqrt(sum(abs(h).^2))))
        self.windows = self.windows * (
            1.0 / np.sqrt(np.sum(np.abs(self.windows) ** 2, axis=0)))
        self.lh = (self.windows.shape[0] - 1) // 2
        self.hcol = self.windows.shape[1]

    def run(self):
        """Compute the MMCE combination of spectrograms."""
        N = self.n_fbins
        xrow = self.signal.shape[0]
        nmax = int(_matlab_round(N / 2.0)) - 1
        tfr = np.zeros((N, self.ts.shape[0]))
        for icol in range(self.ts.shape[0]):
            ti = int(self.ts[icol])  # 0-based
            tau = np.arange(-min([nmax, self.lh, ti]),
                            min([nmax, self.lh, xrow - 1 - ti]) + 1)
            indices = np.remainder(N + tau, N)
            slides = np.zeros((N, self.hcol), dtype=complex)
            for ih in range(self.hcol):
                slides[indices, ih] = (self.signal[ti + tau]
                                       * np.conj(self.windows[self.lh + tau, ih]))
            slides = np.abs(np.fft.fft(slides, axis=0)) ** 2
            tfr[:, icol] = np.prod(slides, axis=1) ** (1.0 / self.hcol)
        tfr = tfr * (np.sum(np.abs(self.signal[self.ts.astype(int)]) ** 2)
                     / np.sum(tfr))
        return tfr, self.ts, self.freqs


class ZaoAtlasMarksDistribution(GeneralizedCohenBase):
    """Zao-Atlas-Marks time-frequency distribution (MATLAB ``tfrzam.m``).

    A generalized Cohen-class distribution with a center-normalized time
    window (``g = g/g(Lg)``), no per-tau renormalization, no kernel, and no
    Nyquist special bin. The ``tfr[0]`` row is scaled by ``g(Lg)``.
    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.05, 0.3)[0] + fmlin(128, 0.15, 0.4)[0]
    >>> tfr, t, f = ZaoAtlasMarksDistribution(sig).run()
    """

    name = "zao-atlas-marks"
    _g_center_norm = True
    _g2_sum_norm = False
    _has_special = False
    _tfr1_g = True
    _points_tau_limit = True


class MargenauHillSpectrogramDistribution(BaseTFRepresentation):
    """Margenau-Hill-Spectrogram time-frequency distribution (MATLAB
    ``tfrmhs.m``).

    Product of two windowed transforms:
    ``tfr = real(fft(tfr_g) * conj(fft(tfr_h)))`` where ``tfr_g`` uses the
    time window ``g`` and ``tfr_h`` uses the frequency window ``h``.
    ``h`` is additionally renormalized by ``Kgh = sum(h .* conj(g))`` so the
    representation preserves signal energy.
    :Example:

    >>> from tftb.generators import fmlin
    >>> sig = fmlin(128, 0.1, 0.4)[0]
    >>> tfr, t, f = MargenauHillSpectrogramDistribution(sig).run()
    """

    name = "margenau-hill-spectrogram"

    def __init__(self, signal, timestamps=None, n_fbins=None,
                 twindow=None, fwindow=None, **kwargs):
        self._raw_signal = np.asarray(signal)
        # Cross input (n, 2): pass the first column so the base builds the
        # default windows from n (not 2n).
        base_signal = (self._raw_signal[:, 0]
                       if (self._raw_signal.ndim == 2
                           and self._raw_signal.shape[1] == 2)
                       else self._raw_signal)
        super(MargenauHillSpectrogramDistribution, self).__init__(
            base_signal, timestamps=timestamps, n_fbins=n_fbins, fwindow=fwindow,
            **kwargs)
        if self.n_fbins is None:
            self.n_fbins = self.signal.shape[0]
        N = self.n_fbins
        # time window g
        if twindow is None:
            glength = int(np.floor(N / 10.0))
            glength += 1 - (glength % 2)
            self.twindow = tftb_window(glength, 'Hamming')
        else:
            self.twindow = np.asarray(twindow, dtype=float).ravel()
        if self.twindow.shape[0] % 2 == 0:
            raise ValueError("twindow must have an odd length.")
        # frequency window h (base already normalized by center)
        if self.fwindow is None:
            hlength = int(np.floor(N / 4.0))
            hlength += 1 - (hlength % 2)
            self.fwindow = tftb_window(hlength, 'Hamming')
        if self.fwindow.shape[0] % 2 == 0:
            raise ValueError("fwindow must have an odd length.")
        # MHS renormalization: h = h / Kgh, Kgh = sum(h .* conj(g))
        Lgh = min((self.twindow.shape[0] - 1) // 2,
                  (self.fwindow.shape[0] - 1) // 2)
        points = np.arange(-Lgh, Lgh + 1)
        Lg = (self.twindow.shape[0] - 1) // 2
        Lh = (self.fwindow.shape[0] - 1) // 2
        Kgh = np.sum(self.fwindow[Lh + points] * np.conj(self.twindow[Lg + points]))
        self.fwindow = self.fwindow / Kgh

    def run(self):
        raw = self._raw_signal
        if raw.ndim == 2 and raw.shape[1] == 2:
            x1, x2, n = raw[:, 0], raw[:, 1], raw.shape[0]
        else:
            x1 = x2 = self.signal
            n = self.signal.shape[0]
        ts = self.ts
        if ts.ndim == 1 and ts.shape[0] == n and len(ts) > 0 and ts[0] == 1:
            ts = ts - 1
        N = self.n_fbins
        g, h = self.twindow, self.fwindow
        Lg = (g.shape[0] - 1) // 2
        Lh = (h.shape[0] - 1) // 2
        tfr_g = np.zeros((N, ts.shape[0]), dtype=complex)
        tfr_h = np.zeros((N, ts.shape[0]), dtype=complex)
        for icol in range(ts.shape[0]):
            t0 = int(ts[icol])
            # g-windowed: tau = -min(round(N/2)-1, Lg, t0) : min(round(N/2)-1, Lg, n-t0-1)
            tau_lo = -min(int(np.round(N / 2.0)) - 1, Lg, t0)
            tau_hi = min(int(np.round(N / 2.0)) - 1, Lg, n - t0 - 1)
            tau = np.arange(tau_lo, tau_hi + 1)
            indices = np.remainder(N + tau, N)
            tfr_g[indices, icol] = x1[t0 + tau] * np.conj(g[Lg + tau])
            # h-windowed
            tau_lo = -min(int(np.round(N / 2.0)) - 1, Lh, t0)
            tau_hi = min(int(np.round(N / 2.0)) - 1, Lh, n - t0 - 1)
            tau = np.arange(tau_lo, tau_hi + 1)
            indices = np.remainder(N + tau, N)
            tfr_h[indices, icol] = x2[t0 + tau] * np.conj(h[Lh + tau])
        self.tfr = np.real(np.fft.fft(tfr_g, axis=0)
                           * np.conj(np.fft.fft(tfr_h, axis=0)))
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, **kwargs):
        self.tfr = np.abs(self.tfr) ** 2
        self.tfr = self.tfr[:self.tfr.shape[0] // 2, :]
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(MargenauHillSpectrogramDistribution, self).plot(kind=kind, **kwargs)


class MargenauHillDistribution(BaseTFRepresentation):

    name = "margenau-hill"

    def run(self):
        for icol in range(self.ts.shape[0]):
            ti = self.ts[icol]
            tau = np.arange(-min((self.n_fbins - ti,
                                  self.signal.shape[0] - ti)) + 1, ti)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins)
            self.tfr[indices, icol] = self.signal[ti] * \
                np.conj(self.signal[ti - tau])

        self.tfr = np.real(np.fft.fft(self.tfr, axis=0))
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, sqmod=True, **kwargs):
        self.tfr = self.tfr[:(self.tfr.shape[0] // 2), :]
        if sqmod:
            self.tfr = np.abs(self.tfr) ** 2
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        extent = [0, self.ts.max(), 0, 0.5]
        super(MargenauHillDistribution, self).plot(kind=kind, extent=extent, **kwargs)


class PseudoMargenauHillDistribution(MargenauHillDistribution):

    name = "pseudo margenau-hill"

    def _make_window(self):
        hlength = int(np.floor(self.n_fbins / 4.0))
        if hlength % 2 == 0:
            hlength += 1
        # MATLAB tfrpmh uses tftb_window(hlength) (symmetric 'Hamming'), not
        # scipy.signal.hamming (periodic form) — see PseudoPageRepresentation.
        fwindow = tftb_window(hlength, 'Hamming')
        lh = (fwindow.shape[0] - 1) // 2
        return fwindow / fwindow[lh]

    def run(self):
        lh = (self.fwindow.shape[0] - 1) // 2
        xrow = self.signal.shape[0]
        # MATLAB tfrpmh (1-based ti = icol+1):
        #   tau = -min(round(N/2)-1, Lh, xrow-ti) : min(round(N/2)-1, Lh, ti-1)
        #   -> 0-based: -min(nyq-1, lh, xrow-1-icol) : min(nyq-1, lh, icol)
        #   signal conj index x(ti-tau) -> 0-based x[icol - tau] (the earlier
        #   port used x[icol-tau-1], a one-sample shift corrupting every N).
        nyq = int(_matlab_round(self.n_fbins / 2.0))
        for icol in range(self.ts.shape[0]):
            start = min([nyq - 1, lh, xrow - 1 - icol])
            end = min([nyq - 1, lh, icol])
            tau = np.arange(-start, end + 1).astype(int)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins).astype(int)
            self.tfr[indices, icol] = self.fwindow[lh + tau] * self.signal[icol] * \
                np.conj(self.signal[icol - tau])
        # MATLAB tfrpmh returns real(fft(tfr)); the base class does the same.
        self.tfr = np.real(np.fft.fft(self.tfr, axis=0))
        return self.tfr, self.ts, self.freqs


class WignerVilleDistribution(BaseTFRepresentation):

    name = "wigner-ville"

    def run(self):
        # MATLAB round (half-away-from-zero), not Python's banker's round: for
        # odd n_fbins with n_fbins/2 == k + 0.5 (N ≡ 1 mod 4, e.g. 121, 129)
        # the two differ by 1. The special Nyquist bin reads boundary samples
        # that amplitude windowing usually zeroes (so the golden cases pass
        # either way), but non-windowed signals make the convention live.
        tausec = int(_matlab_round(self.n_fbins / 2.0))
        winlength = tausec - 1
        taulens = np.min(np.c_[np.arange(self.signal.shape[0]),
                               self.signal.shape[0] - np.arange(self.signal.shape[0]) - 1,
                         winlength * np.ones(self.ts.shape)], axis=1)
        conj_signal = np.conj(self.signal)
        for icol in range(self.ts.shape[0]):
            taumax = taulens[icol]
            tau = np.arange(-taumax, taumax + 1).astype(int)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins).astype(int)
            self.tfr[indices, icol] = self.signal[icol + tau] * \
                conj_signal[icol - tau]
            if (icol <= self.signal.shape[0] - tausec - 1) and (icol >= tausec):
                self.tfr[tausec, icol] = 0.5 * (self.signal[icol + tausec] * \
                    conj_signal[icol - tausec] + \
                    self.signal[icol - tausec] * conj_signal[icol + tausec])
        self.tfr = np.fft.fft(self.tfr, axis=0)
        self.tfr = np.real(self.tfr)
        self.freqs = 0.5 * np.arange(self.n_fbins, dtype=float) / self.n_fbins
        return self.tfr, self.ts, self.freqs

    def plot(self, kind='cmap', threshold=0.05, sqmod=False, **kwargs):
        scale = kwargs.pop("scale", "linear")
        if scale == "log":
            maxi = np.amax(self.tfr)
            mini = max(np.amin(self.tfr), maxi * threshold)
            levels = np.logspace(np.log10(mini), np.log10(maxi), 65)
            kwargs['levels'] = levels
        if sqmod:
            self.tfr = np.abs(self.tfr) ** 2
        else:
            self.tfr = np.abs(self.tfr)
        _threshold = np.amax(self.tfr) * threshold
        self.tfr[self.tfr <= _threshold] = 0.0
        super(WignerVilleDistribution, self).plot(kind=kind, threshold=threshold,
                                                  **kwargs)


class PseudoWignerVilleDistribution(WignerVilleDistribution):

    name = "pseudo winger-ville"

    def run(self):
        lh = (self.fwindow.shape[0] - 1) // 2
        # MATLAB round (half-away-from-zero), not np.round (banker's): see
        # WignerVilleDistribution.run for why the convention matters at the
        # Nyquist bin for odd n_fbins (masked by the default short fwindow,
        # live for long user fwindows / non-windowed signals).
        nyq = int(_matlab_round(self.n_fbins / 2.0))
        for icol in range(self.ts.shape[0]):
            # taumax cap is round(nfb/2)-1 (matches WVD's winlength and MATLAB
            # tfrpwv.m line 42), NOT round(nfb/2): the latter double-counts the
            # Nyquist row (the main loop would also write row T via tau=+/-T),
            # which corrupts odd n_fbins.
            taumaxvals = (icol, self.signal.shape[0] - icol - 1, nyq - 1, lh)
            taumax = np.min(taumaxvals)
            tau = np.arange(-taumax, taumax + 1).astype(int)
            indices = np.remainder(self.n_fbins + tau, self.n_fbins).astype(int)
            self.tfr[indices, icol] = self.fwindow[lh + tau] * self.signal[icol + tau] * \
                np.conj(self.signal[icol - tau])
            tau = nyq
            if (icol <= self.signal.shape[0] - tau - 1) and (icol >= tau) and (tau <= lh):
                self.tfr[tau, icol] = 0.5 * (self.fwindow[lh + tau] * \
                    self.signal[icol + tau] * np.conj(self.signal[icol - tau]) + \
                    self.fwindow[lh - tau] * self.signal[icol - tau] * \
                    np.conj(self.signal[icol + tau]))

        self.tfr = np.fft.fft(self.tfr, axis=0)
        return np.real(self.tfr), self.ts, self.freqs

    def plot(self, **kwargs):
        if "kind" not in kwargs:
            kwargs['kind'] = "contour"
        super(PseudoWignerVilleDistribution, self).plot(**kwargs)


def smoothed_pseudo_wigner_ville(signal, timestamps=None, freq_bins=None,
                                 twindow=None, fwindow=None):
    """Smoothed Pseudo Wigner-Ville time-frequency distribution.
    :param signal: signal to be analyzed
    :param timestamps: time instants of the signal
    :param freq_bins: number of frequency bins
    :param twindow: time smoothing window
    :param fwindow: frequency smoothing window
    :type signal: array-like
    :type timestamps: array-like
    :type freq_bins: int
    :type twindow: array-like
    :type fwindow: array-like
    :return: Smoothed pseudo Wigner Ville distribution
    :rtype: array-like
    """
    if timestamps is None:
        timestamps = np.arange(signal.shape[0])
    if freq_bins is None:
        freq_bins = signal.shape[0]

    if fwindow is None:
        # Default MUST be the MATLAB window (tftb_window 'Hamming'), not
        # scipy.signal.hamming — the two are different windows (scipy is
        # periodic-style, MATLAB is symmetric), and MATLAB tfrspwv uses
        # tftb_window. Using scipy here corrupted every N (caught by the
        # spwv golden cases).
        winlength = freq_bins // 4
        winlength = winlength + 1 - (winlength % 2)
        fwindow = tftb_window(int(winlength), 'Hamming')
    elif fwindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')

    if twindow is None:
        timelength = freq_bins // 10
        timelength += 1 - (timelength % 2)
        twindow = tftb_window(int(timelength), 'Hamming')
    elif twindow.shape[0] % 2 == 0:
        raise ValueError('The smoothing fwindow must have an odd length.')

    tfr = np.zeros((freq_bins, timestamps.shape[0]), dtype=complex)
    lg = (twindow.shape[0] - 1) // 2
    lh = (fwindow.shape[0] - 1) // 2
    # Phase 5.1: Numba kernel (prange over time columns), auto-fallback.
    if _accel_backend.get_backend("spwv", signal.shape[0]) == "numba":
        try:
            from tftb._accel import kernels as _k
            _ts = np.asarray(timestamps, dtype=np.int64)
            _nyq = int(_matlab_round(freq_bins / 2.0))
            tfr = _k.spwv_numba(np.asarray(signal, dtype=np.complex128),
                                np.asarray(twindow, dtype=float),
                                np.asarray(fwindow, dtype=float),
                                freq_bins, _ts, _nyq - 1, _nyq)
            tfr = np.fft.fft(tfr, axis=0)
            return np.real(tfr)
        except Exception:
            _accel_backend.report_failure("spwv")
    # Index convention matches MATLAB tfrspwv (1-based ti_mat = python ti + 1):
    #   base column: points = -min(Lg, xrow-ti_mat) : min(Lg, ti_mat-1)
    #                -> 0-based signal[ti - points]
    #   tau column:  x(ti+tau-p) * conj(x(ti-tau-p)), window fwindow[lh+tau],
    #                written to row `tau` (MATLAB 1-based row 1+tau)
    # The earlier port used x(ti+tau-p-1) / row 1+tau — a one-sample signal
    # shift that corrupted every N (caught by the spwv golden cases; only the
    # reassigned variant, which is correct, had a golden before).
    n = signal.shape[0]
    for icol in range(timestamps.shape[0]):
        ti = timestamps[icol]
        taumax = min([ti + lg, n - 1 - ti + lg,
                      int(_matlab_round(freq_bins / 2.0)) - 1, lh])
        points = np.arange(-min([lg, n - 1 - ti]),
                           min([lg, ti]) + 1).astype(int)
        lg_points = (lg + points).astype(int)
        g2 = twindow[lg_points]
        g2 = g2 / np.sum(g2)
        signal_idx = (ti - points).astype(int)
        tfr[0, icol] = np.sum(g2 * signal[signal_idx] * np.conj(signal[signal_idx]))
        for tau in range(1, int(taumax) + 1):
            points = np.arange(-min([lg, n - 1 - ti - tau]),
                               min([lg, ti - tau]) + 1).astype(int)
            lg_points = (lg + points).astype(int)
            g2 = twindow[lg_points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points).astype(int)
            idx2 = (ti - tau - points).astype(int)
            R = np.sum(g2 * signal[idx1] * np.conj(signal[idx2]))
            tfr[tau, icol] = fwindow[lh + tau] * R
            R = np.sum(g2 * signal[idx2] * np.conj(signal[idx1]))
            tfr[freq_bins - tau, icol] = fwindow[lh - tau] * R
        tau = int(_matlab_round(freq_bins / 2.0))
        if (ti <= n - tau - 1) and (ti >= tau) and (tau <= lh):
            points = np.arange(-min([lg, n - 1 - ti - tau]),
                               min([lg, ti - tau]) + 1).astype(int)
            lg_points = (lg + points).astype(int)
            g2 = twindow[lg_points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points).astype(int)
            idx2 = (ti - tau - points).astype(int)
            _x = np.sum(g2 * signal[idx1] * np.conj(signal[idx2]))
            _x *= fwindow[lh + tau]
            _y = np.sum(g2 * signal[idx2] * np.conj(signal[idx1]))
            _y *= fwindow[lh - tau]
            tfr[tau, icol] = (_x + _y) * 0.5
    tfr = np.fft.fft(tfr, axis=0)
    return np.real(tfr)


if __name__ == '__main__':
    from tftb.generators import anapulse
    sig = anapulse(128)
    t = np.linspace(0, 1, 128)
    spec = WignerVilleDistribution(sig, timestamps=t)
    spec.run()
    spec.plot(kind="contour", scale="log")
