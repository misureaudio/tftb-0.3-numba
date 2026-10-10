# tftb — Time-Frequency Toolbox (Python port)
#
# Python port of the MATLAB Time-Frequency Toolbox
# (F. Auger, P. Flandrin, P. Gonçalves, O. Lemoine).
# Copyright (C) 2026 tftb Python port contributors.
#
# This file is part of tftb and is licensed under the GNU General Public
# License, version 2 or (at your option) any later version. See the
# COPYING file for the full license text.

"""Analysis utilities (Phase 4 Task 4.5).

Faithful ports of the ``tftb-0.3`` analysis utility family into a single
module.  Each function is a direct transcription of the corresponding
``.m`` file; the MATLAB ``integ``/``integ2d`` builtins (P. Goncalves) that
several of these rely on are reproduced as private helpers so the ports are
self-contained and match the oracle exactly.

* 4.5a ``momftfr`` / ``momttfr``  <- ``momftfr.m`` / ``momttfr.m``
* 4.5b ``margtfr``                <- ``margtfr.m``
* 4.5c ``holder``                 <- ``holder.m``
* 4.5d ``if2phase``               <- ``if2phase.m``
* 4.5e ``midpoint`` / ``plotsid`` <- ``midpoint.m`` / ``plotsid.m``
* 4.5f ``d2statio``               <- ``d2statio.m`` (uses a faithful
                                       ``_tfrspwv`` <- ``tfrspwv.m``)
* 4.5g ``tffilter`` / ``tfristft``<- ``tffilter.m`` / ``tfristft.m``

(``griffitc`` lives in ``tftb.generators``; ``tftb_window`` is already
ported in ``tftb.processing.utils``.)
"""
import numpy as np

from tftb.processing.utils import tftb_window


# ---------------------------------------------------------------------------
# MATLAB integ / integ2d helpers (P. Goncalves, 1995)
# ---------------------------------------------------------------------------
def _integ(y, x=None):
    """MATLAB ``integ(y, x)``: trapezoidal integral of each ROW of the matrix
    ``y`` (along its columns) with column spacing ``x``.  Returns one value
    per row (shape ``(M,)``).

    MATLAB source:  ``dy = y(:,1:N-1) + y(:,2:N);  dx = (x(2:N)-x(1:N-1))/2;
    som = dy*dx';``
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    M, N = y.shape
    if x is None:
        x = np.arange(1, N + 1, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != N:
        raise ValueError("Y must have as many columns as X")
    dy = y[:, :-1] + y[:, 1:]
    dx = (x[1:] - x[:-1]) / 2.0
    return dy @ dx


def _integ2d(mat, x=None, y=None):
    """MATLAB ``integ2d(mat, x, y)``: 2-D trapezoidal integral of ``mat`` over
    abscissa ``x`` (columns, uniform spacing ``x(2)-x(1)``) and ordinate
    ``y`` (rows, actual spacing).  Returns a scalar.

    MATLAB source:
    ``mat = (sum(mat.').' - mat(:,1)/2 - mat(:,N)/2) .* (x(2)-x(1));
    dmat = mat(1:M-1) + mat(2:M);  dy = (y(2:M)-y(1:M-1))/2;
    som = sum(dmat.*dy);``
    """
    mat = np.asarray(mat, dtype=complex)
    M, N = mat.shape
    if x is None:
        x = np.arange(1, N + 1, dtype=float)
    if y is None:
        y = np.arange(1, M + 1, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    rowsum = mat.sum(axis=1)
    mat1 = (rowsum - mat[:, 0] / 2.0 - mat[:, -1] / 2.0) * (x[1] - x[0])
    dmat = mat1[:-1] + mat1[1:]
    dy = (y[1:] - y[:-1]) / 2.0
    return np.sum(dmat * dy)


# ---------------------------------------------------------------------------
# ideal TFR (used by the moment reference tests)
# ---------------------------------------------------------------------------
def tfrideal(iflaws, t=None, N=None, trace=0):
    """Ideal TFR for given instantaneous frequency laws (port of
    ``tfrideal.m``).  ``[tfr, t, f] = tfrideal(iflaws, t, N, trace)`` places a
    unit impulse at the frequency bin nearest each IF-law value, for each time
    and each component.

    :param iflaws: ``(M, P)`` array — one column per component, values in
        ``[0, 0.5]`` (``NaN`` marks a component absent at that time).
    :param t: time instants (1-based, default ``1:M``).
    :param N: number of frequency bins (default ``M``).
    :param trace: progress flag (ignored).
    :return: ``tfr`` (shape ``(N, len(t))``), ``t``, ``f`` (length ``N``).
    """
    iflaws = np.asarray(iflaws, dtype=float)
    if iflaws.ndim == 1:
        iflaws = iflaws.reshape(-1, 1)
    ifrow, ifcol = iflaws.shape
    if t is None:
        t = np.arange(1, ifrow + 1)
    t = np.atleast_1d(np.asarray(t, dtype=int)).ravel()
    if N is None:
        N = ifrow
    tcol = t.shape[0]
    if np.any(iflaws > 0.5) or np.any(iflaws < 0):
        raise ValueError("The values of IFLAWS must be between 0 and 0.5")
    tfr = np.zeros((N, tcol), dtype=float)
    for icol in range(1, tcol + 1):
        ti = t[icol - 1]
        for fi in range(1, ifcol + 1):
            v = iflaws[ti - 1, fi - 1]
            if np.isnan(v):
                if fi - 1 < N:
                    tfr[fi - 1, icol - 1] = np.nan
            else:
                tfr[int(round(v * 2 * (N - 1))), icol - 1] = 1.0
    f = 0.5 * np.arange(N) / N
    return tfr, t, f


# ---------------------------------------------------------------------------
# 4.5a  time / frequency moments
# ---------------------------------------------------------------------------
def momftfr(tfr, tmin=None, tmax=None, time=None):
    """Frequency moments of a TFR (port of ``momftfr.m``).

    ``[tm, D2] = momftfr(tfr, tmin, tmax, time)`` computes, for each frequency
    row, the averaged time (1st moment) ``tm`` and the squared time duration
    (2nd moment) ``D2``, over the time columns ``tmin:tmax``.

    :param tfr: TFR of shape ``(M, N)``.
    :param tmin: smallest column (1-based, default 1).
    :param tmax: largest column (1-based, default N).
    :param time: true time instants (default ``tmin:tmax``).
    :return: ``tm`` (shape ``(M,)``), ``D2`` (shape ``(M,)``).
    """
    tfr = np.asarray(tfr, dtype=complex)
    M, N = tfr.shape
    if tmin is None:
        tmin = 1
        tmax = N
    elif tmax is None:
        tmax = N
    tmin = int(tmin)
    tmax = int(tmax)
    if time is None:
        time = np.arange(tmin, tmax + 1, dtype=float)
    else:
        time = np.asarray(time, dtype=float).ravel()
    if tmin > tmax or tmin <= 0 or tmax > N:
        raise ValueError("1<=TMIN<=TMAX<=Ncol")
    sub = tfr[:, tmin - 1:tmax]          # M x K
    E = np.sum(sub, axis=1)              # per frequency row (MATLAB sum(tfr.'))
    tm = (time @ sub.T) / E              # (M,)
    D2 = (time ** 2 @ sub.T) / E - tm ** 2
    return tm, D2


_TYPE2 = {'TFRWV', 'TFRPWV', 'TFRSPWV', 'TFRCW', 'TFRZAM', 'TFRBJ', 'TFRBUD',
          'TFRGRD', 'TFRRSPWV', 'TFRRPWV', 'TFRRIDB', 'TFRRIDH', 'TFRRIDT',
          'TFRASPW', 'TFRDFLA', 'TFRSPBK', 'TFRUNTAC', 'TFRUNTPA', 'TFRBERT',
          'TFRSCALO', 'TYPE2'}
_TYPE1 = {'TFRPMH', 'TFRRPMH', 'TFRSP', 'TFRRSP', 'TFRPPAGE', 'TFRRPPAG',
          'TFRMHS', 'TFRRGAB', 'TFRMH', 'TFRMMCE', 'TFRMSC', 'TFRRMSC',
          'TFRPAGE', 'TFRGABOR', 'TFRRI', 'TYPE1'}


def momttfr(tfr, method, fbmin=None, fbmax=None, freqs=None):
    """Time moments of a TFR (port of ``momttfr.m``).

    ``[fm, B2] = momttfr(tfr, method, fbmin, fbmax, freqs)`` computes, for
    each time column, the averaged frequency (1st moment) ``fm`` and the
    frequency band (2nd moment) ``B2``, over the frequency rows ``fbmin:fbmax``.

    :param tfr: TFR of shape ``(M, N)``.
    :param method: representation name (selects the TYPE1/TYPE2 frequency axis).
    :param fbmin: smallest frequency bin (1-based, default 1).
    :param fbmax: largest frequency bin (1-based, default M).
    :param freqs: true frequency of each bin (length ``fbmax-fbmin+1``); if
        omitted it is derived from ``method`` (TYPE1: ``-0.5..0.5`` axis,
        TYPE2: ``0..0.5`` axis).
    :return: ``fm`` (shape ``(N,)``), ``B2`` (shape ``(N,)``).
    """
    tfr = np.asarray(tfr, dtype=complex)
    M, N = tfr.shape
    if fbmin is None:
        fbmin = 1
        fbmax = M
    elif fbmax is None:
        fbmax = M
    fbmin = int(fbmin)
    fbmax = int(fbmax)
    if fbmin > fbmax or fbmin == 0 or fbmax > M:
        raise ValueError("1<=FBMIN<=FBMAX<=Nrow")
    method = str(method).upper()
    if method in _TYPE2:
        typertf = 'TYPE2'
    elif method in _TYPE1:
        typertf = 'TYPE1'
    else:
        raise ValueError("Unknown representation.")
    if freqs is None:
        if typertf == 'TYPE1':
            freqs = np.mod(np.arange(fbmin - 1, fbmax) / M + 0.5, 1.0) - 0.5
        else:
            freqs = 0.5 * np.arange(fbmin - 1, fbmax) / M
    else:
        freqs = np.asarray(freqs, dtype=float).ravel()
        if freqs.shape[0] != (fbmax - fbmin + 1):
            raise ValueError("FREQS must have FBMAX-FBMIN+1 elements")
    sub = tfr[fbmin - 1:fbmax, :]        # K x N
    E = np.sum(sub, axis=0)              # per time column (MATLAB sum(tfr(fbmin:fbmax,:)))
    fm = (freqs @ sub) / E               # (N,)
    B2 = (freqs ** 2 @ sub) / E - fm ** 2
    return fm, B2


# ---------------------------------------------------------------------------
# 4.5b  marginals + energy
# ---------------------------------------------------------------------------
def margtfr(tfr, t=None, f=None):
    """Marginals and energy of a TFR (port of ``margtfr.m``).

    ``[margt, margf, E] = margtfr(tfr, t, f)``: the time marginal, the
    frequency marginal, and the energy of the representation.

    :param tfr: TFR of shape ``(M, N)``.
    :param t: time samples (default ``1:N``).
    :param f: frequency samples (default ``1:M``), not necessarily uniform.
    :return: ``margt`` (shape ``(N,)``), ``margf`` (shape ``(M,)``), ``E``.
    """
    tfr = np.asarray(tfr, dtype=complex)
    M, N = tfr.shape
    if t is None:
        t = np.arange(1, N + 1, dtype=float)
        f = np.arange(1, M + 1, dtype=float)
    elif f is None:
        f = np.arange(1, M + 1, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    E = np.real(_integ2d(tfr, t, f) / M)
    margt = np.real(_integ(tfr.T, f) / M)     # integ(tfr.', f')
    margf = np.real(_integ(tfr, t) / N)       # integ(tfr, t)
    return margt, margf, E


# ---------------------------------------------------------------------------
# 4.5c  Holder exponent
# ---------------------------------------------------------------------------
def holder(tfr, f, n1=None, n2=None, t=None, pl=0):
    """Holder exponent through an affine TFR (port of ``holder.m``).

    ``h = holder(tfr, f, n1, n2, t, pl)`` estimates the Holder exponent from
    the log-log slope of the frequency marginal (global, ``t`` omitted) or of
    the TFR column at ``t`` (local).  ``h = (-slope - 1)/2``.

    :param tfr: affine TFR (magnitude taken).
    :param f: frequency values of the spectral analysis.
    :param n1: index of the minimum frequency for the regression (default 1).
    :param n2: index of the maximum frequency (default ``len(f)``).
    :param t: time instant(s) (1-based); if omitted the global estimate is
        returned, otherwise the local estimate(s).
    :param pl: plot flag (ignored in the port).
    :return: scalar (global or single ``t``) or array (vector ``t``).
    """
    tfr = np.abs(np.asarray(tfr, dtype=complex))
    f = np.asarray(f, dtype=float).ravel()
    M = f.shape[0]
    if n1 is None:
        n1 = 1
    if n2 is None:
        n2 = M
    n1 = int(n1)
    n2 = int(n2)
    lf = np.log(f[n1 - 1:n2])

    def _h_from(y):
        p = np.polyfit(lf, np.log(y), 1)
        return (-p[0] - 1) / 2.0

    if t is None:
        fmarg = tfr.mean(axis=1)              # mean(tfr.')'
        return _h_from(fmarg[n1 - 1:n2])
    t = np.atleast_1d(np.asarray(t, dtype=int)).ravel()
    if t.shape[0] == 1:
        return _h_from(tfr[n1 - 1:n2, t[0] - 1])
    h = np.zeros(t.shape[0])
    for j, k in enumerate(t):
        h[j] = _h_from(tfr[n1 - 1:n2, k - 1])
    return h


# ---------------------------------------------------------------------------
# 4.5d  phase from instantaneous frequency
# ---------------------------------------------------------------------------
def if2phase(iflaw):
    """Generate the phase from the instantaneous frequency (port of
    ``if2phase.m``).  ``phi(k) = 2*pi * integ(iflaw(1:k), 1:k)`` — the
    cumulative trapezoidal integral of the IF law.

    :param iflaw: instantaneous-frequency law (1-D).
    :return: phase (1-D, same length).
    """
    iflaw = np.asarray(iflaw).real.ravel()
    N = iflaw.shape[0]
    x = np.arange(1, N + 1, dtype=float)
    phi = np.zeros(N)
    for k in range(1, N + 1):
        phi[k - 1] = 2.0 * np.pi * np.sum(_integ(iflaw[:k], x[:k]))
    return phi


# ---------------------------------------------------------------------------
# 4.5e  interference-diagram midpoint + schematic plot
# ---------------------------------------------------------------------------
def midpoint(t1, f1, t2, f2, k):
    """Mid-point construction for the interference diagram (port of
    ``midpoint.m``).

    ``[ti, fi] = midpoint(t1, f1, t2, f2, k)`` gives the time-frequency
    coordinates of the interference term between the points ``(t1, f1)`` and
    ``(t2, f2)`` for a distribution in the affine class perfectly localized on
    power-law group delays ``tx(nu) = t0 + c*nu^(k-1)``.

    :param t1, f1: coordinates of the first point (row vectors, ``f > 0``).
    :param t2, f2: coordinates of the second point.
    :param k: power of the group-delay law (2=WV, 0.5=D-Flandrin, 0=Bertrand,
        -1=Unterberger, inf=Margenau-Hill-Rihaczek).
    :return: ``ti`` (time), ``fi`` (frequency).  For ``k=inf`` these are the
        stacked pairs ``[t1; t2]`` / ``[f2; f1]``.
    """
    t1 = np.atleast_1d(np.asarray(t1, dtype=float)).ravel()
    f1 = np.atleast_1d(np.asarray(f1, dtype=float)).ravel()
    t2 = np.atleast_1d(np.asarray(t2, dtype=float)).ravel()
    f2 = np.atleast_1d(np.asarray(f2, dtype=float)).ravel()
    n = t1.shape[0]
    if not (f1.shape[0] == n and f2.shape[0] == n and t2.shape[0] == n):
        raise ValueError("T1, T2, F1 and F2 must have the same size")
    if np.any(f1 <= 0) or np.any(f2 <= 0):
        raise ValueError("F1 and F2 must be >0")
    eps = np.finfo(float).eps

    if k == 2:
        fi = (f1 + f2) / 2.0
        ti = (t1 + t2) / 2.0
        return ti, fi
    if np.isinf(k):
        ti = np.vstack([t1, t2])
        fi = np.vstack([f2, f1])
        return ti, fi

    fi = np.zeros(n)
    ti = np.zeros(n)
    I = np.abs(f1 - f2) > np.sqrt(eps)
    if np.any(I):
        f1i, f2i = f1[I], f2[I]
        t1i, t2i = t1[I], t2[I]
        if k == 1:
            fi[I] = np.exp((f1i * (np.log(f1i) - 1) - f2i * (np.log(f2i) - 1))
                           / (f1i - f2i))
            ti[I] = ((t1i * f1i - t2i * f2i) / (f1i - f2i)
                     - (t1i - t2i) / (np.log(f1i) - np.log(f2i)))
        elif k == 0:
            fi[I] = (f1i - f2i) / (np.log(f1i) - np.log(f2i))
            ti[I] = ((t1i * f1i - t2i * f2i) / (f1i - f2i)
                     + f1i * f2i * (t2i - t1i) * (np.log(f1i) - np.log(f2i))
                     / (f2i - f1i) ** 2)
        else:
            t0 = (t1i * f2i ** (k - 1) - t2i * f1i ** (k - 1)) \
                / (f2i ** (k - 1) - f1i ** (k - 1))
            fi[I] = ((f1i ** k - f2i ** k) / (f1i - f2i) / k) ** (1.0 / (k - 1))
            ti[I] = t0 + (t2i - t1i) / (f2i ** (k - 1) - f1i ** (k - 1)) \
                * fi[I] ** (k - 1)
    return ti, fi


def plotsid(t, iflaws, k=2):
    """Schematic interference diagram of FM signals (port of ``plotsid.m``).

    Plots the IF laws (via :func:`tftb.processing.plotifl.plotifl`) together
    with the auto-term and cross-term interference points computed with
    :func:`midpoint`.  This is a plotting utility (the 0.3 test does not test
    it — it has no numeric output); the interference points are returned so
    they can be checked.

    :param t: time instants.
    :param iflaws: matrix of instantaneous frequencies (one column per
        component; ``NaN`` marks a component that is absent at a given time).
    :param k: distribution (2=WV, 0.5=D-Flandrin, 0=Bertrand, -1=Unterberger,
        inf=MHR).
    :return: list of ``(ti, fi)`` point-pairs (auto terms first, then crosses).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    iflaws = np.atleast_2d(np.asarray(iflaws, dtype=float))
    iflaws = np.where(np.isnan(iflaws), np.nan, iflaws)
    valid = iflaws[~np.isnan(iflaws)]
    if valid.size and (valid.min() < 0 or valid.max() > 0.5):
        raise ValueError("each element of IFLAWS must be between 0 and 0.5")

    nrow, ncol = iflaws.shape
    points = []
    for j in range(ncol):                       # auto-terms
        idx = np.where(~np.isnan(iflaws[:, j]))[0]
        for i in range(idx.shape[0] - 1):
            ta = np.full(idx.shape[0] - i - 1, t[idx[i]])
            fa = np.full(idx.shape[0] - i - 1, iflaws[idx[i], j])
            tb = t[idx[i + 1:]]
            fb = iflaws[idx[i + 1:], j]
            ti, fi = midpoint(ta, fa, tb, fb, k)
            points.append((ti, fi))
    for j1 in range(ncol):                      # cross-terms
        idx1 = np.where(~np.isnan(iflaws[:, j1]))[0]
        for j2 in range(j1 + 1, ncol):
            idx2 = np.where(~np.isnan(iflaws[:, j2]))[0]
            for i in range(idx1.shape[0]):
                ta = np.full(idx2.shape[0], t[idx1[i]])
                fa = np.full(idx2.shape[0], iflaws[idx1[i], j1])
                tb = t[idx2]
                fb = iflaws[idx2, j2]
                ti, fi = midpoint(ta, fa, tb, fb, k)
                points.append((ti, fi))
    return points


# ---------------------------------------------------------------------------
# 4.5f  distance to stationarity (uses a faithful tfrspwv)
# ---------------------------------------------------------------------------
def _tfrspwv(sig, t=None, N=None, g=None, h=None, trace=0):
    """Smoothed pseudo Wigner-Ville distribution (port of ``tfrspwv.m``,
    auto case).  Default time window ``g = tftb_window(floor(N/10) made odd)``
    and frequency window ``h = tftb_window(floor(N/4) made odd)`` normalized so
    its centre is 1 (matching MATLAB exactly — note this differs from the
    scipy-hamming default of the 0.2.0 ``smoothed_pseudo_wigner_ville``).
    """
    sig = np.asarray(sig, dtype=complex).ravel()
    xrow = sig.shape[0]
    if N is None:
        N = xrow
    if t is None:
        t = np.arange(1, xrow + 1)
    t = np.atleast_1d(np.asarray(t, dtype=int)).ravel()
    tcol = t.shape[0]
    hlength = N // 4
    hlength = hlength + 1 - (hlength % 2)
    glength = N // 10
    glength = glength + 1 - (glength % 2)
    if g is None:
        g = tftb_window(glength, 'Hamming')
    if h is None:
        h = tftb_window(hlength, 'Hamming')
    g = np.asarray(g, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    Lg = (g.shape[0] - 1) // 2
    Lh = (h.shape[0] - 1) // 2
    h = h / h[Lh]                               # h = h/h(Lh+1), centre = 1
    tfr = np.zeros((N, tcol), dtype=complex)
    for icol in range(1, tcol + 1):
        ti = t[icol - 1]
        taumax = int(min([ti + Lg - 1, xrow - ti + Lg, round(N / 2) - 1, Lh]))
        points = np.arange(-min([Lg, xrow - ti]), min([Lg, ti - 1]) + 1)
        g2 = g[Lg + points]
        g2 = g2 / np.sum(g2)
        idx = (ti - points) - 1
        tfr[0, icol - 1] = np.sum(g2 * sig[idx] * np.conj(sig[idx]))
        for tau in range(1, taumax + 1):
            points = np.arange(-min([Lg, xrow - ti - tau]),
                               min([Lg, ti - tau - 1]) + 1)
            g2 = g[Lg + points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points) - 1
            idx2 = (ti - tau - points) - 1
            R = np.sum(g2 * sig[idx1] * np.conj(sig[idx2]))
            tfr[tau, icol - 1] = h[Lh + tau] * R
            R = np.sum(g2 * sig[idx2] * np.conj(sig[idx1]))
            tfr[N - tau, icol - 1] = h[Lh - tau] * R
        tau = int(round(N / 2))
        if (ti <= xrow - tau) and (ti >= tau + 1) and (tau <= Lh):
            points = np.arange(-min([Lg, xrow - ti - tau]),
                               min([Lg, ti - tau - 1]) + 1)
            g2 = g[Lg + points]
            g2 = g2 / np.sum(g2)
            idx1 = (ti + tau - points) - 1
            idx2 = (ti - tau - points) - 1
            tfr[tau, icol - 1] = 0.5 * (
                h[Lh + tau] * np.sum(g2 * sig[idx1] * np.conj(sig[idx2]))
                + h[Lh - tau] * np.sum(g2 * sig[idx2] * np.conj(sig[idx1])))
    tfr = np.fft.fft(tfr, axis=0)
    tfr = np.real(tfr)
    f = 0.5 * np.arange(N) / N
    return tfr, t, f


def d2statio(sig):
    """Distance to stationarity (port of ``d2statio.m``).

    ``[d, f] = d2statio(sig)`` evaluates the distance of the signal to
    stationarity using the pseudo Wigner-Ville distribution: for each frequency
    bin the squared deviation of the TFR from its time-mean, normalized by
    ``||sig||^2`` and averaged over time.

    :param sig: signal (real or complex).
    :return: ``d`` (shape ``(N,)``), ``f`` (frequency bins, shape ``(N,)``).
    """
    sig = np.asarray(sig, dtype=complex).ravel()
    N = sig.shape[0]
    tfr, t, f = _tfrspwv(sig)
    rowmean = tfr.mean(axis=1, keepdims=True)
    d2 = ((tfr - rowmean) / np.linalg.norm(sig)) ** 2
    d = d2.mean(axis=1)
    return d, f


# ---------------------------------------------------------------------------
# 4.5g  time-frequency filtering + inverse STFT
# ---------------------------------------------------------------------------
def tffilter(tfr, x, t, trace=0):
    """Time-frequency filtering of a signal (port of ``tffilter.m``).

    ``y = tffilter(tfr, x, t, trace)`` filters the (analytic) signal ``x`` with
    the non-stationary filter whose Wigner-Ville distribution is ``tfr``
    (frequency axis 0.0..0.5).  Based on the Kozek-Hlawatsch comparative study
    of linear/non-linear time-frequency filters (1992).

    :param tfr: WVD of the filter (shape ``(N, NbPoints)``).
    :param x: input signal (analytic), length ``NbPoints``.
    :param t: time instants (1-based, one row).
    :param trace: progress flag (ignored).
    :return: filtered signal ``y`` (shape ``(len(t),)``).
    """
    tfr = np.asarray(tfr, dtype=complex)
    x = np.asarray(x, dtype=complex).ravel()
    t = np.atleast_1d(np.asarray(t, dtype=int)).ravel()
    N, NbPoints = tfr.shape
    if x.shape[0] != NbPoints:
        raise ValueError("tfr should have as many columns as X has rows.")
    if t.min() < 1 or t.max() > NbPoints:
        raise ValueError("the values of T must be between 1 and xrow.")
    tfr = np.fft.ifft(tfr, axis=0)
    y = np.zeros(t.shape[0], dtype=complex)
    for icol in range(t.shape[0]):
        ti = t[icol]
        lo = max(1, 2 - ti, ti - N // 2 + 1)
        hi = min(NbPoints, 2 * NbPoints - ti, ti + N // 2)
        for tj in range(lo, hi + 1):
            tmid = int(np.floor(0.5 * (ti + tj)))
            tdiff = int(np.floor(ti - tj))
            indice = (N + tdiff) % N + 1       # 1-based
            y[icol] += tfr[indice - 1, tmid - 1] * x[tj - 1]
    return y


def tfristft(tfr, t, h, trace=0):
    """Inverse short-time Fourier transform (port of ``tfristft.m``).

    ``[x, t] = tfristft(tfr, t, h, trace)`` computes the inverse STFT of a
    time-frequency decomposition (frequency axis -0.5..0.5) using the unit-
    energy frequency smoothing window ``h`` (odd length).  Used for
    time-frequency synthesis.

    :param tfr: TFR decomposition (complex, shape ``(N, NbPoints)``).
    :param t: time instants (1-based, must be every sample).
    :param h: frequency smoothing window (odd length, column).
    :param trace: progress flag (ignored).
    :return: reconstructed signal ``x`` (shape ``(len(t),)``), ``t``.
    """
    tfr = np.asarray(tfr, dtype=complex)
    t = np.atleast_1d(np.asarray(t, dtype=int)).ravel()
    h = np.asarray(h, dtype=complex).ravel()
    N, NbPoints = tfr.shape
    if h.shape[0] % 2 == 0:
        raise ValueError("H must be a smoothing window with odd length")
    Lh = (h.shape[0] - 1) // 2
    tcol = t.shape[0]
    if tcol != NbPoints:
        raise ValueError("tfr should have as many columns as t has rows.")
    Deltat = t[1:] - t[:-1]
    if not (Deltat.min() == 1 and Deltat.max() == 1):
        raise ValueError("The tfr must be computed at each time sample.")
    h = h / np.linalg.norm(h)
    tfr = np.fft.ifft(tfr, axis=0)
    x = np.zeros(tcol, dtype=complex)
    for icol in range(1, tcol + 1):
        lo = max(1, icol - N // 2, icol - Lh)
        hi = min(tcol, icol + N // 2, icol + Lh)
        valuestj = np.arange(lo, hi + 1)
        acc = 0.0 + 0.0j
        for tj in valuestj:
            tau = icol - tj
            indices = (N + tau) % N + 1        # 1-based
            acc += tfr[indices - 1, tj - 1] * h[Lh + tau]
        x[icol - 1] = acc / np.sum(np.abs(h[Lh + (icol - valuestj)]) ** 2)
    return x, t
